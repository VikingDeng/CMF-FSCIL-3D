from .common import *

from collections import OrderedDict
import timm
from models.pointnet2.pointnet2 import Pointnet2_Ssg
from easydict import EasyDict
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import mahalanobis
import heapq


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class Model_with_Image(nn.Module):
    def __init__(self, point_encoder, **kwargs):

        super().__init__()
        kwargs = EasyDict(kwargs)
        self.context_length = kwargs.context_length
        self.vision_width = kwargs.vision_width
        self.visual = kwargs.vision_model

        self.transformer = Transformer(
            width=kwargs.transformer_width,
            layers=kwargs.transformer_layers,
            heads=kwargs.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = kwargs.vocab_size
        self.token_embedding = nn.Embedding(kwargs.vocab_size, kwargs.transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, kwargs.transformer_width)
        )
        self.ln_final = LayerNorm(kwargs.transformer_width)

        self.image_projection = nn.Parameter(
            torch.empty(kwargs.vision_width, kwargs.embed_dim)
        )
        self.text_projection = nn.Parameter(
            torch.empty(kwargs.transformer_width, kwargs.embed_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self.point_encoder = point_encoder

        self.pc_projection = nn.Parameter(torch.empty(kwargs.pc_feat_dims, 512))
        nn.init.normal_(self.pc_projection, std=512**-0.5)

    def encode_image(self, image):
        x = self.visual(image)
        x = x @ self.image_projection

        return x

    def encode_text(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def build_attention_mask(self):

        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.image_projection, std=self.vision_width**-0.5)
        nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def encode_pc(self, pc):
        pc_feat, pc_all, pc_main = self.point_encoder(pc)
        pc_embed = pc_feat @ self.pc_projection.to(pc.device)
        return pc_embed, pc_all, pc_main

    def forward(self, pc, text, image=None):
        text_embed_all = []
        for i in range(text.shape[0]):
            text_for_one_sample = text[i]
            text_embed = self.encode_text(text_for_one_sample)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed = text_embed.mean(dim=0)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
            text_embed_all.append(text_embed)

        text_embed_all = torch.stack(text_embed_all)
        pc_embed, pc_all, pc_main = self.encode_pc(pc)
        if image is not None:
            image_embed = self.encode_image(image)
            return {
                "text_embed": text_embed_all,
                "pc_embed": pc_embed,
                "pc_all": pc_all,
                "pc_main": pc_main,
                "image_embed": image_embed,
                "logit_scale": self.logit_scale.exp(),
            }

        else:
            return {
                "text_embed": text_embed_all,
                "pc_embed": pc_embed,
                "pc_all": pc_all,
                "pc_main": pc_main,
                "logit_scale": self.logit_scale.exp(),
            }


class IncrementalMedianCalculator:
    def __init__(self, num_dimensions=256):

        self.min_heaps = [[] for _ in range(num_dimensions)]
        self.max_heaps = [[] for _ in range(num_dimensions)]

    def add_number(self, num):

        for i in range(len(num)):
            if len(self.max_heaps[i]) == 0 or num[i] <= -self.max_heaps[i][0]:
                heapq.heappush(self.max_heaps[i], -num[i])
            else:
                heapq.heappush(self.min_heaps[i], num[i])

            if len(self.max_heaps[i]) > len(self.min_heaps[i]) + 1:
                heapq.heappush(self.min_heaps[i], -heapq.heappop(self.max_heaps[i]))
            elif len(self.min_heaps[i]) > len(self.max_heaps[i]):
                heapq.heappush(self.max_heaps[i], -heapq.heappop(self.min_heaps[i]))

    def get_median(self):

        medians = []
        for i in range(len(self.min_heaps)):
            if len(self.max_heaps[i]) == len(self.min_heaps[i]):
                median = (-self.max_heaps[i][0] + self.min_heaps[i][0]) / 2.0
            else:
                median = -self.max_heaps[i][0]
            medians.append(median)
        return np.array(medians)


class NCMClassfier(nn.Module):
    def __init__(self, pc_encoder):
        super().__init__()
        self.encoder = pc_encoder
        self.feature_center = {}
        self.cate_num = {}
        self.total_weight = defaultdict(float)
        self.device = torch.device("cpu")
        self.median_calculators = defaultdict(IncrementalMedianCalculator)
        self.method = "mean"

    def forward(self, x):
        pass

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = args[0] if args else kwargs.get("device", None)
        if isinstance(device, torch.device):
            self.device = device
        elif isinstance(device, str):
            self.device = torch.device(device)
        return self

    def train(self, category, pc_datas, weights=None, method="mean"):
        self.method = method
        pc_datas = [pc.to(self.device) for pc in pc_datas]
        if len(pc_datas) == 1:
            return
        pc_embed, _, _ = self.encoder(torch.stack(pc_datas, dim=0))
        for i, (feature, cat) in enumerate(zip(pc_embed, category)):
            if method == "mean":
                self._update_mean(cat, feature)
            elif method == "weighted_mean" and weights is not None:
                self._update_weighted_mean(cat, feature, weights[i : i + 1])
            elif method == "median":
                self.median_calculators[cat].add_number(feature.detach().cpu().numpy())
        torch.cuda.empty_cache()

    def _update_mean(self, label, features):
        if label not in self.feature_center:
            self.feature_center[label] = torch.zeros(features.shape).to(self.device)
            self.cate_num[label] = 0
        self.feature_center[label] += features
        self.cate_num[label] += 1

    def _update_weighted_mean(self, label, features, weights):
        if label not in self.feature_center:
            self.feature_center[label] = torch.zeros(features.shape).to(self.device)
            self.total_weight[label] = 0
        self.feature_center[label] += features * weights[0]
        self.total_weight[label] += weights[0]

    def train_last(self, method="mean"):
        if method == "mean":
            for cate in self.feature_center:
                self.feature_center[cate] /= self.cate_num[cate]
        elif method == "weighted_mean":
            for cate in self.feature_center:
                self.feature_center[cate] /= self.total_weight[cate]
        elif method == "median":
            for label in self.median_calculators:
                self.feature_center[label] = torch.tensor(
                    np.array([self.median_calculators[label].get_median()])
                ).to(self.device)


    def predict(self, pc_data):
        pc_emb, _, _ = self.encoder(pc_data)
        batch_size = pc_emb.size(0)

        results = {"cos": [], "ed": [], "dp": []}
        for i in range(batch_size):
            max_cos = {"val": -1, "cate": -1}
            min_ed = {"val": float("inf"), "cate": -1}
            max_dp = {"val": float("-inf"), "cate": -1}
            emb = pc_emb[i]
            emb = emb.to(dtype=float)
            for cate, center in self.feature_center.items():
                if center.ndim == 2:
                    center = center.squeeze(0)
                center = center.to(dtype=float)
                cos = NCMClassfier.cosine_similarity(emb, center)
                if cos > max_cos["val"]:
                    max_cos["val"] = cos
                    max_cos["cate"] = cate

                ed = torch.norm(emb - center)
                if ed < min_ed["val"]:
                    min_ed["val"] = ed
                    min_ed["cate"] = cate

                dp = torch.dot(emb, center)
                if dp > max_dp["val"]:
                    max_dp["val"] = dp
                    max_dp["cate"] = cate

            results["cos"].append(max_cos)
            results["ed"].append(min_ed)
            results["dp"].append(max_dp)

        return results

    @staticmethod
    def cosine_similarity(a, b):
        return torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)
        ).item()


# def get_loss():
#     return loss.LosswithIMG()


# def get_metric_names():
#     return ["loss", "pc_image_acc", "pc_text_acc", "pc_all_acc", "pc_main_acc"]


def PN_SSG():
    vision_model = timm.create_model("vit_base_patch16_224", num_classes=0)

    point_encoder = Pointnet2_Ssg()
    pc_feat_dims = 256

    model = Model_with_Image(
        embed_dim=512,
        vision_width=768,
        point_encoder=point_encoder,
        vision_model=vision_model,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        pc_feat_dims=pc_feat_dims,
    )

    # pretrain_slip_model = torch.load(
    #     "data/initialize_models/slip_base_100ep.pt", map_location=torch.device("cpu")
    # )
    # pretrain_slip_model_params = pretrain_slip_model["state_dict"]
    # pretrain_slip_model_params = {
    #     param_name.replace("module.", ""): param
    #     for param_name, param in pretrain_slip_model_params.items()
    # }

    # for name, param in model.named_parameters():
    #     if name not in pretrain_slip_model_params:
    #         continue

    #     if isinstance(pretrain_slip_model_params[name], Parameter):
    #         param_new = pretrain_slip_model_params[name].data
    #     else:
    #         param_new = pretrain_slip_model_params[name]

    #     param.requires_grad = False
    #     print("load {} and freeze".format(name))
    #     param.data.copy_(param_new)

    return model
