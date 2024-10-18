from src.dataset import ShapeNetTest
from src.model import PN_SSG
from tqdm import tqdm
import torch 
from torch.nn.parameter import Parameter
checkpoint_path = "/data1/refactor/outputs/checkpoint_79.pth"
model1 = PN_SSG()
for name, _ in model1.named_parameters():
    print(name)
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model1_state_dict = checkpoint['model_state_dict']
model1_params = {param_name.replace('module.', ''): param for param_name, param in model1_state_dict.items()}
model1.load_state_dict(model1_params, strict=False)

# for name, param in model1.named_parameters():
#     if name not in model1_params:
#         continue

#     if isinstance(model1_params[name], Parameter):
#         param_new = model1_params[name].data
#     else:
#         param_new = model1_params[name]

#     param.requires_grad = False
#     print('load {} and freeze'.format(name))
#     param.data.copy_(param_new)

model2 = PN_SSG()
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model2_state_dict = checkpoint['model_state_dict']
model2_params = {param_name.replace('module.', ''): param for param_name, param in model2_state_dict.items()}
model2.load_state_dict(model2_params)

# for name, param in model2.named_parameters():
#     if name not in model2_params:
#         continue

#     if isinstance(model2_params[name], Parameter):
#         param_new = model2_params[name].data
#     else:
#         param_new = model2_params[name]

#     param.requires_grad = False
#     print('load {} and freeze'.format(name))
#     param.data.copy_(param_new)

data = ShapeNetTest(task_id=6)
all_pcs = [pc for pc, _, _ in data]
model1.eval()
model2.eval()
for j in tqdm(range(0, len(all_pcs), 16),desc="testing"):
    batch_pcs = torch.stack(all_pcs[j:j + 16])
    _,_,logits1 = model1.encode_pc(batch_pcs)
    _,_,logits2 = model2.encode_pc(batch_pcs)
    breakpoint()
    assert torch.equal(logits1.argmax(dim=-1),logits2.argmax(dim=-1))
print("finish")