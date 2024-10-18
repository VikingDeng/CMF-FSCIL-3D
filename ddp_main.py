import os
import socket
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.cuda.amp as amp
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parameter import Parameter
from tqdm import tqdm
from collections import OrderedDict
from src.model import PN_SSG, get_metric_names, get_loss, NCMClassfier
from src.config.label import shapenet_label
from src.dataset import ShapeNetTest, ShapeNetTrain
from src.tokenizer import SimpleTokenizer
from src.config.config import config
from src.loss import LosswithIMG
from pprint import pformat
import datetime
from concurrent.futures import ThreadPoolExecutor

os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
log_dir = "log/"
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def find_free_port():
    """自动找到一个可用的端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup(rank, world_size, seed, master_port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    logger.info(f"Rank {rank}: Initializing process group on port {master_port}...")

    dist.init_process_group(
        backend="nccl",
        init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}',
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=30),
    )

    torch.cuda.set_device(rank)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    logger.info(f"Rank {rank}/{world_size} process initialized.")


def train_ddp(rank, world_size, config, master_port):
    setup(rank, world_size, config.seed, master_port)

    model = PN_SSG().to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.wd,
    )
    scaler = amp.GradScaler(enabled=not config.disable_amp)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.epochs)

    start_epoch = 0
    best_acc = 0.0
    if config.resume:
        if os.path.isfile(config.resume):
            logger.info(f"Resuming from checkpoint: {config.resume}")
            checkpoint = torch.load(config.resume, map_location=f"cuda:{rank}")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_acc = checkpoint["best_acc"]
            logger.info(
                f"Checkpoint loaded. Resuming from epoch {start_epoch} with best accuracy {best_acc:.4f}."
            )
        else:
            logger.warning(
                f"No checkpoint found at {config.resume}. Starting from scratch."
            )
    else:
        logger.info("No resume checkpoint specified. Starting from scratch.")

    metric_names = get_metric_names()
    metrics = OrderedDict([(name, AverageMeter(name, ":.2e")) for name in metric_names])

    train_sampler = DistributedSampler(
        ShapeNetTrain(task_id=0), num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(
        ShapeNetTrain(task_id=0),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    val_loader = DataLoader(
        ShapeNetTest(task_id=0),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=True,
    )

    for epoch in range(start_epoch, config.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        logger.info(f"Epoch {epoch + 1}/{config.epochs} started.")

        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{config.epochs}",
            unit="batch",
            disable=rank != 0,
        ) as pbar:
            for data_iter, inputs in enumerate(train_loader):
                pc = inputs[0].to(rank)
                image = inputs[1].to(rank)
                texts = inputs[2].to(rank)
                all_label = inputs[3].to(rank)
                main_label = torch.div(all_label, 5, rounding_mode="floor").to(rank)

                with torch.cuda.amp.autocast(enabled=not config.disable_amp):
                    outputs = model(pc, texts, image)
                    criterion = LosswithIMG(logger=logger)
                    loss_dict = criterion(main_label, all_label, outputs)
                    loss = loss_dict["loss"] / config.update_freq

                if not torch.isfinite(loss):
                    logger.error(
                        f"Loss is not finite at batch {data_iter + 1}: {loss.item()}"
                    )
                    logger.error(f"PC: {pc}")
                    logger.error(f"Image: {image}")
                    logger.error(f"Texts: {texts}")
                    logger.error(f"All Labels: {all_label}")
                    logger.error(f"Main Labels: {main_label}")
                    raise ValueError(
                        f"Non-finite loss detected at batch {data_iter + 1}"
                    )

                scaler.scale(loss).backward()

                if (data_iter + 1) % config.update_freq == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                for k in loss_dict:
                    metrics[k].update(loss_dict[k].item(), config.batch_size)

                if rank == 0:
                    pbar.set_postfix({k: f"{v.avg:.4f}" for k, v in metrics.items()})
                    pbar.update(1)

        torch.cuda.empty_cache()

        if epoch % config.eval_freq == 0 or epoch == config.epochs - 1:
            eval_metric = evaluate_model(model, val_loader, world_size)
            is_best = eval_metric["top1_label_main"] > best_acc
            if is_best:
                best_acc = eval_metric["top1_label_main"]

            if rank == 0:
                checkpoint = {
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }
                torch.save(
                    checkpoint,
                    os.path.join(config.output_dir, f"checkpoint_{epoch}.pth"),
                )
                logger.info(f"Checkpoint saved at epoch {epoch}.")
                if is_best:
                    torch.save(
                        checkpoint,
                        os.path.join(config.output_dir, "best_checkpoint.pth"),
                    )
                    logger.info(
                        f"Best checkpoint updated at epoch {epoch} with accuracy {best_acc:.4f}."
                    )

            torch.cuda.empty_cache()
            model.train()


def all_reduce_metrics(metric_dict, world_size):
    for key in metric_dict:
        tensor = torch.tensor(metric_dict[key].avg).cuda()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        metric_dict[key].avg = tensor.item() / world_size


def evaluate_model(model, val_loader, world_size):
    model.eval()
    cur_device = model.device_ids[0]
    top1_label_main = AverageMeter("Acc@1_label_main", ":6.2f")
    top5_label_main = AverageMeter("Acc@5_label_main", ":6.2f")
    top1_label_all = AverageMeter("Acc@1_label_all", ":6.2f")
    top5_label_all = AverageMeter("Acc@5_label_all", ":6.2f")
    top1_text = AverageMeter("Acc@1_text", ":6.2f")
    top5_text = AverageMeter("Acc@5_text", ":6.2f")

    tokenizer = SimpleTokenizer()

    with torch.no_grad():
        text_features = []
        labels = shapenet_label[:25]
        for label in labels:
            text = f"a point cloud model of {label}."
            captions = [tokenizer(text)]
            texts = torch.stack(captions).to(cur_device)

            class_embeddings = model.module.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(
                dim=-1, keepdim=True
            )
            text_features.append(class_embeddings)

        text_features = torch.stack(text_features, dim=0).to(cur_device)

        for pc, target, _ in val_loader:
            pc = pc.to(cur_device)
            target = target.to(cur_device)

            logits_pc_text, logits_label_all, logits_label_main = (
                model.module.encode_pc(pc)
            )
            logits_text = logits_pc_text @ text_features.t()

            acc1_label_main, acc5_label_main = accuracy(
                logits_label_main, target, topk=(1, 5)
            )
            top1_label_main.update(acc1_label_main.item(), pc.size(0))
            top5_label_main.update(acc5_label_main.item(), pc.size(0))

            logits_label_all_summed = logits_label_all.reshape(
                logits_label_all.size(0), 25, 5
            ).sum(dim=2)
            acc1_label_all, acc5_label_all = accuracy(
                logits_label_all_summed, target, topk=(1, 5)
            )
            top1_label_all.update(acc1_label_all.item(), pc.size(0))
            top5_label_all.update(acc5_label_all.item(), pc.size(0))

            acc1_text, acc5_text = accuracy(logits_text, target, topk=(1, 5))
            top1_text.update(acc1_text.item(), pc.size(0))
            top5_text.update(acc5_text.item(), pc.size(0))

    all_reduce_metrics(
        {
            "top1_label_main": top1_label_main,
            "top5_label_main": top5_label_main,
            "top1_label_all": top1_label_all,
            "top5_label_all": top5_label_all,
            "top1_text": top1_text,
            "top5_text": top5_text,
        },
        world_size,
    )

    logger.info(
        f"Evaluation results - Acc@1_label_main: {top1_label_main.avg:.2f}, "
        f"Acc@5_label_main: {top5_label_main.avg:.2f}, "
        f"Acc@1_label_all: {top1_label_all.avg:.2f}, "
        f"Acc@5_label_all: {top5_label_all.avg:.2f}, "
        f"Acc@1_text: {top1_text.avg:.2f}, Acc@5_text: {top5_text.avg:.2f}"
    )

    torch.cuda.empty_cache()

    return {
        "top1_label_main": top1_label_main.avg,
        "top5_label_main": top5_label_main.avg,
        "top1_label_all": top1_label_all.avg,
        "top5_label_all": top5_label_all.avg,
        "top1_text": top1_text.avg,
        "top5_text": top5_text.avg,
    }


def increament_init(checkpoint_path, method, mode, device):
    def setup_logger(name, log_file, level=logging.INFO):
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger

    logger = setup_logger(mode, f"log/test_{mode}_ssg_{method}_0-6.log")
    model_ssg = PN_SSG()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model_state_dict = checkpoint["model_state_dict"]
    pretrain_model_ssg_params = {
        param_name.replace("module.", ""): param
        for param_name, param in model_state_dict.items()
    }
    model_ssg.load_state_dict(pretrain_model_ssg_params)
    model_ssg.eval()
    model_ssg.point_encoder.eval()
    model = NCMClassfier(model_ssg.point_encoder).to(device)
    return model, logger


def increament_test(task_id, model, batch_size=80, mode="with"):

    test_dataset = ShapeNetTest(task_id=task_id)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x
    )

    cos_count, ed_count, dp_count = 0, 0, 0
    l = len(test_dataset)
    model.encoder.eval()
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="testing"):
            batch_pcs = torch.stack([item[0] for item in batch]).to(model.device)
            batch_targets = [item[1] for item in batch]

            if len(batch_pcs) == 1:
                cos_count += 1
                ed_count += 1
                dp_count += 1
                continue

            preds = model.predict(batch_pcs)
            if mode == "with":
                _, _, logits_main = model.encoder(batch_pcs)
            for k, target in enumerate(batch_targets):
                if mode == "with" and target < 25:
                    if logits_main[k].argmax(-1).item() == target:
                        cos_count += 1
                        ed_count += 1
                        dp_count += 1
                else:
                    # if target >= 25:
                    #     keys_to_delete = [k for k in model.feature_center.keys() if k < 25]

                    #     # 在遍历结束后再删除这些键
                    #     for k in keys_to_delete:
                    #         del model.feature_center[k]
                    #     breakpoint()
                    if preds["cos"][k]["cate"] == target:
                        cos_count += 1
                    if preds["ed"][k]["cate"] == target:
                        ed_count += 1
                    if preds["dp"][k]["cate"] == target:
                        dp_count += 1

    return cos_count / l, ed_count / l, dp_count / l


def increament_train(i, model, train_batch_size=80, method="mean"):
    task_id = i if i != 0 else -2
    train_dataset = ShapeNetTrain(task_id=task_id)
    pc_datas, cates = [], []
    with torch.no_grad():
        with amp.autocast():
            for j in tqdm(
                range(0, len(train_dataset), train_batch_size), desc="training"
            ):
                samples = [
                    train_dataset[k]
                    for k in range(j, min(j + train_batch_size, len(train_dataset)))
                ]
                pc_datas = [sample[0] for sample in samples]
                if i == 0:
                    cates = [
                        torch.div(sample[1], 5, rounding_mode="floor").item()
                        for sample in samples
                    ]
                else:
                    cates = [sample[1] for sample in samples]
                model.train(cates, pc_datas, method=method)
                torch.cuda.empty_cache()
        model.train_last(method=method)
    torch.cuda.empty_cache()
    return model


def train_and_test_with(start, end, model_with, batch_size, acc_with, method):
    for i in range(start, end):
        if i != 0:
            model_with = increament_train(
                i, model_with, train_batch_size=batch_size, method=method
            )
        for lst, val in zip(
            (acc_with["cos"], acc_with["ed"], acc_with["dp"]),
            increament_test(i, model_with, batch_size=batch_size, mode="with"),
        ):
            lst.append(val)
        torch.cuda.empty_cache()


def train_and_test_without(start, end, model_without, batch_size, acc_without, method):
    for i in range(start, end):
        model_without = increament_train(
            i, model_without, train_batch_size=batch_size, method=method
        )
        for lst, val in zip(
            (acc_without["cos"], acc_without["ed"], acc_without["dp"]),
            increament_test(i, model_without, batch_size=batch_size, mode="without"),
        ):
            lst.append(val)
        torch.cuda.empty_cache()


def increament_process_multi(checkpoint_path, method="mean"):
    model_with, logger_with = increament_init(checkpoint_path, method, "with", "cuda:0")
    model_without, logger_without = increament_init(checkpoint_path, method, "without", "cuda:1")
    acc_with = {"cos": [], "ed": [], "dp": []}
    acc_without = {"cos": [], "ed": [], "dp": []}

    with torch.no_grad():
        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.submit(train_and_test_with, 0, 7, model_with, 80, acc_with, method)
            executor.submit(train_and_test_without, 0, 7, model_without, 80, acc_without, method)

    logger_with.info("Accuracies:\n")
    logger_with.info(pformat(acc_with))

    logger_without.info("Accuracies:\n")
    logger_without.info(pformat(acc_without))


def increament_process_single(checkpoint_path, method="mean"):
    model_with, logger_with = increament_init(checkpoint_path, method, "with", "cuda:0")
    model_without, logger_without = increament_init(
        checkpoint_path, method, "without", "cuda:1"
    )
    acc_with = {"cos": [], "ed": [], "dp": []}
    acc_without = {"cos": [], "ed": [], "dp": []}
    train_and_test_without(0, 7, model_without, 80, acc_without, method)
    # train_and_test_with(0, 7, model_with, 80, acc_with, method)

    logger_with.info("Accuracies:\n")
    logger_with.info(pformat(acc_with))

    logger_without.info("Accuracies:\n")
    logger_without.info(pformat(acc_without))


def test_if_continous():
    for i in range(7):
        task_id = i if i != 0 else -2
        train_dataset = ShapeNetTrain(task_id=task_id)
        last_cate = -1
        count = 0
        for j in tqdm(range(len(train_dataset)), desc=f"task{i}"):
            _, cate = train_dataset[j]
            if cate != last_cate:
                count += 1
                last_cate = cate
        assert count < 100


def main():
    world_size = len(config.gpu)
    os.makedirs(config.output_dir, exist_ok=True)

    logger.info("Starting distributed training")
    logger.info(f"World size: {world_size}, GPUs: {config.gpu}")

    if "MASTER_PORT" not in os.environ:
        master_port = find_free_port()
        os.environ["MASTER_PORT"] = str(master_port)
    else:
        master_port = os.environ["MASTER_PORT"]

    mp.spawn(
        train_ddp, args=(world_size, config, master_port), nprocs=world_size, join=True
    )

    logger.info("Distributed training completed")


if __name__ == "__main__":
    increament_process_single(
        "/data1/backup/refactor/outputs/checkpoint_79.pth", method="median"
    )

