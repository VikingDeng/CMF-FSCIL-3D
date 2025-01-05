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
from tqdm import tqdm
from collections import OrderedDict
from src.model import PN_SSG, get_metric_names, get_loss, NCMClassfier
from src.config.label import shapenet_label
from src.dataset import ShapeNetTest, ShapeNetTrain
from src.tokenizer import SimpleTokenizer
from src.config.config import config
from src.loss import LosswithIMG
import wandb

# Set NCCL environment variables
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"

def find_free_port():
    """Automatically find a free port for DDP."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def setup_ddp(rank, world_size, master_port):
    """Initialize the DDP process group."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(
        backend="nccl",
        init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}',
        rank=rank,
        world_size=world_size,
        timeout=torch.distributed.Timeout(600)
    )
    torch.cuda.set_device(rank)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = True

def cleanup_ddp():
    """Destroy the DDP process group."""
    dist.destroy_process_group()

def average_reduce(tensor, world_size):
    """Average a tensor across all processes."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor

def train_ddp(rank, world_size, master_port):
    """Main training loop for each DDP process."""
    setup_ddp(rank, world_size, master_port)

    # Initialize wandb only in the main process
    if rank == 0:
        wandb.init(
            project=config.wandb_project,
            config=config,
            save_code=True,
            name=config.run_name,
            resume=config.resume and "allow" or "never"
        )
        wandb.run.save()

    # Model setup
    model = PN_SSG().to(rank)
    model = DDP(model, device_ids=[rank])

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.wd,
    )
    scaler = amp.GradScaler(enabled=not config.disable_amp)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.epochs)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    if config.resume and os.path.isfile(config.resume):
        checkpoint = torch.load(config.resume, map_location=f"cuda:{rank}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        if rank == 0:
            wandb.restore(config.resume)

    # Data loaders
    train_dataset = ShapeNetTrain(task_id=0)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
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

    metric_names = get_metric_names()
    metrics = {name: 0.0 for name in metric_names}

    for epoch in range(start_epoch, config.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_metrics = {name: 0.0 for name in metric_names}

        if rank == 0:
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config.epochs}")

        for batch_idx, inputs in enumerate(train_loader):
            pc, image, texts, all_label = [x.to(rank) for x in inputs]
            main_label = (all_label // 5).to(rank)

            with amp.autocast(enabled=not config.disable_amp):
                outputs = model(pc, texts, image)
                criterion = LosswithIMG()
                loss_dict = criterion(main_label, all_label, outputs)
                loss = loss_dict["loss"] / config.update_freq

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.update_freq == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Accumulate metrics
            for k in loss_dict:
                epoch_metrics[k] += loss_dict[k].item()

            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({k: f"{v / (batch_idx + 1):.4f}" for k, v in epoch_metrics.items()})

        # Average metrics across all batches
        for k in epoch_metrics:
            epoch_metrics[k] /= len(train_loader)
            tensor = torch.tensor(epoch_metrics[k]).cuda()
            avg = average_reduce(tensor, world_size).item()
            metrics[k] = avg
            if rank == 0:
                wandb.log({f"train/{k}": avg}, step=epoch)

        # Validation
        if epoch % config.eval_freq == 0 or epoch == config.epochs - 1:
            if rank == 0:
                val_metrics = evaluate_model(model, val_loader, rank, world_size)
                # Log validation metrics
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)

                # Checkpointing
                is_best = val_metrics["top1_label_main"] > best_acc
                best_acc = max(val_metrics["top1_label_main"], best_acc)

                checkpoint = {
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }

                # Define checkpoint paths aligned with wandb
                checkpoint_dir = os.path.join(wandb.run.dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                latest_path = os.path.join(checkpoint_dir, "latest.pth")
                best_path = os.path.join(checkpoint_dir, "best.pth")

                # Save latest checkpoint
                torch.save(checkpoint, latest_path)
                wandb.save(latest_path)

                # Save best checkpoint
                if is_best:
                    torch.save(checkpoint, best_path)
                    wandb.save(best_path)
                    wandb.run.summary["best_acc"] = best_acc

                logger.info(f"Epoch {epoch+1} completed. Best Acc: {best_acc:.4f}")

        scheduler.step()

    cleanup_ddp()
    if rank == 0:
        wandb.finish()

def evaluate_model(model, val_loader, rank, world_size):
    """Evaluate the model and return metrics."""
    model.eval()
    device = model.device_ids[0]
    metrics = {
        "top1_label_main": 0.0,
        "top5_label_main": 0.0,
        "top1_label_all": 0.0,
        "top5_label_all": 0.0,
        "top1_text": 0.0,
        "top5_text": 0.0,
    }

    tokenizer = SimpleTokenizer()
    text_features = []
    labels = shapenet_label[:25]

    with torch.no_grad():
        for label in labels:
            text = f"a point cloud model of {label}."
            captions = [tokenizer(text)]
            texts = torch.stack(captions).to(device)
            class_embeddings = model.module.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)

        text_features = torch.stack(text_features, dim=0).to(device)

        for pc, target, _ in val_loader:
            pc = pc.to(device)
            target = target.to(device)

            logits_pc_text, logits_label_all, logits_label_main = model.module.encode_pc(pc)
            logits_text = logits_pc_text @ text_features.t()

            acc1_label_main, acc5_label_main = accuracy(logits_label_main, target, topk=(1, 5))
            acc1_label_all, acc5_label_all = accuracy(
                logits_label_all.reshape(logits_label_all.size(0), 25, 5).sum(dim=2), target, topk=(1, 5)
            )
            acc1_text, acc5_text = accuracy(logits_text, target, topk=(1, 5))

            metrics["top1_label_main"] += acc1_label_main.item() * pc.size(0)
            metrics["top5_label_main"] += acc5_label_main.item() * pc.size(0)
            metrics["top1_label_all"] += acc1_label_all.item() * pc.size(0)
            metrics["top5_label_all"] += acc5_label_all.item() * pc.size(0)
            metrics["top1_text"] += acc1_text.item() * pc.size(0)
            metrics["top5_text"] += acc5_text.item() * pc.size(0)

    # Average metrics
    for k in metrics:
        tensor = torch.tensor(metrics[k]).cuda()
        avg = average_reduce(tensor, world_size).item()
        metrics[k] = avg / len(val_loader.dataset)

    return metrics

def accuracy(output, target, topk=(1,)):
    """Compute the accuracy over the k top predictions for the specified values of k."""
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

def main():
    world_size = len(config.gpu)
    os.makedirs(config.output_dir, exist_ok=True)

    if "MASTER_PORT" not in os.environ:
        master_port = find_free_port()
    else:
        master_port = int(os.environ["MASTER_PORT"])

    mp.spawn(
        train_ddp,
        args=(world_size, master_port),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
