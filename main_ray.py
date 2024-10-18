import os
import tempfile
import logging
import torch
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict
from tqdm import tqdm
from ray import train
from ray.train import Checkpoint, ScalingConfig
from ray.train.torch import TorchTrainer
from accelerate import Accelerator
from src.model import PN_SSG, get_metric_names, get_loss,NCMClassfier
from src.config.config import config
from src.config.label import shapenet_label
from src.dataset import ShapeNetTest, ShapeNetTrain
from src.tokenizer import SimpleTokenizer
from collections import OrderedDict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAY_DEDUP_LOGS"] = '0'

# Logger设置
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
log_dir = 'log/'
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 平均值计算类
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

# 准确率计算函数
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

# 加载检查点
def load_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        logger.warning(f"No checkpoint found at {checkpoint_dir}")
        return None
    
    checkpoint = Checkpoint.from_directory(checkpoint_dir)
    with checkpoint.as_directory() as checkpoint_dir:
        model_state = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location=torch.device('cpu'))
        optimizer_state = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"), map_location=torch.device('cpu'))
        scaler_state = torch.load(os.path.join(checkpoint_dir, "scaler.pt"), map_location=torch.device('cpu'))
        scheduler_state = torch.load(os.path.join(checkpoint_dir, "scheduler.pt"), map_location=torch.device('cpu'))
        training_state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"), map_location=torch.device('cpu'))
    
    return {
        'epoch': training_state['epoch'],
        'best_acc': training_state['best_acc'],
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'scaler_state_dict': scaler_state,
        'scheduler_state_dict': scheduler_state
    }

# 保存检查点
def save_checkpoint(epoch, model, optimizer, scaler, best_acc, scheduler, output_dir, is_best=False, save_interval=10):
    if epoch % save_interval == 0 or is_best:
        with tempfile.TemporaryDirectory() as temp_dir:
            torch.save(model.state_dict(), os.path.join(temp_dir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(temp_dir, "optimizer.pt"))
            torch.save(scaler.state_dict(), os.path.join(temp_dir, "scaler.pt"))
            torch.save(scheduler.state_dict(), os.path.join(temp_dir, "scheduler.pt"))
            torch.save({'epoch': epoch, 'best_acc': best_acc}, os.path.join(temp_dir, "training_state.pt"))

            checkpoint = Checkpoint.from_directory(temp_dir)
            checkpoint_dir = checkpoint.to_directory(output_dir)
            logger.info(f"Checkpoint saved for epoch {epoch} at {checkpoint_dir}")

            if is_best:
                best_dir = os.path.join(output_dir, 'best_checkpoint')
                if not os.path.exists(best_dir):
                    os.makedirs(best_dir)
                torch.save({'epoch': epoch, 'best_acc': best_acc}, os.path.join(best_dir, 'best_checkpoint.pt'))
                logger.info(f"Best checkpoint saved at {best_dir}")

def train_func(config):
    os.chdir('/data1/refactor')
    accelerator = Accelerator()

    # 初始化模型、优化器、缩放器、调度器
    model = PN_SSG()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], betas=config["betas"],
                                  eps=config["eps"], weight_decay=config["wd"])
    scaler = amp.GradScaler()
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(config["train_loader"]))

    # 检查点恢复
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model_state = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location="cpu")
            optimizer_state = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"), map_location="cpu")
            scaler_state = torch.load(os.path.join(checkpoint_dir, "scaler.pt"), map_location="cpu")
            scheduler_state = torch.load(os.path.join(checkpoint_dir, "scheduler.pt"), map_location="cpu")
            training_state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"), map_location="cpu")

            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            scaler.load_state_dict(scaler_state)
            scheduler.load_state_dict(scheduler_state)
            start_epoch = training_state['epoch'] + 1
            best_acc = training_state['best_acc']
    else:
        start_epoch = 0
        best_acc = 0.0

    model, optimizer, scaler, scheduler = accelerator.prepare(model, optimizer, scaler, scheduler)

    metric_names = get_metric_names()
    metrics = OrderedDict([(name, AverageMeter(name, ':.2e')) for name in metric_names])

    for epoch in tqdm(range(start_epoch, config["num_epochs"])):
        config["train_loader"].dataset.shuffle()

        model.train()
        for data_iter, inputs in tqdm(enumerate(config["train_loader"])):
            pc = inputs[0].to(model.device)
            image = inputs[1].to(model.device)
            texts = inputs[2].to(model.device)
            all_label = inputs[3].to(model.device)
            main_label = torch.div(all_label, 5, rounding_mode='floor').to(model.device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(pc, texts, image)
                criterion = get_loss()
                loss_dict = criterion(main_label, all_label, outputs)
                loss = loss_dict['loss'] / config["update_freq"]

            scaler.scale(loss).backward()

            if (data_iter + 1) % config["update_freq"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            for k in loss_dict:
                metrics[k].update(loss_dict[k].item(), config["batch_size"])

        # 评估模型并保存检查点
        eval_metric = evaluate_model(model, config["val_loader"])
        is_best = eval_metric['top1_label_main'] > best_acc
        if is_best:
            best_acc = eval_metric['top1_label_main']

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(model.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(temp_checkpoint_dir, "optimizer.pt"))
            torch.save(scaler.state_dict(), os.path.join(temp_checkpoint_dir, "scaler.pt"))
            torch.save(scheduler.state_dict(), os.path.join(temp_checkpoint_dir, "scheduler.pt"))
            torch.save({'epoch': epoch, 'best_acc': best_acc}, os.path.join(temp_checkpoint_dir, "training_state.pt"))
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report({"epoch": epoch, "loss": loss.item(), **eval_metric}, checkpoint=checkpoint)

def evaluate_model(model, val_loader):
    os.chdir('/data1/refactor')
    model.eval()
    cur_device = model.device
    top1_label_main = AverageMeter('Acc@1_label_main', ':6.2f')
    top5_label_main = AverageMeter('Acc@5_label_main', ':6.2f')
    top1_label_all = AverageMeter('Acc@1_label_all', ':6.2f')
    top5_label_all = AverageMeter('Acc@5_label_all', ':6.2f')
    top1_text = AverageMeter('Acc@1_text', ':6.2f')
    top5_text = AverageMeter('Acc@5_text', ':6.2f')

    tokenizer = SimpleTokenizer()

    with torch.no_grad():
        text_features = []
        labels = shapenet_label[:25]
        for label in labels:
            text = f"a point cloud model of {label}."
            captions = [tokenizer(text)]
            texts = torch.stack(captions).to(cur_device)
            
            class_embeddings = model.module.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        
        text_features = torch.stack(text_features, dim=0).to(model.device)

        for pc, target, _ in val_loader:
            pc = pc.to(model.device)
            target = target.to(model.device)

            logits_text, logits_label_all, logits_label_main = model.module.encode_pc(pc)
            logits_text = logits_text @ text_features.t()

            # 更新主类别(main)的准确率
            acc1_label_main, acc5_label_main = accuracy(logits_label_main, target, topk=(1, 5))
            top1_label_main.update(acc1_label_main.item(), pc.size(0))
            top5_label_main.update(acc5_label_main.item(), pc.size(0))

            # 计算所有类别(all)的准确率
            logits_label_all_summed = logits_label_all.reshape(logits_label_all.size(0), 25, 5).sum(dim=2)
            acc1_label_all, acc5_label_all = accuracy(logits_label_all_summed, target, topk=(1, 5))
            top1_label_all.update(acc1_label_all.item(), pc.size(0))
            top5_label_all.update(acc5_label_all.item(), pc.size(0))

            # 更新文本类别的准确率
            acc1_text, acc5_text = accuracy(logits_text, target, topk=(1, 5))
            top1_text.update(acc1_text.item(), pc.size(0))
            top5_text.update(acc5_text.item(), pc.size(0))

    return {
        'top1_label_main': top1_label_main.avg,
        'top5_label_main': top5_label_main.avg,
        'top1_label_all': top1_label_all.avg,
        'top5_label_all': top5_label_all.avg,
        'top1_text': top1_text.avg,
        'top5_text': top5_text.avg
    }

def increament_test(pc_encoder):
    model = NCMClassfier(pc_encoder).to('cuda')
    acc = {'cos':[],'ed':[],'dp':[]}
    for i in range(1,8):
        train_dataset = ShapeNetTrain(task_id=i)
        
        for j in range(0,len(train_dataset),5):
            pc_datas,cates = train_dataset[j:j+5]
            model.train(cates,pc_datas)
        
        val_dataset = ShapeNetTest(task_id=i)
        
        cos_count,ed_count,dp_count = 0,0,0
        for pc, target, _ in tqdm(val_dataset,desc=f"task{i} training&evaluate"):
            pc = pc.to(model.device)
            target = target.to(model.device)
            pred = model.predict(pc)
            if pred['cos'] == target:
                cos_count += 1
            if pred['ed'] == target:
                ed_count += 1
            if pred['dp'] == target:
                dp_count += 1
        acc['cos'].append(cos_count / len(val_dataset)) 
        acc['ed'].append(ed_count / len(val_dataset)) 
        acc['dp'].append(dp_count / len(val_dataset)) 
    return acc       

def get_datasets():
    train_dataset = ShapeNetTrain(task_id=0)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True, drop_last=True)

    val_dataset = ShapeNetTest(task_id=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True, drop_last=False)

    return {"train_loader": train_loader, "val_loader": val_loader}

def main():
    cudnn.benchmark = True
    datasets = get_datasets()
    experiment_path = os.path.expanduser("~/ray_results/dl_restore_autoresume")
    if TorchTrainer.can_restore(experiment_path):
        trainer = TorchTrainer.restore(experiment_path, datasets=datasets)
        result = trainer.fit()
    else:
        trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config={
                "num_epochs": config.epochs,
                "train_loader": datasets["train_loader"],
                "val_loader": datasets["val_loader"],
                "lr": config.lr,
                "betas": config.betas,
                "eps": config.eps,
                "wd": config.wd,
                "update_freq": config.update_freq,
                "batch_size": config.batch_size,
            },
            scaling_config=ScalingConfig(num_workers=2, use_gpu=True, resources_per_worker={"GPU": 1}),
            run_config=train.RunConfig(
                storage_path=os.path.expanduser("~/ray_results"),
                name="dl_restore_autoresume",
            ),
        )
    result = trainer.fit()
    logger.info(f"Training completed with best accuracy: {result.metrics.get('best_acc', 'N/A')}")

if __name__ == '__main__':
    main()
    