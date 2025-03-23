import os
import socket
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.amp as amp
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parameter import Parameter
from tqdm import tqdm
from collections import OrderedDict
from src.model import PN_SSG, NCMClassfier
from src.config.label import shapenet_label
from src.dataset import ShapeNetTest, ShapeNetTrain
from src.config.config import config
from pprint import pformat
import datetime
from concurrent.futures import ThreadPoolExecutor

def increament_init(checkpoint_path, method, mode, device):
    os.makedirs("log", exist_ok=True)
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
                    #     
                    if preds["cos"][k]["cate"] == target:
                        cos_count += 1
                    if preds["ed"][k]["cate"] == target:
                        ed_count += 1
                    if preds["dp"][k]["cate"] == target:
                        dp_count += 1
    print(f"{cos_count / l} {ed_count / l} {dp_count / l}")
    return cos_count / l, ed_count / l, dp_count / l


def increament_train(i, model, train_batch_size=80, method="mean"):
    task_id = i if i != 0 else -2
    train_dataset = ShapeNetTrain(task_id=task_id)
    pc_datas, cates = [], []
    with torch.no_grad():
        with amp.autocast(enabled=True, device_type="cuda"):
            for j in tqdm(
                range(0, len(train_dataset), train_batch_size), desc="training"
            ):
                samples = [
                    train_dataset[k]
                    for k in range(j, min(j + train_batch_size, len(train_dataset)))
                ]
                pc_datas = [sample[0] for sample in samples]
                # if i == 0:
                #     cates = [
                #         torch.div(sample[1], 5, rounding_mode="floor").item()
                #         for sample in samples
                #     ]
                # else:
                cates = [sample[1] for sample in samples]

                model.train(cates, pc_datas, method=method)
                torch.cuda.empty_cache()
        model.train_last(method=method)
    torch.cuda.empty_cache()
    return model


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


def increament_process_single(checkpoint_path, method="mean",batch_size=80):
    # model_with, logger_with = increament_init(checkpoint_path, method, "with", "cuda:0")
    model_without, logger_without = increament_init(
        checkpoint_path, method, "without", "cuda:0"
    )
    # acc_with = {"cos": [], "ed": [], "dp": []}
    acc_without = {"cos": [], "ed": [], "dp": []}
    train_and_test_without(0, 7, model_without, batch_size, acc_without, method)
    # train_and_test_with(0, 7, model_with, batch_size, acc_with, method)

    # logger_with.info("Accuracies:\n")
    # logger_with.info(pformat(acc_with))

    logger_without.info("Accuracies:\n")
    logger_without.info(pformat(acc_without))


if __name__ == "__main__":
    increament_process_single(
        "save_model/best_checkpoint.pth", method="median",batch_size=20
    )
    # main_single_gpu()