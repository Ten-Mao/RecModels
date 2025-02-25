import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from data.dataset import SeqRecDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from models.squencial_recommender.bert4rec import Bert4Rec
from util.evaluate import ndcg_at_k, recall_at_k
from util.logger import Logger
from util.util import ensure_dir, ensure_file
import pandas as pd



def parser_args():
    parser = argparse.ArgumentParser(description="SASRec")
    # global
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")

    # data
    parser.add_argument("--data_path", type=str, default="../data/")
    parser.add_argument("--dataset", choices=["Beauty2014", "Yelp"], default="Beauty2014")
    parser.add_argument("--num_workers", type=int, default=4)

    # model
    parser.add_argument("--emb_dropout", type=float, default=0.1)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--inner_dim", type=int, default=32)
    parser.add_argument("--ffn_activation", choices=["relu", "gelu"], default="gelu")
    parser.add_argument("--ffn_dropout", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--loss_type", choices=["bpr", "ce"], default="ce")

    # train and eval
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--valid_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--optimizer", choices=["adamw"], default="adamw")
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--scheduler_type", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--eval_step", type=int, default=1)
    parser.add_argument("--early_stop_step", type=int, default=10)

    # test
    parser.add_argument("--metrics", nargs="+", choices=["Recall", "NDCG"], default=["Recall", "NDCG"])
    parser.add_argument("--topk", nargs="+", type=int, default=[5, 10])

    # log, save and result
    parser.add_argument("--log_root_path", type=str, default="../log/")
    parser.add_argument("--save_root_path", type=str, default="../save/")
    parser.add_argument("--result_root_path", type=str, default="../result/")
    parser.add_argument(
        "--params_in_model_result", 
        nargs="+", 
        default=[
            "seed",
            "emb_dropout", 
            "d_model", 
            "n_heads", 
            "attn_dropout", 
            "inner_dim", 
            "ffn_activation", 
            "ffn_dropout", 
            "eps", 
            "num_layers", 
            "loss_type",

            "epochs",
            "train_batch_size",
            "valid_batch_size",
            "test_batch_size",
            "max_len",
            "mask_ratio",
            "lr",
            "weight_decay",
            "optimizer",
            "warmup_ratio",
            "scheduler_type",
            "eval_step",
            "early_stop_step",

            "time",
            "Recall@5",
            "NDCG@5",
            "Recall@10",
            "NDCG@10"
        ]
    )
    parser.add_argument(
        "--params_in_all_model_result", 
        nargs="+", 
        default=[
            "Model",
            "Recall@5",
            "NDCG@5",
            "Recall@10",
            "NDCG@10"
        ]
    )
    parser.add_argument("--selected_best_model_metric", choices=["Recall@5", "NDCG@5", "Recall@10", "NDCG@10"], default="Recall@5")
   

    return parser.parse_args()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_device(args):
    return torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")

def initial_dataLoader(args):

    datasets = {
        "train": SeqRecDataset(args.data_path, args.dataset, args.max_len, "train", mask_ratio=args.mask_ratio),
        "valid": SeqRecDataset(args.data_path, args.dataset, args.max_len, "valid", mask_ratio=args.mask_ratio),
        "test": SeqRecDataset(args.data_path, args.dataset, args.max_len, "test", mask_ratio=args.mask_ratio),
    }

    dataloaders = {
        "train": DataLoader(
            datasets["train"], 
            batch_size=args.train_batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True
        ),
        "valid": DataLoader(
            datasets["valid"], 
            batch_size=args.valid_batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True
        ),
        "test": DataLoader(
            datasets["test"], 
            batch_size=args.test_batch_size, 
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True
        )
    }

    return dataloaders["train"], dataloaders["valid"], dataloaders["test"], datasets["train"].num_items

def initial_model(args, device):
    model = Bert4Rec(
        args.num_items,
        args.emb_dropout,
        args.max_len,
        args.d_model,
        args.n_heads,
        args.attn_dropout,
        args.inner_dim,
        args.ffn_activation,
        args.ffn_dropout,
        args.eps,
        args.num_layers,
        loss_type=args.loss_type
    ).to(device)

    return model

def initial_optimizer_scheduler(args, model, batchnum_per_epoch):
    # optimizer 
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError("Invalid optimizer.")
    
    # scheduler
    total_steps = args.epochs * batchnum_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda_cos(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    def lr_lambda_linear(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lr_lambda_cos if args.scheduler_type == "cosine" else lr_lambda_linear
    )

    return optimizer, scheduler
    
def train_epoch(
    epoch,
    model,
    train_loader,
    device,
    optimizer,
    scheduler,
    logger
):
    model.train()
    total_loss = []
    logger.log(f"Epoch [{epoch + 1}] - Start Training")
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss.append(loss.item())

        if (step + 1) % 10 == 0:
            logger.log(
                f"Step [{step + 1}/{len(train_loader)}] - "
                f"Avg Loss: {np.mean(total_loss):.4f}, "
                f"Current lr: {scheduler.get_last_lr()[0]:.10f}"
            )
    
    logger.log(f"Epoch [{epoch + 1}] - Avg Loss: {np.mean(total_loss):.4f}")
            
@torch.no_grad()
def eval_epoch(
    epoch,
    model,
    valid_loader,
    device,
    logger
):
    model.eval()
    total_loss = []
    logger.log(f"Epoch [{epoch + 1}] - Start Evaluating")
    for step, batch in enumerate(valid_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(batch)
        total_loss.append(loss.item())
    valid_metric = np.mean(total_loss)
    logger.log(f"Epoch [{epoch + 1}] - Avg Evaluate Loss: {valid_metric:.4f}")

    return valid_metric

@torch.no_grad()
def test(
    model,
    test_loader,
    device,
    logger,
    args,
    model_result_path,
):
    model.eval()
    logger.log("Start Testing")

    result = {
        metric: {k: [] for k in args.topk}
        for metric in args.metrics
    }
    data_num = 0
    for step, batch in enumerate(test_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        scores = model.inference(batch)
        _, indices = torch.topk(scores, max(args.topk), dim=-1, largest=True, sorted=True)
        for metric in args.metrics:
            if metric == "Recall":
                for k in args.topk:
                    result[metric][k].append(
                        recall_at_k(indices + 1, batch["next_items"], k) * batch["next_items"].shape[0]
                    )
            elif metric == "NDCG":
                for k in args.topk:
                    result[metric][k].append(
                        ndcg_at_k(indices + 1, batch["next_items"], k) * batch["next_items"].shape[0]
                    )
            else:
                raise ValueError("Invalid metric.")
        data_num += batch["next_items"].shape[0]
    
    for metric in args.metrics:
        for k in args.topk:
            result[metric][k] = np.sum(result[metric][k]) / data_num
            logger.log(f"{metric}@{k}: {result[metric][k]:.4f}")
    

    # save model result
    model_result_df = pd.read_csv(model_result_path)
    new_line = {k: v for k, v in args.__dict__.items() if k in args.params_in_model_result}
    new_line.update({f"{metric}@{k}": v for metric, values in result.items() for k, v in values.items()})
    new_line_df = pd.DataFrame([new_line])
    if model_result_df.empty:
        model_result_df = new_line_df  # 直接赋值，避免 concat() 产生警告
    else:
        model_result_df = pd.concat([model_result_df, new_line_df], ignore_index=True)
    model_result_df.to_csv(model_result_path, index=True)


def run():
    args = parser_args()

    # set seed
    set_seed(args)

    # set device
    device = get_device(args)

    # initial dataLoader
    train_loader, valid_loader, test_loader, num_items = initial_dataLoader(args)
    args.num_items = num_items

    # initial model
    model = initial_model(args, device)

    # initial optimizer and scheduler
    optimizer, scheduler = initial_optimizer_scheduler(args, model, len(train_loader))

    # ensure the log, save and result path
    time_now = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    args.time = time_now
    log_file_path = os.path.join(args.log_root_path, args.dataset, f"SASRec-{time_now}.log")
    save_dir_path = os.path.join(args.save_root_path, args.dataset)
    save_file_path = os.path.join(args.save_root_path, args.dataset, f"SASRec-{time_now}.pth")
    model_result_file_path = os.path.join(args.result_root_path, args.dataset, f"SASRec.result.csv")
    # all_model_result_path = os.path.join(args.result_root_path, args.dataset, "All.result.csv")
    ensure_file(log_file_path)
    ensure_dir(save_dir_path)
    ensure_file(model_result_file_path, args.params_in_model_result)
    # ensure_file(all_model_result_path, args.params_in_all_model_result)

    # initial logger
    args_part_msg = {
        "seed": "# global \n",
        "data_path": "\n# data \n",
        "emb_dropout": "\n# model \n",
        "epochs": "\n# train and eval \n",
        "metrics": "\n# test \n",
        "log_root_path": "\n# log, save and result \n"
    }
    logger = Logger(log_file_path)
    logger.args_log(args, args_part_msg)


    # train and eval
    best_valid_metric = math.inf
    best_epoch = -1
    patience = 0
    for epoch in range(args.epochs):
        train_epoch(epoch, model, train_loader, device, optimizer, scheduler, logger)
        if epoch % args.eval_step == 0:
            valid_metric = eval_epoch(epoch, model, valid_loader, device, logger)
            if valid_metric < best_valid_metric:
                best_valid_metric = valid_metric
                best_epoch = epoch
                torch.save(model.state_dict(), save_file_path)
                logger.log(f"Save model at epoch [{epoch + 1}]")
            else:
                patience += 1
                logger.log(f"Patience: {patience}/{args.early_stop_step}")
                if patience >= args.early_stop_step:
                    logger.log(f"Early stop at epoch [{epoch + 1}]")
                    break
    logger.log(f"Best epoch: {best_epoch + 1}, Best valid metric: {best_valid_metric:.4f}")
    
    # test
    model.load_state_dict(torch.load(save_file_path, weights_only=True))
    test(model, test_loader, device, logger, args, model_result_file_path)


if __name__ == "__main__":
    run()