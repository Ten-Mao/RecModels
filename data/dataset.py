import os
from typing import Literal
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random

class SeqRecDataset(Dataset):

    def __init__(self, data_root_path, dataset, max_len, mode:Literal["train", "valid", "test"], mask_ratio=0.15):
        self.data_root_path = data_root_path
        self.dataset = dataset
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        self.max_mask_len = int(max_len * mask_ratio)
        self.mode = mode

        self.inter_path = os.path.join(data_root_path, dataset, f"{dataset}.inter.csv")
        assert os.path.exists(self.inter_path), f"Inter file not found in {self.inter_path}"

        self.inters = self._load_inter(self.inter_path)
        self.num_items = self.get_item_num()
        self.num_users = self.get_user_num()


        if self.mode == "train":
            self.inter_data = self._process_train_data()
        elif self.mode == "valid":
            self.inter_data = self._process_valid_data()
        elif self.mode == "test":
            self.inter_data = self._process_test_data()
    
    def __len__(self):
        _len = len(self.inter_data)
        return _len
    
    def __getitem__(self, idx):
        sample = self.inter_data[idx]
        return sample

    def _load_inter(self, inter_path):
        inters = {}
        inter_df = pd.read_csv(inter_path)
        for row in inter_df.itertuples():
            user_id = getattr(row, "user_id")
            item_id = getattr(row, "item_id")
            if user_id not in inters:
                inters[user_id] = []
            inters[user_id].append(item_id)
        return inters

    def _process_train_data(self):
        inter_data = []
        for user_id, items in self.inters.items():
            items = items[:-2]
            items = [item + 1 for item in items] # 0 for padding
            for i in range(1, len(items)):
                seq = items[:i]
                target = items[i]
                if len(seq) < self.max_len:
                    seq = seq + [0] * (self.max_len - len(seq))
                else:
                    seq = seq[-self.max_len:]
                neg_target = random.choice(range(self.num_items)) + 1

                mask_seq_items_index = random.sample(range(self.max_len), self.max_mask_len)
                masked_seq = seq.copy()
                for item in mask_seq_items_index:
                    masked_seq[item] = self.num_items + 1
                mask_seq_items = []
                for item in mask_seq_items_index:
                    mask_seq_items.append(seq[item])
                neg_mask_seq_items = []
                for item in mask_seq_items_index:
                    neg_mask_seq_items.append(random.choice(range(self.num_items)) + 1)
                sample = {
                    "user_seqs": np.array(user_id),
                    "his_seqs": np.array(seq),
                    "next_items": np.array(target),
                    "next_neg_items": np.array(neg_target),

                    "masked_his_seqs": np.array(masked_seq),
                    "mask_indices": np.array(mask_seq_items_index),
                    "mask_items": np.array(mask_seq_items),
                    "mask_neg_items": np.array(neg_mask_seq_items)
                }
                inter_data.append(sample)
        return inter_data
    
    def _process_valid_data(self):
        inter_data = []
        for user_id, items in self.inters.items():
            items = items[:-1]
            items = [item + 1 for item in items]
            seq = items[:-1]
            target = items[-1]
            if len(seq) < self.max_len:
                seq = seq + [0] * (self.max_len - len(seq))
            else:
                seq = seq[-self.max_len:]
            neg_target = random.choice(range(self.num_items)) + 1

            mask_seq_items_index = random.sample(range(self.max_len), self.max_mask_len)
            masked_seq = seq.copy()
            for item in mask_seq_items_index:
                masked_seq[item] = self.num_items + 1
            mask_seq_items = []
            for item in mask_seq_items_index:
                mask_seq_items.append(seq[item])
            neg_mask_seq_items = []
            for item in mask_seq_items_index:
                neg_mask_seq_items.append(random.choice(range(self.num_items)) + 1)
            sample = {
                "user_seqs": np.array(user_id),
                "his_seqs": np.array(seq),
                "next_items": np.array(target),
                "next_neg_items": np.array(neg_target),

                "masked_his_seqs": np.array(masked_seq),
                "mask_indices": np.array(mask_seq_items_index),
                "mask_items": np.array(mask_seq_items),
                "mask_neg_items": np.array(neg_mask_seq_items)
            }
            inter_data.append(sample)
        return inter_data
    
    def _process_test_data(self):
        inter_data = []
        for user_id, items in self.inters.items():
            items = [item + 1 for item in items]
            seq = items[:-1]
            target = items[-1]
            if len(seq) < self.max_len:
                seq = seq + [0] * (self.max_len - len(seq))
            else:
                seq = seq[-self.max_len:]
            neg_target = random.choice(range(self.num_items)) + 1

            mask_seq_items_index = random.sample(range(self.max_len), self.max_mask_len)
            masked_seq = seq.copy()
            for item in mask_seq_items_index:
                masked_seq[item] = self.num_items + 1
            mask_seq_items = []
            for item in mask_seq_items_index:
                mask_seq_items.append(seq[item])
            neg_mask_seq_items = []
            for item in mask_seq_items_index:
                neg_mask_seq_items.append(random.choice(range(self.num_items)) + 1)
            sample = {
                "user_seqs": np.array(user_id),
                "his_seqs": np.array(seq),
                "next_items": np.array(target),
                "next_neg_items": np.array(neg_target),

                "masked_his_seqs": np.array(masked_seq),
                "mask_indices": np.array(mask_seq_items_index),
                "mask_items": np.array(mask_seq_items),
                "mask_neg_items": np.array(neg_mask_seq_items)
            }
            inter_data.append(sample)
        return inter_data
    
    def get_item_num(self):
        return max([max(items) for items in self.inters.values()]) + 1
    
    def get_user_num(self):
        return len(self.inters)

class GenRecDataset(Dataset):
    def __init__(self, data_root_path, dataset, mode:Literal["train", "valid", "test"]):
        self.data_root_path = data_root_path
        self.dataset = dataset
        self.mode = mode

        self.inter_path = os.path.join(data_root_path, dataset, f"{dataset}.inter.csv")
        assert os.path.exists(self.inter_path), f"Inter file not found in {self.inter_path}"

        self.inters = self._load_inter(self.inter_path)
        self.num_items = self.get_item_num()
        self.num_users = self.get_user_num()

        if self.mode == "train":
            self.inter_data = self._process_train_data()
        elif self.mode == "valid":
            self.inter_data = self._process_valid_data()
        elif self.mode == "test":
            self.inter_data = self._process_test_data()

    def __len__(self):
        _len = len(self.inter_data)
        return _len
    
    def __getitem__(self, idx):
        sample = self.inter_data[idx]
        return sample
    
    def _load_inter(self, inter_path):
        inters = {}
        inter_df = pd.read_csv(inter_path)
        for row in inter_df.itertuples():
            user_id = getattr(row, "user_id")
            item_id = getattr(row, "item_id")
            if user_id not in inters:
                inters[user_id] = []
            inters[user_id].append(item_id)
        return inters
    
    def _process_train_data(self):
        inter_data = []
        for user_id, items in self.inters.items():
            items = items[:-2]
            for item in items:
                neg_item = random.choice(range(self.num_items))
                while neg_item in items:
                    neg_item = random.choice(range(self.num_items))
                sample = {
                    "users": np.array(user_id),
                    "items": np.array(item),
                    "neg_items": np.array(neg_item)
                }
                inter_data.append(sample)
        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for user_id, items in self.inters.items():
            item = items[-2]

            neg_item = random.choice(range(self.num_items))
            while neg_item == item:
                neg_item = random.choice(range(self.num_items))

            sample = {
                "users": np.array(user_id),
                "items": np.array(item),
                "neg_items": np.array(neg_item)
            }
            inter_data.append(sample)
        return inter_data

    def _process_test_data(self):
        inter_data = []
        for user_id, items in self.inters.items():
            item = items[-1]

            neg_item = random.choice(range(self.num_items))
            while neg_item == item:
                neg_item = random.choice(range(self.num_items))
                
            sample = {
                "users": np.array(user_id),
                "items": np.array(item),
                "neg_items": np.array(neg_item)
            }
            inter_data.append(sample)
        return inter_data

    def get_item_num(self):
        return max([max(items) for items in self.inters.values ]) + 1

    def get_user_num(self):
        return len(self.inters)

    def get_adjacency_matrix(self):
        indices = []
        values = []
        for user_id, items in self.inters.items():
            for item in items:
                indices.append([user_id, item])
                values.append(1)
        adj_matrix = torch.sparse_coo_tensor(
            indices=torch.tensor(indices).t(),
            values=torch.tensor(values),
            size=(self.num_users, self.num_items)
        )
        return adj_matrix
