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

    def __init__(self, data_root_path, dataset, max_len, mode:Literal["train", "valid", "test"], mask_ratio=0.15, pair_num_per_pos=100):
        self.data_root_path = data_root_path
        self.dataset = dataset
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        self.max_mask_len = int(max_len * mask_ratio)
        self.mode = mode
        self.pair_num_per_pos = pair_num_per_pos

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
                # normal data sample
                seq = items[:i]
                target = items[i]
                if len(seq) < self.max_len:
                    seq = seq + [0] * (self.max_len - len(seq))
                else:
                    seq = seq[-self.max_len:]
                
                neg_targets = []
                while len(neg_targets) < self.pair_num_per_pos:
                    neg_target = random.choice(range(self.num_items)) + 1
                    while neg_target == target:
                        neg_target = random.choice(range(self.num_items)) + 1
                    neg_targets.append(neg_target)

                # mask motivated data sample
                seq_mask_motivated = items.copy()
                if len(seq_mask_motivated) < self.max_len:
                    seq_mask_motivated = seq_mask_motivated + [0] * (self.max_len - len(seq_mask_motivated))
                else:
                    seq_mask_motivated = seq_mask_motivated[-self.max_len:]

                mask_seq_items_index = random.sample(range(self.max_len), self.max_mask_len)

                masked_seq = seq_mask_motivated.copy()
                for item in mask_seq_items_index:
                    masked_seq[item] = self.num_items + 1

                mask_seq_items = []
                for item in mask_seq_items_index:
                    mask_seq_items.append(seq_mask_motivated[item])
                
                neg_mask_seq_items = []
                for item in mask_seq_items_index:
                    neg_mask_seq_items_i = []
                    while len(neg_mask_seq_items_i) < self.pair_num_per_pos:
                        neg_mask_seq_item = random.choice(range(self.num_items)) + 1
                        while neg_mask_seq_item == seq_mask_motivated[item]:
                            neg_mask_seq_item = random.choice(range(self.num_items)) + 1
                        neg_mask_seq_items_i.append(neg_mask_seq_item)
                    neg_mask_seq_items.append(neg_mask_seq_items_i)
                
                sample = {
                    "user_seqs": np.array(user_id),
                    "his_seqs": np.array(seq),
                    "next_items": np.array(target),
                    "next_neg_items": np.array(neg_targets),

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

            # normal data sample
            seq = items[:-1]
            target = items[-1]
            if len(seq) < self.max_len:
                seq = seq + [0] * (self.max_len - len(seq))
            else:
                seq = seq[-self.max_len:]

            neg_targets = []
            while len(neg_targets) < self.pair_num_per_pos:
                neg_target = random.choice(range(self.num_items)) + 1
                while neg_target == target:
                    neg_target = random.choice(range(self.num_items)) + 1
                neg_targets.append(neg_target)

            # mask motivated data sample
            seq_mask_motivated = items.copy()
            if len(seq_mask_motivated) < self.max_len:
                seq_mask_motivated = seq_mask_motivated + [0] * (self.max_len - len(seq_mask_motivated))
            else:
                seq_mask_motivated = seq_mask_motivated[-self.max_len:]

            mask_seq_items_index = random.sample(range(self.max_len), self.max_mask_len)

            masked_seq = seq_mask_motivated.copy()
            for item in mask_seq_items_index:
                masked_seq[item] = self.num_items + 1

            mask_seq_items = []
            for item in mask_seq_items_index:
                mask_seq_items.append(seq_mask_motivated[item])

            neg_mask_seq_items = []
            for item in mask_seq_items_index:
                neg_mask_seq_items_i = []
                while len(neg_mask_seq_items_i) < self.pair_num_per_pos:
                    neg_mask_seq_item = random.choice(range(self.num_items)) + 1
                    while neg_mask_seq_item == seq_mask_motivated[item]:
                        neg_mask_seq_item = random.choice(range(self.num_items)) + 1
                    neg_mask_seq_items_i.append(neg_mask_seq_item)
                neg_mask_seq_items.append(neg_mask_seq_items_i)

            sample = {
                "user_seqs": np.array(user_id),
                "his_seqs": np.array(seq),
                "next_items": np.array(target),
                "next_neg_items": np.array(neg_targets),

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

            # normal data sample
            seq = items[:-1]
            target = items[-1]
            if len(seq) < self.max_len:
                seq = seq + [0] * (self.max_len - len(seq))
            else:
                seq = seq[-self.max_len:]

            neg_targets = []
            while len(neg_targets) < self.pair_num_per_pos:
                neg_target = random.choice(range(self.num_items)) + 1
                while neg_target == target:
                    neg_target = random.choice(range(self.num_items)) + 1
                neg_targets.append(neg_target)

            # mask motivated data sample
            seq_mask_motivated = items.copy()
            if len(seq_mask_motivated) < self.max_len:
                seq_mask_motivated = seq_mask_motivated + [0] * (self.max_len - len(seq_mask_motivated))
            else:
                seq_mask_motivated = seq_mask_motivated[-self.max_len:]

            mask_seq_items_index = random.sample(range(self.max_len), self.max_mask_len)

            masked_seq = seq_mask_motivated.copy()
            for item in mask_seq_items_index:
                masked_seq[item] = self.num_items + 1

            mask_seq_items = []
            for item in mask_seq_items_index:
                mask_seq_items.append(seq_mask_motivated[item])

            neg_mask_seq_items = []
            for item in mask_seq_items_index:
                neg_mask_seq_items_i = []
                while len(neg_mask_seq_items_i) < self.pair_num_per_pos:
                    neg_mask_seq_item = random.choice(range(self.num_items)) + 1
                    while neg_mask_seq_item == seq_mask_motivated[item]:
                        neg_mask_seq_item = random.choice(range(self.num_items)) + 1
                    neg_mask_seq_items_i.append(neg_mask_seq_item)
                neg_mask_seq_items.append(neg_mask_seq_items_i)

            sample = {
                "user_seqs": np.array(user_id),
                "his_seqs": np.array(seq),
                "next_items": np.array(target),
                "next_neg_items": np.array(neg_targets),

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
    def __init__(self, data_root_path, dataset, mode:Literal["train", "valid", "test"], pair_num_per_pos=100):
        self.data_root_path = data_root_path
        self.dataset = dataset
        self.mode = mode
        self.pair_num_per_pos = pair_num_per_pos

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
            train_items = items[:-2]
            for item in train_items:
                neg_items = []
                while len(neg_items) < self.pair_num_per_pos:
                    neg_item = random.choice(range(self.num_items))
                    while neg_item in items:
                        neg_item = random.choice(range(self.num_items))
                    neg_items.append(neg_item)

                sample = {
                    "users": np.array(user_id),
                    "items": np.array(item),
                    "neg_items": np.array(neg_items)
                }
                inter_data.append(sample)
        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for user_id, items in self.inters.items():
            item = items[-2]
            neg_items = []
            while len(neg_items) < self.pair_num_per_pos:
                neg_item = random.choice(range(self.num_items))
                while neg_item in items:
                    neg_item = random.choice(range(self.num_items))
                neg_items.append(neg_item)

            sample = {
                "users": np.array(user_id),
                "items": np.array(item),
                "neg_items": np.array(neg_items)
            }
            inter_data.append(sample)
        return inter_data

    def _process_test_data(self):
        inter_data = []
        for user_id, items in self.inters.items():
            item = items[-1]
            neg_items = []
            while len(neg_items) < self.pair_num_per_pos:
                neg_item = random.choice(range(self.num_items))
                while neg_item in items:
                    neg_item = random.choice(range(self.num_items))
                neg_items.append(neg_item)

            sample = {
                "users": np.array(user_id),
                "items": np.array(item),
                "neg_items": np.array(neg_items)
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

class ConRecDataset(Dataset):
    def __init__(self, data_root_path, dataset, mode:Literal["train", "valid", "test"]):
        self.data_root_path = data_root_path
        self.dataset = dataset
        self.mode = mode

        self.inter_path = os.path.join(data_root_path, dataset, f"{dataset}.inter.csv")
        assert os.path.exists(self.inter_path), f"Inter file not found in {self.inter_path}"

        self.item_path = os.path.join(data_root_path, dataset, f"{dataset}.item.csv")
        assert os.path.exists(self.item_path), f"Item file not found in {self.item_path}"

        self.user_path = os.path.join(data_root_path, dataset, f"{dataset}.user.csv")
        assert os.path.exists(self.user_path), f"User file not found in {self.user_path}"

        self.inters = self._load_inter(self.inter_path)
        self.items = self._load_item(self.item_path)
        self.users = self._load_user(self.user_path)

        self.num_items = self.get_item_num()
        self.num_users = self.get_user_num()

        self.token_field_value_num_list = self.get_token_field_value_num_list()
        self.token_sequence_field_value_num_list = self.get_token_sequence_field_value_num_list()

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
    
    def _load_item(self, item_path):
        items = {}
        item_df = pd.read_csv(item_path)
        column_name_list = item_df.columns.difference(['index']).to_list()
        type_list = {}
        for idx, row in enumerate(item_df.itertuples(index=False)):
            if idx == 0:
                for column_name in column_name_list:
                    type_i = getattr(row, column_name)
                    type_list[column_name] = type_i
            else:
                item_id = getattr(row, "item_id")
                token_field_values = []
                token_sequence_field_values = []
                for column_name in column_name_list:
                    value_i = getattr(row, column_name)
                    if type_list[column_name] == "token":
                        token_field_values.append(value_i)
                    elif type_list[column_name] == "token_seq":
                        token_sequence_field_values.append(value_i.strip().split(" "))
                    else:
                        continue
                items[item_id] = {
                    "token_field_values": token_field_values,
                    "token_sequence_field_values": token_sequence_field_values
                }
        return items
    
    def _load_user(self, user_path):
        users = {}
        user_df = pd.read_csv(user_path)
        column_name_list = user_df.columns.difference(['index']).to_list()
        type_list = {}
        for idx, row in enumerate(user_df.itertuples(index=False)):
            if idx == 0:
                for column_name in column_name_list:
                    type_i = getattr(row, column_name)
                    type_list[column_name] = type_i
            else:
                user_id = getattr(row, "item_id")
                token_field_values = []
                token_sequence_field_values = []
                for column_name in column_name_list:
                    value_i = getattr(row, column_name)
                    if type_list[column_name] == "token":
                        token_field_values.append(value_i)
                    elif type_list[column_name] == "token_seq":
                        token_sequence_field_values.append(value_i.strip().split(" "))
                    else:
                        continue
                users[user_id] = {
                    "token_field_values": token_field_values,
                    "token_sequence_field_values": token_sequence_field_values
                }
        return users

    def _process_train_data(self):
        inter_data = []
        for user_id, items in self.inters.items():
            items = items[:-2]
            token_field_values = [user_token_value for user_token_value in self.users[user_id]["token_field_values"]]
            token_sequence_field_values = [np.array(user_token_seq_value) for user_token_seq_value in self.users[user_id]["token_sequence_field_values"]]
            for item in items:
                token_field_values.extend(self.items[item]["token_field_values"])
                token_sequence_field_values.extend(
                    [
                        np.array(item_token_seq_value) 
                        for item_token_seq_value in self.items[item]["token_sequence_field_values"]
                    ]
                )
                sample = {
                    "token_field_values": np.array(token_field_values),
                    "token_sequence_field_values": token_sequence_field_values,
                    "labels": np.array(1)
                }
                inter_data.append(sample)

                # negative sampling
                neg_item = random.choice(range(self.num_items))
                while neg_item in items:
                    neg_item = random.choice(range(self.num_items))
                token_field_values = [user_token_value for user_token_value in self.users[user_id]["token_field_values"]]
                token_sequence_field_values = [np.array(user_token_seq_value) for user_token_seq_value in self.users[user_id]["token_sequence_field_values"]]
                token_field_values.extend(self.items[neg_item]["token_field_values"])
                token_sequence_field_values.extend(
                    [
                        np.array(neg_item_token_seq_value) 
                        for neg_item_token_seq_value in self.items[neg_item]["token_sequence_field_values"]
                    ]
                )
                sample = {
                    "token_field_values": np.array(token_field_values),
                    "token_sequence_field_values": token_sequence_field_values,
                    "labels": np.array(0)
                }
        return inter_data

    def _process_valid_data(self):
        inter_data = []
        for user_id, items in self.inters.items():
            item = items[-2]
            token_field_values = [user_token_value for user_token_value in self.users[user_id]["token_field_values"]]
            token_sequence_field_values = [np.array(user_token_seq_value) for user_token_seq_value in self.users[user_id]["token_sequence_field_values"]]
            token_field_values.extend(self.items[item]["token_field_values"])
            token_sequence_field_values.extend(
                [
                    np.array(item_token_seq_value) 
                    for item_token_seq_value in self.items[item]["token_sequence_field_values"]
                ]
            )
            sample = {
                "token_field_values": np.array(token_field_values),
                "token_sequence_field_values": token_sequence_field_values,
                "labels": np.array(1)
            }
            inter_data.append(sample)

            # negative sampling
            neg_item = random.choice(range(self.num_items))
            while neg_item == item:
                neg_item = random.choice(range(self.num_items))
            token_field_values = [user_token_value for user_token_value in self.users[user_id]["token_field_values"]]
            token_sequence_field_values = [np.array(user_token_seq_value) for user_token_seq_value in self.users[user_id]["token_sequence_field_values"]]
            token_field_values.extend(self.items[neg_item]["token_field_values"])
            token_sequence_field_values.extend(
                [
                    np.array(neg_item_token_seq_value) 
                    for neg_item_token_seq_value in self.items[neg_item]["token_sequence_field_values"]
                ]
            )
            sample = {
                "token_field_values": np.array(token_field_values),
                "token_sequence_field_values": token_sequence_field_values,
                "labels": np.array(0)
            }
            inter_data.append(sample)
        return inter_data

    def _process_test_data(self):
        inter_data = []
        for user_id, items in self.inters.items():
            item = items[-1]
            token_field_values = [user_token_value for user_token_value in self.users[user_id]["token_field_values"]]
            token_sequence_field_values = [np.array(user_token_seq_value) for user_token_seq_value in self.users[user_id]["token_sequence_field_values"]]
            token_field_values.extend(self.items[item]["token_field_values"])
            token_sequence_field_values.extend(
                [
                    np.array(item_token_seq_value) 
                    for item_token_seq_value in self.items[item]["token_sequence_field_values"]
                ]
            )
            sample = {
                "token_field_values": np.array(token_field_values),
                "token_sequence_field_values": token_sequence_field_values,
                "labels": np.array(1)
            }
            inter_data.append(sample)

            # negative sampling
            neg_item = random.choice(range(self.num_items))
            while neg_item == item:
                neg_item = random.choice(range(self.num_items))
            token_field_values = [user_token_value for user_token_value in self.users[user_id]["token_field_values"]]
            token_sequence_field_values = [np.array(user_token_seq_value) for user_token_seq_value in self.users[user_id]["token_sequence_field_values"]]
            token_field_values.extend(self.items[neg_item]["token_field_values"])
            token_sequence_field_values.extend(
                [
                    np.array(neg_item_token_seq_value) 
                    for neg_item_token_seq_value in self.items[neg_item]["token_sequence_field_values"]
                ]
            )
            sample = {
                "token_field_values": np.array(token_field_values),
                "token_sequence_field_values": token_sequence_field_values,
                "labels": np.array(0)
            }
            inter_data.append(sample)
        return inter_data 

    def get_item_num(self):
        return max([max(items) for items in self.inters.values]) + 1

    def get_user_num(self):
        return len(self.inters)
    
    def get_token_field_value_num_list(self):
        token_field_value_num_list = [0 for _ in range(len(self.items[0]["token_field_values"]))]
        for item in self.items.values():
            for idx, token_field_value in enumerate(item["token_field_values"]):
                token_field_value_num_list[idx] = max(token_field_value_num_list[idx], token_field_value)

        return token_field_value_num_list
    
    def get_token_sequence_field_value_num_list(self):
        token_sequence_field_value_num_list = [0 for _ in range(len(self.items[0]["token_sequence_field_values"]))]
        for item in self.items.values():
            for idx, token_sequence_field_values in enumerate(item["token_sequence_field_values"]):
                for token_sequence_value in token_sequence_field_values:
                    token_sequence_field_value_num_list[idx] = max(token_sequence_field_value_num_list[idx], token_sequence_value)
        
        return token_sequence_field_value_num_list
