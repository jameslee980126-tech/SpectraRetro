import os
import subprocess

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from tqdm import tqdm
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessPool

from ..process import RawDataConverter
from ..dataset import DatasetModule

__all__ = ['USPTODataset']

class USPTODataset(DatasetModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        
    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        is_aug_batch = isinstance(batch, list)
        if not is_aug_batch:
            batch = [batch]
        for label, label_dim in self.label_dims.items():
            for b in batch:
                setattr(
                    b, label,
                    F.one_hot(getattr(b, label), label_dim)
                )
        if not is_aug_batch:
            batch = batch[0]
        return batch

    @property
    def raw_file_names(self):
        if self.root.__contains__('50k'):
            return ['raw_train.csv', 'raw_val.csv', 'raw_test.csv']
        elif self.root.__contains__('full'):
            return ['raw_train.csv', 'raw_val.csv', 'raw_test.csv']
        else:
            raise NotImplementedError()


    def download(self):
        expected_paths = '\n'.join(self.split_paths)
        raise FileNotFoundError(
            'USPTO raw data is missing. Place the train, validation, and test '
            f'CSV files at:\n{expected_paths}'
        )


    def process(self):
        if torch.cuda.device_count() > 1:
            raise NotImplementedError()

        processed_path = self.processed_paths[self.file_idx]
        if os.path.exists(processed_path):
            return
        if self.file_idx:
            assert os.path.exists(self.processed_paths[0]), \
                f"Please process the training set first to build the graph vocab."
            assert os.path.exists(self.label_mapping_path)

        converter = RawDataConverter(
            dn_last=self.dn_last,
            max_n_len=self.max_n_len,
            n_dummy=self.n_dummy,
            shuffle_order=self.shuffle_order,
            canonicalize=self.canonicalize
        )
        react_data = self.get_react_data()
        label_data = self.get_label_data()
        if label_data is None:
            label_data = np.array([0] * len(react_data))

        if self.file_idx == 0:
            n_label_type = label_data.max() + 1
            self.save_label_mappings({
                'label': dict(zip(range(n_label_type), range(n_label_type))),
            })

        total_n = len(react_data)
        assert total_n == len(label_data)

        x_enc, e_enc = self.setup_graph_vocab(converter, react_data)

        converter.setup_enc(x_enc, e_enc)
        processor = converter.rawdata2graph
        
        input_pairs = list(zip(react_data, label_data))
        def process_item(args):
            data, label = args
            return processor(data, label=label)

        data_list = []
        successful_conversions = 0
        with ProcessPool(cpu_count()) as pool:
            with tqdm(
                total=total_n, dynamic_ncols=True,
                desc=f"Processing {self.stage} data"
            ) as pbar:
                for pro_data in pool.imap(process_item, input_pairs):
                    if pro_data:
                        data_list.extend([
                            Data(**data) for data in pro_data
                        ])
                        successful_conversions += len(pro_data)
                    pbar.set_postfix({
                        'Graphs': len(data_list), 
                        'Raw data': total_n,
                    })
                    pbar.update()

        print(f"Processing complete: {len(data_list)} graphs generated from {total_n} reactions")
        print(f"Success rate: {len(data_list)/total_n*100:.2f}%")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        torch.save(self.collate(data_list), processed_path)
        print(f"Saved processed data to: {processed_path}")






    def get_react_data(self) -> list[str]:
        table = pd.read_csv(self.split_paths[self.file_idx])
        return table['reactants>reagents>production'].values
    
    def get_label_data(self) -> list[int]:
        if self.root.__contains__('full'):
            return None
        table = pd.read_csv(self.split_paths[self.file_idx])
        return table['class'].values - 1
    
