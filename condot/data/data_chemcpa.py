import torch
from torch.utils.data import Dataset
# from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator
from itertools import cycle
# import random
from condot.data.utils import cast_loader_to_iterator


def create_data_loader(split_data, batch_size, smiles_to_index, align_by_cell_type=True,
                       num_workers=0, pin_memory=False, return_loader=True,
                       return_fields=None):
    """
    创建数据加载器
    :param split_data: 分割的数据
    :param batch_size: 批大小
    :param smiles_to_index: SMILES到索引的映射
    :param align_by_cell_type: 是否按细胞类型对齐
    :param num_workers: 工作线程数
    :param pin_memory: 是否锁定内存
    :param return_loader: 是否返回加载器
    :param return_fields: 指定要返回的字段列表，例如['features', 'SMILES_idx', 'dose']，
                         默认为None时返回所有字段
    """
    # 设置默认返回所有字段
    if return_fields is None:
        return_fields = ['features', 'SMILES_idx', 'dose', 'cell_type']

    # 按 split 和 cell_type 创建 AnnDataDataset
    if align_by_cell_type:
        datasets = {
            split: {
                cell_type: AnnDataDataset(
                    adata=split_data[split][split_data[split].obs['cell_type'] == cell_type],
                    smiles_to_index=smiles_to_index,
                    return_fields=return_fields
                )
                for cell_type in ['A549', 'K562', 'MCF7']
            }
            for split in split_data.keys()
        }
        # 按 split 创建 DataLoader
        dataloaders = {
            split: CombinedDataLoaderAlignedCellTypes(
                dataset_dict=datasets[split],
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                return_fields=return_fields
            )
            for split in datasets.keys()
        }
    else:
        datasets = {
            split: AnnDataDataset(
                adata=split_data[split],
                smiles_to_index=smiles_to_index,
                return_fields=return_fields
            )
            for split in split_data.keys()
        }
        dataloaders = {
            split: CombinedDataLoaderUnalignedCellTypes(
                dataset=datasets[split],
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                return_fields=return_fields
            )
            for split in datasets.keys()
        }
    if not return_loader:
        return datasets
    return datasets, dataloaders


class CombinedDataLoaderUnalignedCellTypes():
    def __init__(self, dataset, batch_size, num_workers=0, pin_memory=False, return_fields=None):
        """
        初始化 CombinedDataLoader
        :param dataset: 数据集
        :param batch_size: 每种细胞类型数据集采样的样本数
        :param num_workers: DataLoader 使用的线程数
        :param pin_memory: 是否锁定内存
        :param return_fields: 指定要返回的字段列表
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.return_fields = return_fields if return_fields else ['features', 'SMILES_idx', 'dose', 'cell_type']

        # 创建自定义的collate函数，只返回指定的字段
        collate_fn = lambda batch: custom_collate(batch, self.return_fields)

        # 为数据集创建 DataLoader
        self.loader = DataLoader(
            dataset=dataset,
            sampler=InfiniteSampler(dataset),
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=True if num_workers > 0 else False,
        )

    def __iter__(self):
        """初始化迭代器"""
        self.iterator = cast_loader_to_iterator(self.loader)
        return self

    def __next__(self):
        """获取下一批数据"""
        return next(self.iterator)


class CombinedDataLoaderAlignedCellTypes:
    def __init__(self, dataset_dict, batch_size, num_workers=0, pin_memory=False, return_fields=None):
        """
        初始化 CombinedDataLoader
        :param dataset_dict: 包含每种细胞类型数据集的字典
        :param batch_size: 每种细胞类型数据集采样的样本数
        :param num_workers: DataLoader 使用的线程数
        :param pin_memory: 是否锁定内存
        :param return_fields: 指定要返回的字段列表
        """
        self.dataset_dict = dataset_dict
        self.batch_size = batch_size
        self.return_fields = return_fields if return_fields else ['features', 'SMILES_idx', 'dose', 'cell_type']

        # 创建自定义的collate函数，只返回指定的字段
        collate_fn = lambda batch: custom_collate(batch, self.return_fields)

        # 为每个细胞类型的 Dataset 创建 DataLoader
        self.loaders = {
            cell_type: DataLoader(
                dataset=dataset,
                sampler=InfiniteSampler(dataset),
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=4 if num_workers > 0 else None,
                persistent_workers=True if num_workers > 0 else False,
            )
            for cell_type, dataset in dataset_dict.items()
        }

    def __iter__(self):
        """初始化迭代器"""
        self.iterators = {cell_type: cast_loader_to_iterator(loader) for cell_type, loader in self.loaders.items()}
        return self

    def __next__(self):
        """获取下一批数据（并行版本）"""
        from concurrent.futures import ThreadPoolExecutor
        
        # 初始化存储字典，只包含需要返回的字段
        batch_data = {field: [] for field in self.return_fields}

        # 并行从每个细胞类型的迭代器获取数据
        with ThreadPoolExecutor(max_workers=3) as executor:
            # 提交所有的 next() 调用
            futures = {cell_type: executor.submit(next, iterator) 
                       for cell_type, iterator in self.iterators.items()}
            
            # 收集结果
            for cell_type, future in futures.items():
                batch = future.result()
                
                # 只处理需要返回的字段
                for field in self.return_fields:
                    if field in batch:
                        if isinstance(batch[field], torch.Tensor):
                            batch_data[field].append(batch[field])
                        elif field == 'cell_type':
                            batch_data[field].extend(batch[field])

        # 合并数据
        combined_batch = {}
        for field in self.return_fields:
            if field in batch_data:
                if field == 'cell_type':
                    combined_batch[field] = batch_data[field]
                else:
                    combined_batch[field] = torch.cat(batch_data[field], dim=0)

        return combined_batch


def custom_collate(batch, return_fields=None):
    """
    自定义批次合并函数
    :param batch: 每次 DataLoader 迭代的样本列表
    :param return_fields: 指定要返回的字段列表
    """
    if return_fields is None:
        return_fields = ['features', 'SMILES_idx', 'dose', 'cell_type']

    result = {}

    # 只处理需要返回的字段
    if 'features' in return_fields:
        result['features'] = torch.stack([item['features'] for item in batch])

    if 'SMILES_idx' in return_fields:
        result['SMILES_idx'] = torch.stack([item['SMILES_idx'] for item in batch])

    if 'dose' in return_fields:
        result['dose'] = torch.stack([item['dose'] for item in batch])

    if 'cell_type' in return_fields:
        result['cell_type'] = [item['cell_type'] for item in batch]

    return result


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return cycle(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class AnnDataDataset(Dataset):
    def __init__(self, adata, smiles_to_index, return_fields=None):
        """
        初始化数据集
        :param adata: AnnData 对象
        :param smiles_to_index: SMILES 字符串到索引的映射
        :param return_fields: 指定要返回的字段列表
        """
        self.return_fields = return_fields if return_fields else ['features', 'SMILES_idx', 'dose', 'cell_type']

        # 只初始化需要返回的字段
        if 'features' in self.return_fields:
            self.features = torch.tensor(adata.X, dtype=torch.float32)

        if 'SMILES_idx' in self.return_fields:
            self.smiles_indices = torch.tensor([smiles_to_index[s] for s in adata.obs['SMILES']], dtype=torch.long)

        if 'dose' in self.return_fields:
            self.doses = torch.tensor(adata.obs['dose'].values, dtype=torch.float32)

        if 'cell_type' in self.return_fields:
            self.cell_type_map = {'A549': 0, 'K562': 1, 'MCF7': 2}
            self.cell_types = (
                adata.obs['cell_type']
                .astype(str)
                .map(self.cell_type_map)
                .fillna(-1)
                .values
            )
            self.cell_types = [[x] for x in self.cell_types]

        # 保存数据长度
        self.length = adata.shape[0]

    def __len__(self):
        """返回数据集的大小"""
        return self.length

    def __getitem__(self, idx):
        """
        获取一个样本的数据
        :param idx: 索引
        """
        result = {}

        # 只返回需要的字段
        if 'features' in self.return_fields:
            result['features'] = self.features[idx]

        if 'SMILES_idx' in self.return_fields:
            result['SMILES_idx'] = self.smiles_indices[idx]

        if 'dose' in self.return_fields:
            result['dose'] = self.doses[idx]

        if 'cell_type' in self.return_fields:
            result['cell_type'] = self.cell_types[idx]

        return result