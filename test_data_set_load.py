import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence
import glob
import time
import pickle
from concurrent.futures import ThreadPoolExecutor

class TrajectoryDataset(Dataset):
    def __init__(self, data_dir, cache_dir="traj_cache", force_reload=False):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.file_paths = glob.glob(os.path.join(data_dir, "*.csv"))
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_map = {}
        self.force_reload = force_reload
        
        # 预构建缓存映射
        self._build_cache_map()

    def _build_cache_map(self):
        """建立CSV文件路径到缓存文件的映射"""
        for path in self.file_paths:
            fname = os.path.basename(path).split('.')[0]
            self.cache_map[path] = os.path.join(self.cache_dir, f"{fname}.pkl")
    
    def _load_or_convert(self, path):
        try:
            cache_path = self.cache_map[path]
            if not self.force_reload and os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            
            # 读取CSV时指定数据类型或转换
            df = pd.read_csv(path)
            
            # 1. 处理时间列 - 转换为时间戳数值
            if 'date' in df.columns:
                # 方法1：转换为Unix时间戳（秒）
                df['date'] = pd.to_datetime(df['date']).astype('int64') // 10**9
                
                # 方法2：或转换为相对时间（秒）
                # df['time'] = (pd.to_datetime(df['time']) - pd.Timestamp("2020-01-01")).dt.total_seconds()
            
            # 2. 确保所有列为数值类型
            numeric_cols = ['lat', 'lon', 'sog', 'cog']
            for col in numeric_cols:
                if col in df.columns:
                    # 处理逗号等格式问题
                    if df[col].dtype == object:
                        df[col] = df[col].str.replace(',', '').astype(float)
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 3. 处理缺失值
            df.fillna(0, inplace=True)  # 或使用插值方法
            
            # 4. 选择最终特征列
            features = df[['date'] + numeric_cols].values
            
            # 5. 确保数据类型为float32
            features = features.astype(np.float32)
            
            tensor = torch.tensor(features, dtype=torch.float32)
            
            # 保存缓存
            with open(cache_path, 'wb') as f:
                pickle.dump((tensor, len(tensor)), f)
            
            return tensor, len(tensor)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        item = self._load_or_convert(self.file_paths[idx])
        print(f"Item shape: {item[0].shape}, Length: {item[1]}")  # 调试行
        # return self._load_or_convert(self.file_paths[idx])
        return item



class DynamicBatchCollator:
    def __init__(self, max_batch_size=32, max_seq_len=512, bin_size=64):
        """
        max_batch_size: 最大批次样本数
        max_seq_len: 允许的最大序列长度（截断超长序列）
        bin_size: 长度分组区间大小
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.bin_size = bin_size
    
    def _group_by_length(self, items):
        """按序列长度分组"""
        # 创建长度区间分组
        bins = {}

        for item in items:
            print(f"Item: {item}")  # 打印每个项
            if item is None or not isinstance(item, tuple) or len(item) != 2:
                print(f"Skipping invalid item: {item}")  # 调试行
                continue
            
            if isinstance(item, tuple) and len(item) == 2:
                tensor, length = item
                for tensor, length in items:
                    # 截断超长序列
                    if length > self.max_seq_len:
                        tensor = tensor[:self.max_seq_len]
                        length = self.max_seq_len
                        
                    bin_key = length // self.bin_size
                    if bin_key not in bins:
                        bins[bin_key] = []
                    bins[bin_key].append((tensor, length))
            else:
                print(f"Unexpected item structure: {item}")  # 调试行


        return bins
    
    def __call__(self, items):
        # 按长度分组
        bins = self._group_by_length(items)
        batches = []
        
        # 为每个分组创建批次
        for _, group in bins.items():
            # 按长度降序排序（提高填充效率）
            group.sort(key=lambda x: x[1], reverse=True)
            
            # 将分组划分为小批次
            for i in range(0, len(group), self.max_batch_size):
                batch_items = group[i:i+self.max_batch_size]
                tensors, lengths = zip(*batch_items)
                
                # 动态填充
                padded = pad_sequence(tensors, batch_first=True)
                
                # 创建填充掩码（0表示填充位置）
                mask = torch.zeros_like(padded[:, :, 0])
                for j, length in enumerate(lengths):
                    mask[j, :length] = 1
                
                batches.append((padded, mask, torch.tensor(lengths)))
        
        # 随机打乱批次顺序
        np.random.shuffle(batches)
        return batches


# 初始化数据集
dataset = TrajectoryDataset(
    data_dir="sample_data",
    cache_dir="traj_cache",
    force_reload=False  # 设置为True强制重新生成缓存
)

# 初始化动态批处理器
collator = DynamicBatchCollator(
    max_batch_size=64,
    max_seq_len=1000,
    bin_size=100
)

# 创建DataLoader（注意：batch_size=None使用全动态批处理）
dataloader = DataLoader(
    dataset,
    batch_size=None,  # 必须设为None
    shuffle=True,
    num_workers=4,    # 并行加载进程数
    collate_fn=collator,
    pin_memory=True    # 加速GPU传输
)

# 训练循环示例
for batch_idx, batch in enumerate(dataloader):
    padded, mask, lengths = batch
    # 将数据转移到GPU
    padded = padded.to('cuda')
    mask = mask.to('cuda')
    
    # 模型前向传播（示例）
    # output = model(padded, mask, lengths)
    
    if batch_idx % 100 == 0:
        print(f"Batch {batch_idx}: shape={padded.shape}, mean_len={lengths.float().mean().item():.1f}")