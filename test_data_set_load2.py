import pandas as pd
import glob

# all_files = glob.glob('data/*.csv')
# dfs = []
# for f in all_files:
#     df = pd.read_csv(f)
#     df['file_id'] = f  # 可选：记录原文件名
#     dfs.append(df)
# big_df = pd.concat(dfs, ignore_index=True)
# big_df.to_parquet('all_trajectory.parquet')
# print('合并完成！')

import pandas as pd
import glob

# all_files = glob.glob('sample_data/*.csv')
# dfs = []
# track_indices = []  # 每条轨迹的起/止行号
# start = 0
# for f in all_files:
#     df = pd.read_csv(f)
#     end = start + len(df)
#     dfs.append(df)
#     track_indices.append((start, end))
#     start = end
# big_df = pd.concat(dfs, ignore_index=True)
# big_df.to_parquet('all_trajectory.parquet')
# pd.DataFrame(track_indices, columns=['start', 'end']).to_csv('track_indices.csv', index=False)
# print('合并和索引完成！')



import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class BigTrajectoryDataset(Dataset):
    def __init__(self, parquet_file, index_file, feature_cols=['date','lat','lon','sog','cog','label']):
        self.df = pd.read_parquet(parquet_file, columns=feature_cols)
        self.index = pd.read_csv(index_file)
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        start, end = self.index.iloc[idx]
        data = self.df.iloc[start:end,1:5].values.astype(float)
        return torch.tensor(data, dtype=torch.float)

def collate_fn(batch):
    padded_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    lengths = torch.tensor([b.shape[0] for b in batch])
    max_len = padded_batch.size(1)
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return padded_batch, mask

dataset = BigTrajectoryDataset('all_trajectory.parquet', 'track_indices.csv')
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)



for batch_data, batch_mask in dataloader:
    # batch_data: [batch_size, max_seq_len, 4]
    # batch_mask: [batch_size, max_seq_len]，1为有效，0为pad
    # 你的训练代码
    print(batch_data.shape, batch_mask.shape)
    print(batch_mask)