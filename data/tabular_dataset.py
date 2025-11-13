import pandas as pd
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df.drop('g_vals', axis=1).values, dtype=torch.float32)
        self.y = torch.tensor(df['g_vals'].astype('category').cat.codes.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
