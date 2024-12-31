import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class UAVDataset(Dataset):
    def __init__(self, file_path, mode='cyber'):
        df = pd.read_csv(file_path)

        # Drop the timestamp columns (timestamp_c, timestamp_p) if they exist
        if 'timestamp_c' in df.columns:
            df = df.drop(columns=['timestamp_c'])
        if 'timestamp_p' in df.columns:
            df = df.drop(columns=['timestamp_p'])

        # Separate features and labels
        self.X = df.drop(columns=['class']).values  # Features, drop class column
        self.y = pd.factorize(df['class'])[0]  # Encode class as integers

        self.num_classes = len(np.unique(self.y))
        self.input_dim = self.X.shape[1]  # Number of features after dropping timestamps

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


if __name__ == '__main__':
    file_path = '/data/cyber_ready.csv'
    dataset = UAVDataset(file_path)
    print(dataset.input_dim)
    print(dataset.num_classes)
    # check label
    print(len(dataset.y))
