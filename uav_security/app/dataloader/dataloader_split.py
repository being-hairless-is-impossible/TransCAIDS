import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class UAVDataset(Dataset):
    def __init__(self, file_path, cutoff, ):
        df = pd.read_csv(file_path)

        # Drop the timestamp columns (timestamp_c, timestamp_p) if they exist
        if 'timestamp_c' in df.columns:
            df = df.drop(columns=['timestamp_c'])
        if 'timestamp_p' in df.columns:
            df = df.drop(columns=['timestamp_p'])

        # Separate features and labels
        self.X = df.drop(columns=['class']).values  # Features
        self.y = pd.factorize(df['class'])[0]      # Encode class as integers

        # Split data based on the cutoff
        self.X_cyber = self.X[:, :cutoff]          # Cyber features before the cutoff
        self.X_physical = self.X[:, cutoff:]       # Physical features after the cutoff

        self.num_classes = len(np.unique(self.y))
        self.input_dim_cyber = self.X_cyber.shape[1]   # Number of cyber features
        self.input_dim_physical = self.X_physical.shape[1]   # Number of physical features

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        cyber_features = torch.tensor(self.X_cyber[idx], dtype=torch.float32)
        physical_features = torch.tensor(self.X_physical[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return cyber_features, physical_features, label


if __name__ == '__main__':
    # print x_cyber column list
    data_path = '/data/fuse_2.csv'
    dataset = UAVDataset(data_path, 38)
    print(dataset.input_dim_cyber)
