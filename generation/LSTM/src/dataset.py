import multiprocessing

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pytorch_lightning as pl


# CORES = multiprocessing.cpu_count() // 2
CORES = 2

# def make_dataloaders(tensor, batch_size):
#     # 入力: 最後の一文字前まで 出力: 最初の二文字目から
#     dataset = TensorDataset(tensor[:, :-1], tensor[:, 1:])
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     return dataloader


class Data(Dataset):
    def __init__(self, smiles_tensor):
        self.in_seq, self.out_seq = smiles_tensor[:, :-1], smiles_tensor[:, 1:]

    def __len__(self):
        return len(self.in_seq)

    def __getitem__(self, index):
        return self.in_seq[index], self.out_seq[index]

class DataModule(pl.LightningDataModule):
    def __init__(self, train_smiles, valid_smiles, batch_size=1):
        super().__init__()
        self.train_smiles = train_smiles
        self.valid_smiles = valid_smiles
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = Data(self.train_smiles)
        self.valid_dataset = Data(self.valid_smiles)

    def train_dataloader(self):
        if len(self.train_dataset) != 0:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=CORES,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True
            )
        else:
            raise Exception("length of train dataset is zero.")

    def val_dataloader(self):
        sample_size = len(self.valid_dataset)
        if sample_size != 0:
            return DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size if sample_size > self.batch_size else sample_size,
                shuffle=False,
                num_workers=CORES,
                pin_memory=True,
                drop_last=False,
                persistent_workers=True
            )
        else:
            raise Exception("length of valid dataset is zero.")

