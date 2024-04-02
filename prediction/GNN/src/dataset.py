from torch_geometric.loader import DataLoader
import lightning as L


CORES = 2


class DataModule(L.LightningDataModule):
    def __init__(self, graph_train, graph_valid, batch_size=1):
        super().__init__()
        self.graph_train, self.graph_valid = graph_train, graph_valid
        self.batch_size = batch_size

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        if len(self.graph_train) != 0:
            return DataLoader(
                self.graph_train,
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
        sample_size = len(self.graph_valid)
        if len(self.graph_valid) != 0:
            return DataLoader(
                self.graph_valid,
                batch_size=self.batch_size if sample_size > self.batch_size else sample_size,
                shuffle=False,
                num_workers=CORES,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True
            )
        else:
            raise Exception("length of valid dataset is zero.")
