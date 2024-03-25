import multiprocessing

from torch.utils.data import Dataset, DataLoader


CORES = multiprocessing.cpu_count()

class Data(Dataset):
    def __init__(self, X, prop):
        self.X = X
        self.prop = prop

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.prop[index]


def make_dataloader(X, prop, batch_size=1, train=False):
    if train:
        shuffle = True
    else:
        shuffle = False
    dataset = Data(X=X, prop=prop)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=CORES//2,
        pin_memory=True,
        persistent_workers=True,
    )
    return dataloader
