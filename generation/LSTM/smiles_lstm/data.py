from torch.utils.data import TensorDataset, DataLoader


def make_dataloaders(tensor, batch_size):
    # 入力: 最後の一文字前まで 出力: 最初の二文字目から
    dataset = TensorDataset(tensor[:, :-1], tensor[:, 1:])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
