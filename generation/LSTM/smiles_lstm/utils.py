from tqdm import tqdm

from rdkit import Chem
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def valid_ratio(smiles_list):
    n_success = 0
    for each_smiles in smiles_list:
        try:
            Chem.MolToSmiles(Chem.MolFromSmiles(each_smiles))
            n_success += 1
        except:
            pass
    return n_success / len(smiles_list)


def trainer(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_function,
    n_epoch=1,
    print_freq=100,
    device="cuda",
):
    model.train()
    model.to(device)

    # smiles_vocab()によってSMILES→整列系列に変換されたTensor
    train_loss_list = []
    val_loss_list = []
    running_loss = 0
    running_sample_size = 0
    batch_idx = 0
    for _ in range(n_epoch):
        for each_train_batch in tqdm(train_dataloader):
            in_seq = each_train_batch[0].to(device)
            out_seq = each_train_batch[1].to(device)
            optimizer.zero_grad()
            output = model(in_seq)
            each_loss = loss_function(output.transpose(1, 2), out_seq)
            # each_loss = model.loss(
            #     each_train_batch[0].to(device), each_train_batch[1].to(device)
            # )
            each_loss = each_loss.mean()
            running_loss += each_loss.item()
            running_sample_size += len(each_train_batch[0])
            each_loss.backward()
            optimizer.step()
            if (batch_idx + 1) % print_freq == 0:
                train_loss_list.append(
                    (batch_idx + 1, running_loss / running_sample_size)
                )
                print(
                    "#update: {},\tper-example "
                    "train loss:\t{}".format(
                        batch_idx + 1, running_loss / running_sample_size
                    )
                )
                running_loss = 0
                running_sample_size = 0
                if (batch_idx + 1) % (print_freq * 10) == 0:
                    val_loss = 0
                    with torch.no_grad():
                        for each_val_batch in val_dataloader:
                            in_seq = each_val_batch[0].to(device)
                            out_seq = each_val_batch[1].to(device)
                            output = model(in_seq)
                            each_val_loss = loss_function(output.transpose(1, 2), out_seq)
                            # each_val_loss = model.loss(
                            #     each_val_batch[0].to(device),
                            #     each_val_batch[1].to(device),
                            # )
                            each_val_loss = each_val_loss.mean()
                            val_loss += each_val_loss.item()
                    val_loss_list.append(
                        (batch_idx + 1, val_loss / len(val_dataloader.dataset))
                    )
                    print(
                        "#update: {},\tper-example "
                        "val loss:\t{}".format(
                            batch_idx + 1, val_loss / len(val_dataloader.dataset)
                        )
                    )
            batch_idx += 1
    return model, train_loss_list, val_loss_list
