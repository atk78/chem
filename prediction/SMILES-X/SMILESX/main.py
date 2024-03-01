import os
import shutil
import pickle

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from SMILESX import utils, utils, token, augm, datas, bayopt, plot
from SMILESX.model import LSTMAttention


def Main(
    data,
    data_name,
    bayopt_bounds=None,
    augmentation=False,
    outdir="../outputs/",
    bayopt_n_epochs=3,
    bayopt_n_iters=25,
    bayopt_on=True,
    lstmunits_ref=512,
    denseunits_ref=512,
    embedding_ref=512,
    batch_size_ref=64,
    lr_ref=3,
    n_gpus=1,
    n_epochs=100,
    tf16=True
):
    if augmentation:
        p_dir_temp = "Augm"
    else:
        p_dir_temp = "Can"

    if tf16:
        precision="16-mixed"
    else:
        precision="32"

    save_dir = outdir + "{}/{}/".format(data_name, p_dir_temp)
    os.makedirs(save_dir, exist_ok=True)

    print("***Sampling and splitting of the dataset.***\n")
    x_train, x_valid, x_test, y_train, y_valid, y_test, scaler = utils.random_split(
        smiles_input=data.smiles,
        prop_input=np.array(data.iloc[:, 1]),
        random_state=42,
        scaling=True,
    )

    # data augmentation or not
    if augmentation == True:
        print("***Data augmentation to {}***\n".format(augmentation))
        canonical = False
        rotation = True
    else:
        print("***No data augmentation has been required.***\n")
        canonical = True
        rotation = False

    x_train_enum, x_train_enum_card, y_train_enum = augm.Augmentation(
        x_train, y_train, canon=canonical, rotate=rotation
    )

    x_valid_enum, x_valid_enum_card, y_valid_enum = augm.Augmentation(
        x_valid, y_valid, canon=canonical, rotate=rotation
    )

    x_test_enum, x_test_enum_card, y_test_enum = augm.Augmentation(
        x_test, y_test, canon=canonical, rotate=rotation
    )

    print(
        "Enumerated SMILES:\n\tTraining set: {}\n\tValidation set: {}\n\tTest set: {}\n".format(
            x_train_enum.shape[0], x_valid_enum.shape[0], x_test_enum.shape[0]
        )
    )

    print("***Tokenization of SMILES.***\n")
    # Tokenize SMILES per dataset
    x_train_enum_tokens = token.get_tokens(x_train_enum)
    x_valid_enum_tokens = token.get_tokens(x_valid_enum)
    x_test_enum_tokens = token.get_tokens(x_test_enum)

    print(
        "Examples of tokenized SMILES from a training set:\n{}\n".format(
            x_train_enum_tokens[:5]
        )
    )

    # Vocabulary size computation
    all_smiles_tokens = x_train_enum_tokens + x_valid_enum_tokens + x_test_enum_tokens

    # Check if the vocabulary for current dataset exists already
    if os.path.exists(save_dir + data_name + "_Vocabulary.txt"):
        tokens = token.get_vocab(save_dir + data_name + "_Vocabulary.txt")
    else:
        tokens = token.extract_vocab(all_smiles_tokens)
        token.save_vocab(tokens, save_dir + data_name + "_Vocabulary.txt")
        tokens = token.get_vocab(save_dir + data_name + "_Vocabulary.txt")

    vocab_size = len(tokens)

    train_unique_tokens = token.extract_vocab(x_train_enum_tokens)
    print(
        "Number of tokens only present in a training set: {}\n".format(
            len(train_unique_tokens)
        )
    )
    valid_unique_tokens = token.extract_vocab(x_valid_enum_tokens)
    print(
        "Number of tokens only present in a validation set: {}".format(
            len(valid_unique_tokens)
        )
    )
    print(
        "Is the validation set a subset of the training set: {}".format(
            valid_unique_tokens.issubset(train_unique_tokens)
        )
    )
    print(
        "What are the tokens by which they differ: {}\n".format(
            valid_unique_tokens.difference(train_unique_tokens)
        )
    )
    test_unique_tokens = token.extract_vocab(x_test_enum_tokens)
    print(
        "Number of tokens only present in a test set: {}".format(
            len(test_unique_tokens)
        )
    )
    print(
        "Is the test set a subset of the training set: {}".format(
            test_unique_tokens.issubset(train_unique_tokens)
        )
    )
    print(
        "What are the tokens by which they differ: {}".format(
            test_unique_tokens.difference(train_unique_tokens)
        )
    )
    print(
        "Is the test set a subset of the validation set: {}".format(
            test_unique_tokens.issubset(valid_unique_tokens)
        )
    )
    print(
        "What are the tokens by which they differ: {}\n".format(
            test_unique_tokens.difference(valid_unique_tokens)
        )
    )

    print("Full vocabulary: {}\nOf size: {}\n".format(tokens, vocab_size))

    # Add 'pad', 'unk' tokens to the existing list
    tokens, vocab_size = token.add_extra_tokens(tokens, vocab_size)

    # Maximum of length of SMILES to process
    max_length = np.max([len(ismiles) for ismiles in all_smiles_tokens])
    print(
        "Maximum length of tokenized SMILES: {} tokens (termination spaces included)\n".format(
            max_length
        )
    )

    print("***Bayesian Optimization of the SMILESX's architecture.***\n")

    x_train_enum_tokens, y_train_enum = token.convert_enum_tokens_to_torch_tensor(
        x_train_enum_tokens, y_train_enum, max_length + 1, tokens
    )

    x_valid_enum_tokens, y_valid_enum = token.convert_enum_tokens_to_torch_tensor(
        x_valid_enum_tokens, y_valid_enum, max_length + 1, tokens
    )

    x_test_enum_tokens, y_test_enum = token.convert_enum_tokens_to_torch_tensor(
        x_test_enum_tokens, y_test_enum, max_length + 1, tokens
    )

    if bayopt_on:
        best_hyper_params = bayopt.bayopt_trainer(
            save_dir,
            bayopt_bounds,
            max_length,
            x_train_enum_tokens,
            x_valid_enum_tokens,
            y_train_enum,
            y_valid_enum,
            bayopt_n_epochs=bayopt_n_epochs,
            bayopt_n_iters=bayopt_n_iters,
            precision=precision
        )
    else:
        best_hyper_params = [
            max_length+1,
            lstmunits_ref,
            denseunits_ref,
            embedding_ref,
            batch_size_ref,
            lr_ref
        ]
    print("Best Params")
    print("LSTM units       |", best_hyper_params[1])
    print("Dense units      |", best_hyper_params[2])
    print("Embedding units  |", best_hyper_params[3])
    print("Batch size       |", best_hyper_params[4])
    print("leaning rate     |", best_hyper_params[5])

    print()
    print("***Training of the best model.***\n")

    training_dir = os.path.join(save_dir, "training")

    if os.path.exists(training_dir):
        shutil.rmtree(training_dir)
        os.makedirs(training_dir)
    else:
        os.makedirs(training_dir)

    with open(os.path.join(training_dir, "best_hyper_params.pkl"), mode="wb") as f:
        pickle.dump(best_hyper_params, f)

    with open(os.path.join(training_dir, "scaler.pkl"), mode="wb") as f:
        pickle.dump(scaler, f)

    train_dataloader = datas.make_dataloader(
        x_train_enum_tokens, y_train_enum, batch_size=best_hyper_params[4], train=True
    )
    valid_dataloader = datas.make_dataloader(
        x_valid_enum_tokens, y_valid_enum, batch_size=best_hyper_params[4]
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=training_dir,
        filename="best_weights",
        monitor="MeanSquaredError",
        mode="min",
        save_top_k=1,
        save_last=False
    )
    logger = CSVLogger(save_dir, name="training")
    model = LSTMAttention(
        token_size=max_length + 1,
        learning_rate=best_hyper_params[5],
        lstm_units=best_hyper_params[1],
        dense_units=best_hyper_params[2],
        embedding_dim=best_hyper_params[3],
        log_flag=True
    )
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        precision=precision,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[model_checkpoint],
        default_root_dir=training_dir
    )
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader
    )

    print("Training Finished!!!")

    # train_rmse = np.array(model.train_metrics_epoch["MeanAbsoluteError"])
    # valid_rmse = np.array(model.valid_metrics_epoch["MeanAbsoluteError"])
    metrics_df = pd.read_csv(os.path.join(save_dir, "training/version_0/metrics.csv"))
    valid_rmse = metrics_df["MeanSquaredError"].to_list()
    valid_r2 = metrics_df["R2Score"].to_list()
    plot.plot_hitory_rmse(
        rmse=valid_rmse, r2=valid_r2, save_dir=save_dir, data_name=data_name
    )

    print(f"Best val_loss @ Epoch #{np.argmin(valid_rmse)}\n")

    print("***Predictions from the best model.***\n")

    model.eval()
    with torch.no_grad():
        y_pred_train = model(x_train_enum_tokens).detach().numpy()
    x_train_enum_card = np.array(x_train_enum_card)
    y_pred_train, _ = utils.mean_median_result(x_train_enum_card, y_pred_train)
    y_pred_train = y_pred_train.reshape(-1, 1)
    y_train = scaler.inverse_transform(y_train)
    y_pred_train = scaler.inverse_transform(y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = mean_squared_error(y_train, y_pred_train, squared=True)
    r2_train = r2_score(y_train, y_pred_train)
    print("For the training set:")
    print(f"MAE: {mae_train:.4f} RMSE: {rmse_train:.4f} R^2: {r2_train:.4f}")

    with torch.no_grad():
        y_pred_valid = model(x_valid_enum_tokens).detach().numpy()
    x_valid_enum_card = np.array(x_valid_enum_card)
    y_pred_valid, _ = utils.mean_median_result(x_valid_enum_card, y_pred_valid)
    y_pred_valid = y_pred_valid.reshape(-1, 1)
    y_valid = scaler.inverse_transform(y_valid)
    y_pred_valid = scaler.inverse_transform(y_pred_valid)
    mae_valid = mean_absolute_error(y_valid, y_pred_valid)
    rmse_valid = mean_squared_error(y_valid, y_pred_valid, squared=True)
    r2_valid = r2_score(y_valid, y_pred_valid)
    print("For the validing set:")
    print(f"MAE: {mae_valid:.4f} RMSE: {rmse_valid:.4f} R^2: {r2_valid:.4f}")

    with torch.no_grad():
        y_pred_test = model(x_test_enum_tokens).detach().numpy()
    x_test_enum_card = np.array(x_test_enum_card)
    y_pred_test, _ = utils.mean_median_result(x_test_enum_card, y_pred_test)
    y_pred_test = y_pred_test.reshape(-1, 1)
    y_test = scaler.inverse_transform(y_test)
    y_pred_test = scaler.inverse_transform(y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=True)
    r2_test = r2_score(y_test, y_pred_test)
    print("For the testing set:")
    print(f"MAE: {mae_test:.4f} RMSE: {rmse_test:.4f} R^2: {r2_test:.4f}")

    plot.plot_obserbations_vs_predictions(
        observations=(y_train, y_valid, y_test),
        predictions=(y_pred_train, y_pred_valid, y_pred_test),
        rmse=(rmse_train, rmse_valid, rmse_test),
        r2=(r2_train, r2_valid, r2_test),
        save_dir=save_dir, data_name=data_name
    )
