import shutil
import os
import pickle
import warnings
from tqdm import tqdm

import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
import optuna

from .model import LightningModel
from .token import Token
from . import dataset, utils, plot


warnings.simplefilter("ignore")


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        progress_bar = tqdm(
            disable=True,
        )
        return progress_bar


def bayopt_trainer(
    save_dir,
    bayout_bouds,
    max_length,
    x_train,
    x_valid,
    y_train,
    y_valid,
    bayopt_n_epochs=10,
    bayopt_n_iters=10,
    precision=32,
):
    def objective(trial):
        n_lstm_units = trial.suggest_int(
            "n_lstm_units",
            bayout_bouds["lstm_units"][0],
            bayout_bouds["lstm_units"][1],
        )
        n_dense_units = trial.suggest_int(
            "n_dense_units",
            bayout_bouds["dense_units"][0],
            bayout_bouds["dense_units"][1],
        )
        n_embedding_dim = trial.suggest_int(
            "n_embedding_dim",
            bayout_bouds["embedding_dim"][0],
            bayout_bouds["embedding_dim"][1],
        )
        n_batch_size = trial.suggest_int(
            "n_batch_size",
            bayout_bouds["batch_size"][0],
            bayout_bouds["batch_size"][1],
        )
        lr = trial.suggest_float(
            "learning_rate",
            bayout_bouds["learning_rate"][0],
            bayout_bouds["learning_rate"][1],
            log=True,
        )

        train_dataloader = dataset.make_dataloader(
            x_train, y_train, batch_size=2**n_batch_size, train=True
        )
        valid_dataloader = dataset.make_dataloader(
            x_valid,
            y_valid,
            batch_size=(
                2 ** n_batch_size if len(y_valid) > 2 ** n_batch_size else len(y_valid)
            ),
        )
        if os.path.exists(os.path.join(save_dir, "bays_opt")):
            shutil.rmtree(os.path.join(save_dir, "bays_opt"))
        logger = CSVLogger(save_dir, name="bays_opt")
        opt_model = LightningModel(
            token_size=max_length,
            learning_rate=lr,
            lstm_units=2**n_lstm_units,
            dense_units=2**n_dense_units,
            embedding_dim=2**n_embedding_dim,
            log_flag=False,
        )
        trainer = pl.Trainer(
            max_epochs=bayopt_n_epochs,
            precision=precision,
            logger=logger,
            callbacks=[LitProgressBar()],
            enable_checkpointing=False,
            num_sanity_val_steps=1,
        )
        trainer.fit(
            opt_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )
        loss = trainer.logged_metrics["valid_loss"]

        return loss

    study = optuna.create_study()
    study.optimize(objective, n_trials=bayopt_n_iters)
    trial = study.best_trial

    best_hyper_param = [
        max_length,
        2 ** trial.params["n_lstm_units"],
        2 ** trial.params["n_dense_units"],
        2 ** trial.params["n_embedding_dim"],
        2 ** trial.params["n_batch_size"],
        trial.params["learning_rate"],
    ]

    return best_hyper_param


def trainer(
    save_dir,
    best_hyper_params,
    max_length,
    x_train,
    x_valid,
    y_train,
    y_valid,
    scaler,
    n_epochs=100,
    n_gpus=1,
    precision=32,
):
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
    train_dataloader = dataset.make_dataloader(
        x_train, y_train, batch_size=best_hyper_params[4], train=True
    )
    valid_dataloader = dataset.make_dataloader(
        x_valid, y_valid, batch_size=best_hyper_params[4]
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=training_dir,
        filename="best_weights",
        monitor="MeanSquaredError",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    logger = CSVLogger(save_dir, name="training")
    model = LightningModel(
        token_size=max_length,
        learning_rate=best_hyper_params[5],
        lstm_units=best_hyper_params[1],
        dense_units=best_hyper_params[2],
        embedding_dim=best_hyper_params[3],
        log_flag=True,
    )
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        precision=precision,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[model_checkpoint, LitProgressBar()],
        default_root_dir=training_dir,
        num_nodes=n_gpus,
        num_sanity_val_steps=1,
    )
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader
    )
    print("Training Finished!!!")
    return model


def main(
    data,
    data_name,
    bayopt_bounds=None,
    augmentation=False,
    outdir="../outputs",
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
    tf16=True,
    seed=42,
):
    utils.seed_everything(seed)
    if tf16:
        precision = "16-mixed"
    else:
        precision = 32

    save_dir = os.path.join(outdir, data_name)
    os.makedirs(save_dir, exist_ok=True)

    print("***Sampling and splitting of the dataset.***\n")
    smiles_train, smiles_valid, smiles_test, y_train, y_valid, y_test, scaler = (
        utils.random_split(
            smiles_input=data.smiles,
            prop_input=np.array(data.iloc[:, 1]),
            random_state=42,
            scaling=True,
        )
    )

    token = Token(
        smiles_train,
        smiles_valid,
        smiles_test,
        y_train,
        y_valid,
        y_test,
        augmentation,
        save_dir,
    )

    if bayopt_on:
        best_hyper_params = bayopt_trainer(
            save_dir,
            bayopt_bounds,
            max_length=token.max_length,
            x_train=token.enum_tokens_train,
            x_valid=token.enum_tokens_valid,
            y_train=token.enum_prop_train,
            y_valid=token.enum_prop_valid,
            bayopt_n_epochs=bayopt_n_epochs,
            bayopt_n_iters=bayopt_n_iters,
            precision=precision,
        )
    else:
        best_hyper_params = [
            token.max_length,
            lstmunits_ref,
            denseunits_ref,
            embedding_ref,
            batch_size_ref,
            lr_ref,
        ]
    print("Best Params")
    print("LSTM units       |", best_hyper_params[1])
    print("Dense units      |", best_hyper_params[2])
    print("Embedding units  |", best_hyper_params[3])
    print("Batch size       |", best_hyper_params[4])
    print("leaning rate     |", best_hyper_params[5])

    print()
    print("***Training of the best model.***\n")

    model = trainer(
        save_dir,
        best_hyper_params,
        max_length=token.max_length,
        x_train=token.enum_tokens_train,
        x_valid=token.enum_tokens_valid,
        y_train=token.enum_prop_train,
        y_valid=token.enum_prop_valid,
        scaler=scaler,
        n_epochs=n_epochs,
        n_gpus=n_gpus,
        precision=precision,
    )

    metrics_df = pd.read_csv(os.path.join(save_dir, "training/version_0/metrics.csv"))
    valid_rmse = metrics_df["MeanSquaredError"].to_list()
    valid_r2 = metrics_df["R2Score"].to_list()
    plot.plot_hitory_rmse(
        rmse=valid_rmse, r2=valid_r2, save_dir=save_dir, data_name=data_name
    )

    print(f"Best val_loss @ Epoch #{np.argmin(valid_rmse)}\n")
    print("***Predictions from the best model.***\n")
    model.eval()
    y_train, y_pred_train, mae_train, rmse_train, r2_train = model.evaluation_model(
        token.enum_tokens_train, y_train, token.enum_card_train, scaler=scaler
    )
    print("For the training set:")
    print(f"MAE: {mae_train:.4f} RMSE: {rmse_train:.4f} R^2: {r2_train:.4f}")

    y_valid, y_pred_valid, mae_valid, rmse_valid, r2_valid = model.evaluation_model(
        token.enum_tokens_valid, y_valid, token.enum_card_valid, scaler=scaler
    )
    print("For the validation set:")
    print(f"MAE: {mae_valid:.4f} RMSE: {rmse_valid:.4f} R^2: {r2_valid:.4f}")

    y_test, y_pred_test, mae_test, rmse_test, r2_test = model.evaluation_model(
        token.enum_tokens_test, y_test, token.enum_card_test, scaler=scaler
    )
    print("For the test set:")
    print(f"MAE: {mae_test:.4f} RMSE: {rmse_test:.4f} R^2: {r2_test:.4f}")

    plot.plot_obserbations_vs_predictions(
        observations=(y_train, y_valid, y_test),
        predictions=(y_pred_train, y_pred_valid, y_pred_test),
        rmse=(rmse_train, rmse_valid, rmse_test),
        r2=(r2_train, r2_valid, r2_test),
        save_dir=save_dir,
        data_name=data_name,
    )
