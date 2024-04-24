import shutil
import os
import platform
import pickle
import warnings
import logging
import argparse

from tqdm import tqdm
import yaml
import polars as pl
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from lightning import seed_everything, Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, ProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torchmetrics import (
    MetricCollection, R2Score, MeanAbsoluteError, MeanSquaredError
)

from .model import SmilesX
from . import token, augm, utils, plot, data, evaluate
from .token import Token

warnings.simplefilter("ignore")


class BayOptLoss:
    loss = None
    r2 = None
    number = 0


class LitProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()
        self.bar = None
        self.enabled = True

    def on_train_epoch_start(self, trainer, pl_module):
        if self.enabled:
            self.bar = tqdm(
                total=self.total_train_batches,
                desc=f"Epoch {trainer.current_epoch+1}",
                position=0,
                leave=True,
            )
            self.running_loss = 0.0

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if self.bar:
            self.running_loss += outputs["loss"].item()
            self.bar.update(1)
            loss = self.running_loss / self.total_train_batches
            self.bar.set_postfix(loss=f"{loss:.3f}")

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if self.bar:
            val_loss = trainer.logged_metrics["valid_loss"].item()
            val_r2 = trainer.logged_metrics["valid_r2"].item()
            loss = self.running_loss / self.total_train_batches
            self.bar.set_postfix(
                {
                    "loss": f"{loss:.3f}",
                    "val_loss": f"{val_loss:.3f}",
                    "val_r2": f"{val_r2:.3f}",
                }
            )
            self.bar.close()
            self.bar = None

    def disable(self):
        self.bar = None
        self.enabled = False


class LightningModel(LightningModule):
    def __init__(
        self,
        vocab_size,
        lstm_units=16,
        dense_units=16,
        embedding_dim=32,
        learning_rate=1e-3,
        return_proba=False,
        loss_func="MAE",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=True)
        self.lr = learning_rate
        if loss_func == "MAE":
            self.loss_func = F.l1_loss
            self.train_metrics = MetricCollection(
                {
                    "train_r2": R2Score(),
                    "train_loss": MeanAbsoluteError()
                }
            )
            self.valid_metrics = MetricCollection(
                {
                    "valid_r2": R2Score(),
                    "valid_loss": MeanAbsoluteError()
                }
            )
        else:
            self.loss_func = F.mse_loss
            self.train_metrics = MetricCollection(
                {
                    "train_r2": R2Score(),
                    "train_loss": MeanSquaredError(squared=True)
                }
            )
            self.valid_metrics = MetricCollection(
                {
                    "valid_r2": R2Score(),
                    "valid_loss": MeanSquaredError(squared=True)
                }
            )
        self.model = SmilesX(
            vocab_size, lstm_units, dense_units, embedding_dim, return_proba
        )

    def loss(self, y, y_pred):
        return self.loss_func(y, y_pred)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=self.trainer.max_epochs,
            T_mult=int(self.trainer.max_epochs * 0.1)
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, X):
        # モデルの順伝播
        return self.model(X)

    def training_step(self, batch, batch_idx):
        # モデルの学習
        X, y = batch
        output = self.forward(X)
        loss = self.loss(output, y)
        self.train_metrics(output, y)
        self.log_dict(
            dictionary=self.train_metrics,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # モデルの学習
        X, y = batch
        output = self.forward(X)
        loss = self.valid_metrics(output, y)
        self.log_dict(
            dictionary=self.valid_metrics,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        return loss


def bayopt_trainer(
    output_dir: str,
    bayopt_bounds: dict,
    vocab_size: int,
    x_train: torch.Tensor,
    x_valid: torch.Tensor,
    y_train: torch.Tensor,
    y_valid: torch.Tensor,
    batch_size=128,
    bayopt_n_epochs=10,
    bayopt_n_trials=10,
    loss_func="MAE",
    seed=42,
    precision=32,
    devices=1
):
    # Optunaの学習用関数を内部に作成
    def objective(trial: optuna.trial.Trial):
        n_lstm_units = trial.suggest_int(
            "n_lstm_units",
            bayopt_bounds["lstm_units"][0],
            bayopt_bounds["lstm_units"][1],
        )
        n_dense_units = trial.suggest_int(
            "n_dense_units",
            bayopt_bounds["dense_units"][0],
            bayopt_bounds["dense_units"][1],
        )
        n_embedding_dim = trial.suggest_int(
            "n_embedding_dim",
            bayopt_bounds["embedding_dim"][0],
            bayopt_bounds["embedding_dim"][1],
        )
        lr = trial.suggest_float(
            "learning_rate",
            float(bayopt_bounds["learning_rate"][0]),
            float(bayopt_bounds["learning_rate"][1]),
            log=True,
        )
        data_module = data.DataModule(
            x_train, x_valid, y_train, y_valid, batch_size=batch_size
        )
        opt_model = LightningModel(
            vocab_size=int(vocab_size),
            learning_rate=lr,
            lstm_units=2**n_lstm_units,
            dense_units=2**n_dense_units,
            embedding_dim=2**n_embedding_dim,
            loss_func=loss_func,
        )
        opt_model = utils.torch_compile(opt_model)
        logger = CSVLogger(output_dir, name="bayes_opt")
        trainer = Trainer(
            max_epochs=bayopt_n_epochs,
            precision=precision,
            logger=logger,
            callbacks=[LitProgressBar()],
            enable_checkpointing=False,
            devices=devices,
            strategy="auto" if devices == 1 else "ddp",
            num_sanity_val_steps=1,
            deterministic=True,
        )
        trainer.fit(opt_model, datamodule=data_module)
        loss = trainer.logged_metrics["valid_loss"]
        r2 = trainer.logged_metrics["valid_r2"]
        if BayOptLoss.loss is None:
            BayOptLoss.loss = loss
            BayOptLoss.r2 = r2
        else:
            if BayOptLoss.loss > loss:
                BayOptLoss.loss = loss
                BayOptLoss.r2 = r2
                BayOptLoss.number = trial.number
        logging.info(
            f"Trial {trial.number} finished with value: {loss} and parameters: "
            f"{trial.params}. Best is trial {BayOptLoss.number} "
            f"with value: {BayOptLoss.loss} and R2: {BayOptLoss.r2}."
        )
        return loss

    # ハイパーパラメータの探索の開始
    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=seed)
    )
    study.optimize(objective, n_trials=bayopt_n_trials, n_jobs=1)
    # 探索のうち、一番損失が少なかった条件でのハイパーパラメータを保存
    trial = study.best_trial
    logging.info(
        f"Best Trial: {trial.number} with {loss_func} value: {trial.value}"
    )
    best_hparams = {
        "vocab_size": int(vocab_size),
        "lstm_units": 2 ** trial.params["n_lstm_units"],
        "dense_units": 2 ** trial.params["n_dense_units"],
        "embedding_dim": 2 ** trial.params["n_embedding_dim"],
        "learning_rate": trial.params["learning_rate"],
    }
    return best_hparams


def trainer(
    output_dir,
    best_hparams,
    x_train: torch.Tensor,
    x_valid: torch.Tensor,
    y_train: torch.Tensor,
    y_valid: torch.Tensor,
    batch_size=128,
    n_epochs=100,
    n_early_stopping=100,
    loss_func="MAE",
    precision=32,
    devices=1
):
    training_dir = os.path.join(output_dir, "training")
    data_module = data.DataModule(
        x_train, x_valid, y_train, y_valid, batch_size
    )
    logger = CSVLogger(output_dir, name="training")
    model = LightningModel(**best_hparams, loss_func=loss_func)
    model = utils.torch_compile(model)
    model_checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        filename="training/version_0/best_weights",
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    if n_early_stopping == 0:
        n_early_stopping = n_epochs
    early_stopping = EarlyStopping(
        monitor="valid_loss",
        mode="min",
        patience=n_early_stopping
    )
    trainer = Trainer(
        max_epochs=n_epochs,
        precision=precision,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[model_checkpoint, early_stopping, LitProgressBar()],
        default_root_dir=training_dir,
        devices=devices,
        strategy="auto" if devices == 1 else "ddp",
        num_sanity_val_steps=1,
        deterministic=True,
    )
    trainer.fit(model, datamodule=data_module)
    logging.info("Training Finished!!!")
    model = LightningModel.load_from_checkpoint(
        os.path.join(output_dir, "training/version_0/best_weights.ckpt"),
        hparams_file=os.path.join(output_dir, "training/version_0/hparams.yaml"),
    )
    return model


def run(
    config_filepath="",
    devices=1,
):
    with open(config_filepath) as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)
    if devices is None:
        devices = 1

    augmentation = config["train"]["augmentation"]
    batch_size = config["train"]["batch_size"]
    bayopt_on = config["train"]["bayopt_on"]
    if bayopt_on:
        bayopt_bounds = config["bayopt_bounds"]
    else:
        lstm_units_ref = config["ref_hyperparam"]["lstm_units"]
        dense_units_ref = config["ref_hyperparam"]["dense_units"]
        embedding_dim_ref = config["ref_hyperparam"]["embedding_dim"]
        learning_rate_ref = config["ref_hyperparam"]["learning_rate"]
    bayopt_n_epochs = config["train"]["bayopt_n_epochs"]
    bayopt_n_trials = config["train"]["bayopt_n_trials"]
    n_epochs = config["train"]["n_epochs"]
    n_early_stopping = config["train"]["n_early_stopping"]
    tf16 = config["train"]["tf16"]
    loss_func = config["train"]["loss_func"]  # MSE or MAE
    seed = config["train"]["seed"]
    scaling = config["train"]["scaling"]
    dataset_path = config["dataset"]["dataset_path"]
    smiles_col = config["dataset"]["smiles_col_name"]
    prop_col = config["dataset"]["prop_col_name"]
    output_dir = config["dataset"]["output_path"]
    dataset = pl.read_csv(dataset_path)
    dataset = dataset.select(smiles_col, prop_col)
    print(dataset.head())

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    os.mkdir(log_dir)
    model_dir = os.path.join(output_dir, "model")
    os.mkdir(model_dir)

    logger, logfile = utils.log_setup(log_dir, "training", verbose=True)
    logging.info(f"OS: {platform.system()}")
    seed_everything(seed, workers=True)
    if tf16:
        precision = "16"
        logging.info(f"precision is {precision}")
        torch.set_float32_matmul_precision("medium")
    else:
        precision = "32"
        logging.info(f"precision is {precision}")
        torch.set_float32_matmul_precision("high")
    logging.info("***Sampling and splitting of the dataset.***")
    smiles_train, smiles_valid, smiles_test, y_train, y_valid, y_test, scaler = (
        data.random_split(
            smiles_input=np.array(dataset[:, 0]),
            y_input=np.array(dataset[:, 1]),
            random_state=seed,
            scaling=scaling,
        )
    )
    if scaler is not None:
        with open(os.path.join(model_dir, "scaler.pkl"), mode="wb") as f:
            pickle.dump(scaler, f)
    if smiles_train[0].count("*") == 0:
        logging.info("Setup Molecule Tokens.")
    else:
        logging.info("Setup Polymer Tokens.")

    if augmentation:
        logging.info("***Data augmentation is True.***")
        logging.info("Augmented SMILES data size:")
    else:
        logging.info("***No data augmentation has been required.***")
        logging.info("SMILES data size:")

    tokenizer = Token()
    smiles_train, enum_card_train, y_train = augm.augment_data(
        smiles_train, y_train, augmentation
    )
    tokenized_smiles_train = tokenizer.get_tokens(smiles_train)
    smiles_valid, enum_card_valid, y_valid = augm.augment_data(
        smiles_valid, y_valid, augmentation
    )
    tokenized_smiles_valid = tokenizer.get_tokens(smiles_valid)
    smiles_test, enum_card_test, y_test = augm.augment_data(
        smiles_test, y_test, augmentation
    )
    tokenized_smiles_test = tokenizer.get_tokens(smiles_test)
    logging.info(f"Training set\t| {len(y_train)}")
    logging.info(f"Validation set\t| {len(y_valid)}")
    logging.info(f"Test set\t| {len(y_test)}")

    all_tokenized_smiles = (
        tokenized_smiles_train + tokenized_smiles_valid + tokenized_smiles_test
    )
    tokens = tokenizer.extract_vocab(all_tokenized_smiles)
    vocab_size = len(tokens)
    token.check_unique_tokens(
        tokenized_smiles_train, tokenized_smiles_valid, tokenized_smiles_test
    )
    logging.info(f"Full vocabulary: {tokens}")
    logging.info(f"Of size: {vocab_size}")
    tokens, vocab_size = tokenizer.add_extra_tokens(tokens, vocab_size)
    max_length = np.max([len(i_smiles) for i_smiles in all_tokenized_smiles])
    max_length += 2  # ["unk"]と["pad"]の分
    logging.info(
        f"Maximum length of tokenized SMILES: {max_length} tokens"
        "(termination spaces included)"
    )
    int_tensor_train, y_train = tokenizer.convert_to_int_tensor(
        tokenized_smiles_train, y_train, max_length, tokens
    )
    int_tensor_valid, y_valid = tokenizer.convert_to_int_tensor(
        tokenized_smiles_valid, y_valid, max_length, tokens
    )
    int_tensor_test, y_test = tokenizer.convert_to_int_tensor(
        tokenized_smiles_test, y_test, max_length, tokens
    )
    if bayopt_on:
        optuna.logging.disable_default_handler()
        best_hparams = bayopt_trainer(
            output_dir,
            bayopt_bounds,
            vocab_size=vocab_size,
            x_train=int_tensor_train,
            x_valid=int_tensor_valid,
            y_train=y_train,
            y_valid=y_valid,
            batch_size=batch_size,
            bayopt_n_epochs=bayopt_n_epochs,
            bayopt_n_trials=bayopt_n_trials,
            precision=precision,
            loss_func=loss_func,
            devices=devices,
        )
    else:
        best_hparams = {
            "vocab_size": vocab_size,
            "lstm_units": lstm_units_ref,
            "dense_units": dense_units_ref,
            "embedding_dim": embedding_dim_ref,
            "learning_rate": learning_rate_ref,
        }
    logging.info("Best Params")
    logging.info(f"LSTM units       |{best_hparams['lstm_units']}")
    logging.info(f"Dense units      |{best_hparams['dense_units']}")
    logging.info(f"Embedding units  |{best_hparams['embedding_dim']}")
    logging.info(f"learning rate    |{best_hparams['learning_rate']}")
    logging.info("")

    config["hyper_parameters"] = {
        "model": {
            "vocab_size": best_hparams["vocab_size"],
            "lstm_units": best_hparams["lstm_units"],
            "dense_units": best_hparams["dense_units"],
            "embedding_dim": best_hparams["embedding_dim"]
        },
        "other": {
            # "batch_size": best_hparams["batch_size"],
            "learning_rate": best_hparams["learning_rate"]
        }
    }
    config["token"] = {
        "max_length": max_length,
        "vocabulary": tokens
    }
    with open(os.path.join(model_dir, "all_params.yaml"), mode="w") as f:
        yaml.dump(config, f)

    logging.info("***Training of the best model.***")
    best_model = trainer(
        output_dir,
        best_hparams,
        x_train=int_tensor_train,
        x_valid=int_tensor_valid,
        y_train=y_train,
        y_valid=y_valid,
        batch_size=batch_size,
        n_epochs=n_epochs,
        n_early_stopping=n_early_stopping,
        precision=precision,
        loss_func=loss_func,
        devices=devices
    )
    torch.save(
        obj=best_model.model.state_dict(),
        f=os.path.join(output_dir, "model/best_weights.pth")
    )

    metrics_df = pl.read_csv(
        os.path.join(output_dir, "training/version_0/metrics.csv")
    )
    train_loss = metrics_df.select("train_loss").drop_nulls().cast(pl.Float32).to_numpy()
    valid_loss = metrics_df.select("valid_loss").drop_nulls().cast(pl.Float32).to_numpy()
    train_r2 = metrics_df.select("train_r2").drop_nulls().cast(pl.Float32).to_numpy()
    valid_r2 = metrics_df.select("valid_r2").drop_nulls().cast(pl.Float32).to_numpy()
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    plot.plot_history_loss(
        train_loss,
        train_r2,
        valid_loss,
        valid_r2,
        loss_func,
        img_dir
    )

    logging.info(f"Best val_loss @ Epoch #{np.argmin(valid_loss)}")
    logging.info("***Predictions from the best model.***")

    best_model = SmilesX(
        **config["hyper_parameters"]["model"]
    )
    best_model.load_state_dict(
        torch.load(
            os.path.join(model_dir, "best_weights.pth"),
            map_location=torch.device("cpu"),
        )
    )
    evaluate.model_evaluation(
        best_model,
        img_dir,
        x_train=int_tensor_train,
        x_valid=int_tensor_valid,
        x_test=int_tensor_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        enum_card_train=enum_card_train,
        enum_card_valid=enum_card_valid,
        enum_card_test=enum_card_test,
        scaler=scaler,
        loss_func=loss_func,
    )


def main():
    parser = argparse.ArgumentParser(description="SMILES-X")
    parser.add_argument("config", help="config fileを読み込む")
    parser.add_argument("--devices")
    args = parser.parse_args()
    config_filepath = args.config
    devices = args.devices
    run(config_filepath, devices)


if __name__ == "__main__":
    main()
