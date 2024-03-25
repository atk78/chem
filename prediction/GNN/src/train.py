import shutil
import os
import platform
import joblib
import pickle
import warnings
import random

from tqdm import tqdm
import yaml
import pandas as pd
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import optuna

from .models.LSTMAttention import LightningModel, LSTMAttention
from .token import TrainToken
from . import utils, plot, dataset


warnings.simplefilter("ignore")


def seed_everything(seed=1):
    random.seed(seed)  # Python標準のrandomモジュールのシードを設定
    os.environ["PYTHONHASHSEED"] = str(seed)  # ハッシュ生成のためのシードを環境変数に設定
    np.random.seed(seed)  # NumPyの乱数生成器のシードを設定
    torch.manual_seed(seed)  # PyTorchの乱数生成器のシードをCPU用に設定
    torch.cuda.manual_seed(seed)  # PyTorchの乱数生成器のシードをGPU用に設定
    torch.backends.cudnn.deterministic = True  # PyTorchの畳み込み演算の再現性を確保
    L.seed_everything(seed, workers=True)


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

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
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
            self.bar.set_postfix({"loss": f"{loss:.3f}", "val_loss": f"{val_loss:.3f}", "val_r2": f"{val_r2:.3f}"})
            # self.bar.set_postfix(loss=f"{loss:.3f}", val_loss=f"{val_loss:.3f}")
            self.bar.close()
            self.bar = None

    def disable(self):
        self.bar = None
        self.enabled = False


def bayopt_trainer(
    save_dir: str,
    bayopt_bounds: dict,
    max_length: int,
    x_train: torch.Tensor,
    x_valid: torch.Tensor,
    y_train: torch.Tensor,
    y_valid: torch.Tensor,
    bayopt_n_epochs=10,
    bayopt_n_iters=10,
    precision=32,
    loss_func="MAE",
    n_gpus=1,
    seed=42
):
    """Optunaによるベイズ最適化によりハイパーパラメータを決定するための訓練を行う関数

    Parameters
    ----------
    save_dir : str
        _description_
    bayopt_bounds : dict
        ハイパーパラメータの探索範囲
    max_length : int
        SMILESの最大長
    x_train : TensorType[n_train_samples, max_length]
        訓練に用いるためのトークン化されたSMILES
    x_valid : TensorType[n_valid_samples, max_length]
        検証に用いるためのトークン化されたSMILES
    y_train : TensorType[n_train_samples, 1]
        訓練に用いるための物性データ
    y_valid : TensorType[n_valid_smaples, 1]
        検証に用いるための物性データ
    bayopt_n_epochs : int, optional
        ハイパーパラメータの探索のための学習エポック, by default 10
    bayopt_n_iters : int, optional
        ハイパーパラメータの探索回数, by default 10
    precision : int, optional
        浮動小数点の精度 32だとfloat32を用い、16だとfloat16を用いる, by default 32
    loss_func : str, optional
        損失関数の種類 MAEとRMSEが選択できる, by default "MAE"
    n_gpus : int, optional
        GPUの並列数, by default 1
    seed : int, optional
        ランダムシード値, by default 42

    Returns
    -------
    _type_
        _description_
    """
    # Optunaの学習用関数を内部に作成
    def objective(trial):
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
        n_batch_size = trial.suggest_int(
            "n_batch_size",
            bayopt_bounds["batch_size"][0],
            bayopt_bounds["batch_size"][1],
        )
        lr = trial.suggest_float(
            "learning_rate",
            float(bayopt_bounds["learning_rate"][0]),
            float(bayopt_bounds["learning_rate"][1]),
            log=True,
        )
        data_module = dataset.DataModule(
            x_train, x_valid, y_train, y_valid, batch_size=2 ** n_batch_size
        )
        opt_model = LightningModel(
            token_size=int(max_length),
            learning_rate=lr,
            lstm_units=2**n_lstm_units,
            dense_units=2**n_dense_units,
            embedding_dim=2**n_embedding_dim,
            loss_func=loss_func,
        )
        logger = CSVLogger(save_dir, name="bays_opt")
        if platform.system() != "Windows":
            opt_model = torch.compile(opt_model, mode="reduce-overhead")
        trainer = L.Trainer(
            max_epochs=bayopt_n_epochs,
            precision=precision,
            logger=logger,
            callbacks=[LitProgressBar()],
            enable_checkpointing=False,
            devices=n_gpus,
            num_sanity_val_steps=1,
            deterministic=True
        )
        trainer.fit(opt_model, datamodule=data_module)
        loss = trainer.logged_metrics["valid_loss"]

        return loss
    # ハイパーパラメータの探索の開始
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    study.optimize(objective, n_trials=bayopt_n_iters)
    # 探索のうち、一番損失が少なかった条件でのハイパーパラメータを保存
    trial = study.best_trial
    best_hparams = {
        "max_length": int(max_length),
        "lstm_units": 2 ** trial.params["n_lstm_units"],
        "dense_units": 2 ** trial.params["n_dense_units"],
        "embedding_dim": 2 ** trial.params["n_embedding_dim"],
        "batch_size": 2 ** trial.params["n_batch_size"],
        "learning_rate": trial.params["learning_rate"]
    }

    return best_hparams


def trainer(
    save_dir,
    best_hparams,
    x_train: torch.Tensor,
    x_valid: torch.Tensor,
    y_train: torch.Tensor,
    y_valid: torch.Tensor,
    n_epochs=100,
    n_gpus=1,
    precision=32,
    loss_func="MAE",
):
    training_dir = os.path.join(save_dir, "training")
    data_module = dataset.DataModule(
            x_train, x_valid, y_train, y_valid, batch_size=best_hparams["batch_size"]
    )
    logger = CSVLogger(save_dir, name="training")
    model = LightningModel(
        token_size=best_hparams["max_length"],
        learning_rate=best_hparams["learning_rate"],
        lstm_units=best_hparams["lstm_units"],
        dense_units=best_hparams["dense_units"],
        embedding_dim=best_hparams["embedding_dim"],
        loss_func=loss_func,
    )
    if platform.system() != "Windows":
        model = torch.compile(model, mode="reduce-overhead")  # linuxの場合
    model_checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        filename="training/version_0/best_weights",
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    early_stopping = EarlyStopping(
        monitor="valid_loss",
        mode="min",
        patience=10
    )
    trainer = L.Trainer(
        max_epochs=n_epochs,
        precision=precision,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[model_checkpoint, early_stopping, LitProgressBar()],
        default_root_dir=training_dir,
        devices=n_gpus,
        num_sanity_val_steps=1,
        deterministic=True
    )
    trainer.fit(model, datamodule=data_module)
    model = LightningModel.load_from_checkpoint(
        checkpoint_path=os.path.join(save_dir, "training/version_0/best_weights.ckpt"),
        hparams_file=os.path.join(save_dir, "training/version_0/hparams.yaml")
    )
    torch.save(model.model.state_dict(), os.path.join(save_dir, "model/best_weights.pth"))
    print("Training Finished!!!")


def main(
    data: pd.DataFrame,
    data_name="test",
    bayopt_bounds=None,
    augmentation=False,
    outdir="../reports",
    bayopt_n_epochs=3,
    bayopt_n_iters=25,
    bayopt_on=True,
    lstmunits_ref=512,
    denseunits_ref=512,
    embedding_ref=512,
    batch_size_ref=64,
    lr_ref=1e-3,
    n_gpus=1,
    n_epochs=100,
    tf16=True,
    loss_func="MAE",
    scaling=True,
    seed=42,
):
    print(f"OS: {platform.system()}")
    seed_everything(seed)
    if tf16:
        # precision = "16-mixed"
        precision = 16
        print("precision is", precision)
        torch.set_float32_matmul_precision("medium")
    else:
        precision = 32
        print("precision is", precision)
        torch.set_float32_matmul_precision("high")

    save_dir = os.path.join(outdir, data_name)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    model_dir = os.path.join(save_dir, "model")
    os.makedirs(model_dir)
    with open(os.path.join(model_dir, "bayopt_bounds.yaml"), "w") as f:
        yaml.dump(bayopt_bounds, f, default_flow_style=False, allow_unicode=True)

    print("***Sampling and splitting of the dataset.***\n")
    smiles_train, smiles_valid, smiles_test, y_train, y_valid, y_test, scaler = (
        utils.random_split(
            smiles_input=np.array(data.iloc[:, 0]),
            prop_input=np.array(data.iloc[:, 1]),
            random_state=seed,
            scaling=scaling
        )
    )
    if scaler is not None:
        with open(os.path.join(model_dir, "scaler.pkl"), mode="wb") as f:
            pickle.dump(scaler, f)
    if smiles_train[0].count("*") == 0:
        poly = False
    else:
        poly = True

    token = TrainToken(
        smiles_train,
        smiles_valid,
        smiles_test,
        y_train,
        y_valid,
        y_test,
        augmentation,
        model_dir,
        poly,
    )
    token.setup()
    joblib.dump(token, filename=os.path.join(model_dir, "token.pkl.cmp"), compress=3)
    if bayopt_on:
        best_hparams = bayopt_trainer(
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
            loss_func=loss_func,
            n_gpus=n_gpus
        )
    else:
        best_hparams = {
            "max_length": token.max_length,
            "lstm_units": lstmunits_ref,
            "dense_units": denseunits_ref,
            "embedding_dim": embedding_ref,
            "batch_size": batch_size_ref,
            "learning_rate": lr_ref
        }
    print("Best Params")
    print("LSTM units       |", best_hparams["lstm_units"])
    print("Dense units      |", best_hparams["dense_units"])
    print("Embedding units  |", best_hparams["embedding_dim"])
    print("Batch size       |", best_hparams["batch_size"])
    print("learning rate     |", best_hparams["learning_rate"])
    print()
    with open(os.path.join(model_dir, "best_hparams.yaml"), mode="w") as f:
        yaml.dump(best_hparams, f)

    print("***Training of the best model.***\n")
    trainer(
        save_dir,
        best_hparams,
        x_train=token.enum_tokens_train,
        x_valid=token.enum_tokens_valid,
        y_train=token.enum_prop_train,
        y_valid=token.enum_prop_valid,
        n_epochs=n_epochs,
        n_gpus=n_gpus,
        precision=precision,
        loss_func=loss_func
    )

    metrics_df = pd.read_csv(os.path.join(save_dir, "training/version_0/metrics.csv"))
    train_loss = metrics_df["train_loss"].dropna()
    valid_loss = metrics_df["valid_loss"].dropna()
    train_r2 = metrics_df["train_r2"].dropna()
    valid_r2 = metrics_df["valid_r2"].dropna()
    plot.plot_history_loss(
        train_loss=train_loss,
        train_r2=train_r2,
        valid_loss=valid_loss,
        valid_r2=valid_r2,
        loss_func=loss_func,
        save_dir=save_dir,
        data_name=data_name
    )

    print(f"Best val_loss @ Epoch #{np.argmin(valid_loss)}\n")
    print("***Predictions from the best model.***\n")

    model = LSTMAttention(
        token_size=best_hparams["max_length"],
        lstm_units=best_hparams["lstm_units"],
        dense_units=best_hparams["dense_units"],
        embedding_dim=best_hparams["embedding_dim"]
    )
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_weights.pth"), map_location=torch.device("cpu")))
    y_train, y_pred_train, mae_train, rmse_train, r2_train = utils.evaluation_model(
        model, token.enum_tokens_train, token.enum_prop_train, token.enum_card_train, scaler
    )
    print("For the training set:")
    print(f"MAE: {mae_train:.4f} RMSE: {rmse_train:.4f} R^2: {r2_train:.4f}")

    y_valid, y_pred_valid, mae_valid, rmse_valid, r2_valid = utils.evaluation_model(
        model, token.enum_tokens_valid, token.enum_prop_valid, token.enum_card_valid, scaler
    )
    print("For the validation set:")
    print(f"MAE: {mae_valid:.4f} RMSE: {rmse_valid:.4f} R^2: {r2_valid:.4f}")

    y_test, y_pred_test, mae_test, rmse_test, r2_test = utils.evaluation_model(
        model, token.enum_tokens_test, token.enum_prop_test, token.enum_card_test, scaler
    )
    print("For the test set:")
    print(f"MAE: {mae_test:.4f} RMSE: {rmse_test:.4f} R^2: {r2_test:.4f}")

    if loss_func == "MAE":
        loss_train, loss_valid, loss_test = mae_train, mae_valid, mae_test
    else:
        loss_train, loss_valid, loss_test = rmse_train, rmse_valid, rmse_test

    plot.plot_obserbations_vs_predictions(
        observations=(y_train, y_valid, y_test),
        predictions=(y_pred_train, y_pred_valid, y_pred_test),
        loss=(loss_train, loss_valid, loss_test),
        r2=(r2_train, r2_valid, r2_test),
        loss_func=loss_func,
        save_dir=save_dir,
        data_name=data_name,
    )
