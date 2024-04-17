import shutil
import os
import platform
import pickle
import warnings
import random
import logging

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

# from .models.GCN import LightningModel, MolecularGCN
from .models.GAT import LightningModel, MolecularGAT
from .features import mol2graph, augm
from . import utils, plot, dataset


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


def run(
    config_filepath="",
    outdir="./reports/",
    n_gpus=1,
    n_conv_hidden_ref=1,
    n_dense_hidden_ref=1,
    graph_dim_ref=64,
    dense_dim_ref=64,
    drop_rate_ref=0.1,
    batch_size_ref=64,
    lr_ref=1e-3,
):
    data_name = os.path.splitext(os.path.basename(config_filepath))[0]
    with open(config_filepath) as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    bayopt_on = config["train"]["bayopt_on"]
    if bayopt_on:
        bayopt_bounds = config["bayopt_bounds"]
    augmentation = config["train"]["augmentation"]
    max_n_augm = config["train"]["max_n_augm"]  # 0のとき拡張したデータをすべて利用
    bayopt_n_epochs = config["train"]["bayopt_n_epochs"]
    bayopt_n_trials = config["train"]["bayopt_n_trials"]
    n_epochs = config["train"]["n_epochs"]
    tf16 = config["train"]["tf16"]
    loss_func = config["train"]["loss_func"]  # MSE or MAE
    seed = config["train"]["seed"]
    scaling = config["train"]["scaling"]
    filepath = config["dataset"]["filepath"]
    smiles = config["dataset"]["smiles"]
    prop = config["dataset"]["prop"]
    computable_atoms = config["computable_atoms"]
    chirality = config["chirality"]
    stereochemistry = config["stereochemistry"]
    data = pd.read_csv(filepath)
    data = data[[smiles, prop]]
    print(data.head())

    save_dir = os.path.join(outdir, data_name)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "logs")
    os.mkdir(log_dir)
    model_dir = os.path.join(save_dir, "model")
    os.mkdir(model_dir)

    logger, logfile = utils.log_setup(log_dir, "training", verbose=True)
    logging.info(f"OS: {platform.system()}")
    seed_everything(seed)
    if tf16:
        # precision = "16-mixed"
        precision = 16
        logging.info(f"precision is {precision}")
        torch.set_float32_matmul_precision("medium")
    else:
        precision = 32
        logging.info(f"precision is {precision}")
        torch.set_float32_matmul_precision("high")
    logging.info("***Sampling and splitting of the dataset.***")
    smiles_train, smiles_valid, smiles_test, y_train, y_valid, y_test, scaler = (
        utils.random_split(
            smiles_input=np.array(data.iloc[:, 0]),
            y_input=np.array(data.iloc[:, 1]),
            random_state=seed,
            scaling=scaling,
        )
    )
    if scaler is not None:
        with open(os.path.join(model_dir, "scaler.pkl"), mode="wb") as f:
            pickle.dump(scaler, f)
    if smiles_train[0].count("*") == 0:
        poly_flag = False
        logging.info("Setup Molecule Tokens.")
    else:
        poly_flag = True
        logging.info("Setup Polymer Tokens.")

    if augmentation:
        logging.info("***Data augmentation is True.***")
        logging.info("Augmented SMILES data size:")
    else:
        logging.info("***No data augmentation has been required.***")
        logging.info("SMILES data size:")
    smiles_train, enum_card_train, y_train = augm.augment_data(
        smiles_train, y_train, augmentation, max_n_augm
    )
    graphs_train = mol2graph.get_graph_vectors(
        smiles_train,
        y_train,
        computable_atoms,
        poly_flag,
        chirality,
        stereochemistry
    )
    smiles_valid, enum_card_valid, y_valid = augm.augment_data(
        smiles_valid, y_valid, augmentation
    )
    graphs_valid = mol2graph.get_graph_vectors(
        smiles_valid,
        y_valid,
        computable_atoms,
        poly_flag,
        chirality,
        stereochemistry
    )
    smiles_test, enum_card_test, y_test = augm.augment_data(
        smiles_test, y_test, augmentation
    )
    graphs_test = mol2graph.get_graph_vectors(
        smiles_test,
        y_test,
        computable_atoms,
        poly_flag,
        chirality,
        stereochemistry
    )
    logging.info(f"\tTraining set   | {len(y_train)}")
    logging.info(f"\tValidation set | {len(y_valid)}")
    logging.info(f"\tTest set       | {len(y_test)}")
    n_features = len(graphs_train[0]["x"][0])

    if bayopt_on:
        optuna.logging.disable_default_handler()
        best_hparams = bayopt_trainer(
            save_dir,
            n_features,
            bayopt_bounds,
            graphs_train,
            graphs_valid,
            bayopt_n_epochs=bayopt_n_epochs,
            bayopt_n_trials=bayopt_n_trials,
            precision=precision,
            loss_func=loss_func,
            n_gpus=n_gpus,
            seed=seed
        )
    else:
        best_hparams = {
            "n_features": int(n_features),
            "n_conv_hidden_layer": n_conv_hidden_ref,
            "n_dense_hidden_layer": n_dense_hidden_ref,
            "graph_dim": graph_dim_ref,
            "dense_dim": dense_dim_ref,
            "drop_rate": drop_rate_ref,
            "batch_size": batch_size_ref,
            "learning_rate": lr_ref,
        }
    logging.info("Best Params")
    logging.info(f"Conv hidden layers  |{best_hparams['n_conv_hidden_layer']}")
    logging.info(f"Dense hidden layers |{best_hparams['n_dense_hidden_layer']}")
    logging.info(f"Graph dim           |{best_hparams['graph_dim']}")
    logging.info(f"Dense dim           |{best_hparams['dense_dim']}")
    logging.info(f"Drop rate           |{best_hparams['drop_rate']}")
    logging.info(f"Batch size          |{best_hparams['batch_size']}")
    logging.info(f"learning rate       |{best_hparams['learning_rate']}")
    logging.info("")

    config["hyper_parameters"] = {
        "model": {
            "n_features": best_hparams["n_features"],
            "n_conv_hidden_layer": best_hparams["n_conv_hidden_layer"],
            "n_dense_hidden_layer": best_hparams["n_dense_hidden_layer"],
            "graph_dim": best_hparams["graph_dim"],
            "dense_dim": best_hparams["dense_dim"],
            "drop_rate": best_hparams["drop_rate"]
        },
        "other": {
            "batch_size": best_hparams["batch_size"],
            "learning_rate": best_hparams["learning_rate"]
        }
    }
    with open(os.path.join(model_dir, "all_params.yaml"), mode="w") as f:
        yaml.dump(config, f)

    logging.info("***Training of the best model.***")
    trainer(
        save_dir,
        best_hparams,
        graphs_train=graphs_train,
        graphs_valid=graphs_valid,
        n_epochs=n_epochs,
        n_gpus=n_gpus,
        precision=precision,
        loss_func=loss_func,
    )

    metrics_df = pd.read_csv(
        os.path.join(save_dir, "training/version_0/metrics.csv")
    )
    train_loss = metrics_df["train_loss"].dropna()
    valid_loss = metrics_df["valid_loss"].dropna()
    train_r2 = metrics_df["train_r2"].dropna()
    valid_r2 = metrics_df["valid_r2"].dropna()
    img_dir = os.path.join(save_dir, "images")
    os.mkdir(img_dir)
    plot.plot_history_loss(
        train_loss,
        train_r2,
        valid_loss,
        valid_r2,
        loss_func,
        img_dir,
        data_name
    )

    logging.info(f"Best val_loss @ Epoch #{np.argmin(valid_loss)}")
    logging.info("***Predictions from the best model.***")

    # best_model = MolecularGCN(
    #     **config["hyper_parameters"]["model"]
    # )
    best_model = MolecularGAT(
        **config["hyper_parameters"]["model"]
    )
    best_model.load_state_dict(
        torch.load(
            os.path.join(model_dir, "best_weights.pth"),
            map_location=torch.device("cpu"),
        )
    )
    utils.model_evaluation(
        img_dir,
        data_name,
        model=best_model,
        graphs_train=graphs_train,
        graphs_valid=graphs_valid,
        graphs_test=graphs_test,
        enum_card_train=enum_card_train,
        enum_card_valid=enum_card_valid,
        enum_card_test=enum_card_test,
        scaler=scaler,
        loss_func=loss_func,
    )


def bayopt_trainer(
    save_dir: str,
    n_features,
    bayopt_bounds: dict,
    graphs_train,
    graphs_valid,
    bayopt_n_epochs=10,
    bayopt_n_trials=10,
    precision=32,
    loss_func="MAE",
    n_gpus=1,
    seed=42,
):
    """Optunaによるベイズ最適化によりハイパーパラメータを決定するための訓練を行う関数

    Parameters
    ----------
    save_dir : str
        _description_
    bayopt_bounds : dict
        ハイパーパラメータの探索範囲
    vocab_size : int
        SMILESの最大長
    graphs_train :
        訓練に用いるための
    graphs_valid :
        検証に用いるための
    bayopt_n_epochs : int, optional
        ハイパーパラメータの探索のための学習エポック, by default 10
    bayopt_n_trials : int, optional
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
        n_conv_hidden_layer = trial.suggest_int(
            "n_conv_hidden_layer",
            bayopt_bounds["n_conv_hidden_layer"][0],
            bayopt_bounds["n_conv_hidden_layer"][1],
        )
        n_dense_hidden_layer = trial.suggest_int(
            "n_dense_hidden_layer",
            bayopt_bounds["n_dense_hidden_layer"][0],
            bayopt_bounds["n_dense_hidden_layer"][1],
        )
        n_graph_dim = trial.suggest_int(
            "n_graph_dim",
            bayopt_bounds["graph_dim"][0],
            bayopt_bounds["graph_dim"][1],
        )
        n_dense_dim = trial.suggest_int(
            "n_dense_dim",
            bayopt_bounds["dense_dim"][0],
            bayopt_bounds["dense_dim"][1],
        )
        drop_rate = trial.suggest_discrete_uniform(
            "drop_rate",
            bayopt_bounds["drop_rate"][0],
            bayopt_bounds["drop_rate"][1],
            bayopt_bounds["drop_rate"][2]
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
            graphs_train, graphs_valid, batch_size=2**n_batch_size
        )
        opt_model = LightningModel(
            n_features,
            n_conv_hidden_layer,
            n_dense_hidden_layer,
            2**n_graph_dim,
            2**n_dense_dim,
            drop_rate,
            lr,
            loss_func,
        )
        logger = CSVLogger(save_dir, name="bayes_opt")
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
        "n_features": int(n_features),
        "n_conv_hidden_layer": trial.params["n_conv_hidden_layer"],
        "n_dense_hidden_layer": trial.params["n_dense_hidden_layer"],
        "graph_dim": 2 ** trial.params["n_graph_dim"],
        "dense_dim": 2 ** trial.params["n_dense_dim"],
        "drop_rate": trial.params["drop_rate"],
        "batch_size": 2 ** trial.params["n_batch_size"],
        "learning_rate": trial.params["learning_rate"],
    }
    return best_hparams


def trainer(
    save_dir,
    best_hparams,
    graphs_train,
    graphs_valid,
    n_epochs=100,
    n_gpus=1,
    precision=32,
    loss_func="MAE",
):
    training_dir = os.path.join(save_dir, "training")
    data_module = dataset.DataModule(
        graphs_train, graphs_valid, best_hparams["batch_size"]
    )
    logger = CSVLogger(save_dir, name="training")

    model = LightningModel(
        n_features=best_hparams["n_features"],
        n_conv_hidden_layer=best_hparams["n_conv_hidden_layer"],
        n_dense_hidden_layer=best_hparams["n_dense_hidden_layer"],
        graph_dim=best_hparams["graph_dim"],
        dense_dim=best_hparams["dense_dim"],
        drop_rate=best_hparams["drop_rate"],
        learning_rate=best_hparams["learning_rate"],
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
        patience=20
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
        deterministic=True,
    )
    trainer.fit(model, datamodule=data_module)
    model = LightningModel.load_from_checkpoint(
        os.path.join(save_dir, "training/version_0/best_weights.ckpt"),
        hparams_file=os.path.join(save_dir, "training/version_0/hparams.yaml"),
    )
    torch.save(
        obj=model.model.state_dict(),
        f=os.path.join(save_dir, "model/best_weights.pth")
    )
    logging.info("Training Finished!!!")


def seed_everything(seed=1):
    random.seed(seed)  # Python標準のrandomモジュールのシードを設定
    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # ハッシュ生成のためのシードを環境変数に設定
    np.random.seed(seed)  # NumPyの乱数生成器のシードを設定
    torch.manual_seed(seed)  # PyTorchの乱数生成器のシードをCPU用に設定
    torch.cuda.manual_seed(seed)  # PyTorchの乱数生成器のシードをGPU用に設定
    torch.backends.cudnn.deterministic = True  # PyTorchの畳み込み演算の再現性を確保
    L.seed_everything(seed, workers=True)
