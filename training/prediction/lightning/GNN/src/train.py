import shutil
import os
import platform
import pickle
import warnings
import logging
from pathlib import Path

from tqdm import tqdm
import yaml
import polars as pl
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from lightning import Trainer, LightningModule, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, ProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torchmetrics import (
    MetricCollection, R2Score, MeanAbsoluteError, MeanSquaredError
)

from src.model import MolecularGNN
from src.mol2graph import Mol2Graph
from src import utils, plot, data, augm, evaluate


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
        n_features: int,
        n_conv_hidden_layer=3,
        n_dense_hidden_layer=3,
        graph_dim=64,
        dense_dim=64,
        drop_rate=0.1,
        gnn_type="GAT",
        learning_rate=1e-3,
        loss_func="MAE",
        num_of_outputs=1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=True)
        self.num_of_outputs = num_of_outputs
        self.lr = learning_rate
        self.model = MolecularGNN(
            n_features,
            n_conv_hidden_layer,
            n_dense_hidden_layer,
            graph_dim,
            dense_dim,
            drop_rate,
            gnn_type,
            num_of_outputs
        )
        if loss_func == "MAE":
            self.loss_func = F.l1_loss
            self.train_metrics = MetricCollection({"train_loss": MeanAbsoluteError()})
            self.valid_metrics = MetricCollection({"valid_loss": MeanAbsoluteError()})
        else:
            self.loss_func = F.mse_loss
            self.train_metrics = MetricCollection({"train_loss": MeanSquaredError(squared=True)})
            self.valid_metrics = MetricCollection({"valid_loss": MeanSquaredError(squared=True)})
        self.train_metrics["train_r2"] = R2Score()
        self.valid_metrics["valid_r2"] = R2Score()

    def loss(self, y, y_pred):
        return self.loss_func(y, y_pred)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        t = int(self.trainer.max_epochs * 0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=self.trainer.max_epochs,
            T_mult=t if t > 0 else 1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, X, training=True):
        # モデルの順伝播
        return self.model.forward(X, training)

    def training_step(self, batch, batch_idx):
        # モデルの学習
        X, y = batch, batch.y
        output = self.forward(X, training=True)
        loss = self.loss(output, y)
        if self.num_of_outputs > 1:
            y = y.pow(2).sum(dim=1).pow(0.5).reshape(-1, 1)
            output = output.pow(2).sum(dim=1).pow(0.5).reshape(-1, 1)
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
        X, y = batch, batch.y
        output = self.forward(X, training=False)
        loss = self.valid_metrics(output, y)
        if self.num_of_outputs > 1:
            y = y.pow(2).sum(dim=1).pow(0.5).reshape(-1, 1)
            output = output.pow(2).sum(dim=1).pow(0.5).reshape(-1, 1)
        self.log_dict(
            dictionary=self.valid_metrics,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )
        return loss


def run(
    config_filepath: str,
    n_gpus=1,
    n_conv_hidden_ref=1,
    n_dense_hidden_ref=1,
    graph_dim_ref=64,
    dense_dim_ref=64,
    drop_rate_ref=0.1,
    lr_ref=1e-3,
):
    with open(config_filepath) as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    bayopt_on = config["train"]["bayopt_on"]
    if bayopt_on:
        bayopt_bounds = config["bayopt_bounds"]
    augmentation = config["train"]["augmentation"]
    max_n_augm = config["train"]["max_n_augm"]  # 0のとき拡張したデータをすべて利用
    bayopt_n_epochs = config["train"]["bayopt_n_epochs"]
    bayopt_n_trials = config["train"]["bayopt_n_trials"]
    batch_size = config["train"]["batch_size"]
    n_epochs = config["train"]["n_epochs"]
    early_stopping = config["train"]["early_stopping"]
    tf16 = config["train"]["tf16"]
    loss_func = config["train"]["loss_func"]  # MSE or MAE
    seed = config["train"]["seed"]
    scaling = config["train"]["scaling"]
    filepath = config["dataset"]["filepath"]
    output_dir = config["dataset"]["output_dir"]
    smiles_col_name = config["dataset"]["smiles_col_name"]
    prop_col_name = config["dataset"]["prop_col_name"]
    train_ratio, valid_ratio, test_ratio = config["dataset"]["dataset_ratio"]
    computable_atoms = config["computable_atoms"]
    chirality = config["chirality"]
    stereochemistry = config["stereochemistry"]
    gnn_type = config["gnn_type"]

    if tf16:
        # precision = "16-mixed"
        precision = "16"
        logging.info(f"precision is {precision}")
        torch.set_float32_matmul_precision("medium")
    else:
        precision = "32"
        logging.info(f"precision is {precision}")
        torch.set_float32_matmul_precision("high")
    seed_everything(seed)

    if type(prop_col_name) is str:
        prop_col_name = [prop_col_name]

    num_of_outputs = len(prop_col_name)
    dataset = pl.read_csv(filepath)
    dataset = dataset.select(smiles_col_name, *prop_col_name)
    print(dataset.head())

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)
    log_dir = output_dir.joinpath("logs")
    log_dir.mkdir()
    model_dir = output_dir.joinpath("model")
    model_dir.mkdir()
    img_dir = output_dir.joinpath("images")
    img_dir.mkdir()

    logger, logfile = utils.log_setup(log_dir, "training", verbose=True)
    logging.info(f"OS: {platform.system()}")
    seed_everything(seed)

    logging.info("***Sampling and splitting of the dataset.***")

    smiles_train, smiles_valid, smiles_test, y_train, y_valid, y_test, scaler = (
        data.random_split(
            smiles_input=dataset[smiles_col_name].to_numpy(),
            y_input=dataset[*prop_col_name].to_numpy(),
            train_ratio=train_ratio,
            validation_ratio=valid_ratio,
            test_ratio=test_ratio,
            random_state=seed,
            scaling=scaling,
        )
    )
    if scaler is not None:
        with open(os.path.join(model_dir, "scaler.pkl"), mode="wb") as f:
            pickle.dump(scaler, f)
        logging.info("Scailing output.")
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
    mol2graph = Mol2Graph(
        computable_atoms, poly_flag, chirality, stereochemistry
    )
    n_node_features, n_edge_features = mol2graph.get_graph_features()
    smiles_train, enum_card_train, y_train = augm.augment_data(
        smiles_train, y_train, augmentation, max_n_augm
    )
    graphs_train = mol2graph.get_graph_vectors(
        smiles_train, y_train, n_node_features, n_edge_features
    )
    smiles_valid, enum_card_valid, y_valid = augm.augment_data(
        smiles_valid, y_valid, augmentation, max_n_augm
    )
    graphs_valid = mol2graph.get_graph_vectors(
        smiles_valid, y_valid, n_node_features, n_edge_features
    )
    smiles_test, enum_card_test, y_test = augm.augment_data(
        smiles_test, y_test, augmentation, max_n_augm
    )
    graphs_test = mol2graph.get_graph_vectors(
        smiles_test, y_test, n_node_features, n_edge_features
    )
    logging.info(f"\tTraining set   | {len(y_train)}")
    logging.info(f"\tValidation set | {len(y_valid)}")
    logging.info(f"\tTest set       | {len(y_test)}")
    n_features = len(graphs_train[0]["x"][0])
    data_module = data.DataModule(graphs_train, graphs_valid, batch_size)
    # *****************************************
    # ハイパーパラメータの最適化
    # *****************************************
    if bayopt_on:
        optuna.logging.disable_default_handler()
        best_hparams = bayopt_trainer(
            output_dir,
            n_features,
            bayopt_bounds,
            data_module,
            gnn_type,
            num_of_outputs,
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
            "gnn_type": gnn_type,
            "learning_rate": lr_ref,
            "num_of_outputs": num_of_outputs,
        }
    logging.info("Best Params")
    logging.info(f"GNN Type            |{best_hparams['gnn_type']}")
    logging.info(f"Conv hidden layers  |{best_hparams['n_conv_hidden_layer']}")
    logging.info(f"Dense hidden layers |{best_hparams['n_dense_hidden_layer']}")
    logging.info(f"Graph dim           |{best_hparams['graph_dim']}")
    logging.info(f"Dense dim           |{best_hparams['dense_dim']}")
    logging.info(f"Drop rate           |{best_hparams['drop_rate']}")
    logging.info(f"learning rate       |{best_hparams['learning_rate']}")
    logging.info("")
    # 学習したハイパーパラメータの保存
    config["hyper_parameters"] = {
        "model": {
            "n_features": best_hparams["n_features"],
            "n_conv_hidden_layer": best_hparams["n_conv_hidden_layer"],
            "n_dense_hidden_layer": best_hparams["n_dense_hidden_layer"],
            "graph_dim": best_hparams["graph_dim"],
            "dense_dim": best_hparams["dense_dim"],
            "drop_rate": best_hparams["drop_rate"],
            "gnn_type": best_hparams["gnn_type"],
            "num_of_outputs": best_hparams["num_of_outputs"],
        },
        "other": {
            "learning_rate": best_hparams["learning_rate"]
        }
    }
    with open(os.path.join(model_dir, "all_params.yaml"), mode="w") as f:
        yaml.dump(config, f)

    logging.info("***Training of the best model.***")
    trainer(
        output_dir,
        best_hparams,
        data_module,
        n_epochs=n_epochs,
        early_stopping_epochs=early_stopping,
        n_gpus=n_gpus,
        precision=precision,
        loss_func=loss_func,
    )

    metrics_df = pl.read_csv(
        os.path.join(output_dir, "training/version_0/metrics.csv")
    )
    train_loss = metrics_df["train_loss"].drop_nulls().to_numpy()
    valid_loss = metrics_df["valid_loss"].drop_nulls().to_numpy()
    train_r2 = metrics_df["train_r2"].drop_nulls().to_numpy()
    valid_r2 = metrics_df["valid_r2"].drop_nulls().to_numpy()

    plot.plot_history_loss(
        train_loss, train_r2, valid_loss, valid_r2, img_dir, loss_func
    )

    logging.info(f"Best val_loss @ Epoch #{np.argmin(valid_loss)}")
    logging.info("***Predictions from the best model.***")

    best_model = MolecularGNN(
        **config["hyper_parameters"]["model"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.load_state_dict(
        torch.load(
            model_dir.joinpath("best_weights.pth"),
            map_location=device,
        )
    )
    evaluate.model_evaluation(
        model=best_model,
        img_dir=img_dir,
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
    output_dir: str,
    n_features: int,
    bayopt_bounds: dict,
    data_module: data.DataModule,
    gnn_type="GAT",
    num_of_outputs=1,
    bayopt_n_epochs=10,
    bayopt_n_trials=10,
    precision=32,
    loss_func="MAE",
    n_gpus=1,
    seed=42,
):
    # Optunaの学習用関数を内部に作成
    def _objective(trial: optuna.trial.Trial):
        opt_model = make_opt_model(
            bayopt_bounds,
            n_features,
            gnn_type,
            loss_func,
            trial,
            num_of_outputs
        )
        logger = CSVLogger(output_dir, name="bayes_opt")
        trainer = Trainer(
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
    study.optimize(_objective, n_trials=bayopt_n_trials, n_jobs=1)
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
        "gnn_type": gnn_type,
        "learning_rate": trial.params["learning_rate"],
        "num_of_outputs": num_of_outputs,
    }
    return best_hparams


def make_opt_model(
    bayopt_bounds: dict,
    n_features: int,
    gnn_type: str,
    loss_func: str,
    trial: optuna.trial.Trial,
    num_of_outputs=1,
):
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
    lr = trial.suggest_float(
        "learning_rate",
        float(bayopt_bounds["learning_rate"][0]),
        float(bayopt_bounds["learning_rate"][1]),
        log=True,
    )
    opt_model = LightningModel(
        n_features,
        n_conv_hidden_layer,
        n_dense_hidden_layer,
        2**n_graph_dim,
        2**n_dense_dim,
        drop_rate,
        gnn_type,
        lr,
        loss_func,
        num_of_outputs
    )
    return opt_model


def trainer(
    output_dir: Path,
    best_hparams: dict,
    data_module: data.DataModule,
    n_epochs=100,
    early_stopping_epochs=0,
    n_gpus=1,
    precision=32,
    loss_func="MAE",
):
    training_dir = output_dir.joinpath("training")
    logger = CSVLogger(str(output_dir), name="training")

    training_model = LightningModel(
        n_features=best_hparams["n_features"],
        n_conv_hidden_layer=best_hparams["n_conv_hidden_layer"],
        n_dense_hidden_layer=best_hparams["n_dense_hidden_layer"],
        graph_dim=best_hparams["graph_dim"],
        dense_dim=best_hparams["dense_dim"],
        drop_rate=best_hparams["drop_rate"],
        gnn_type=best_hparams["gnn_type"],
        learning_rate=best_hparams["learning_rate"],
        loss_func=loss_func,
        num_of_outputs=best_hparams["num_of_outputs"]
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        filename="training/version_0/best_weights",
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    early_stopping = EarlyStopping(
        monitor="valid_loss",
        mode="min",
        patience=early_stopping_epochs,
    )
    trainer = Trainer(
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
    trainer.fit(training_model, datamodule=data_module)
    trained_model = LightningModel.load_from_checkpoint(
        output_dir.joinpath("training/version_0/best_weights.ckpt"),
        hparams_file=output_dir.joinpath("training/version_0/hparams.yaml"),
    )
    torch.save(
        obj=trained_model.model.state_dict(),
        f=output_dir.joinpath("model/best_weights.pth")
    )
    logging.info("Training Finished!!!")
