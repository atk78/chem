import shutil
import os
import platform
import pickle
import warnings
import random
from logging import Logger
from pathlib import Path

import yaml
import polars as pl
import numpy as np
import optuna
import torch

from src.model import MolecularGNN
from src.mol2graph import Mol2Graph
from src.data import GraphData
from src.trainer import HoldOutTrainer, CrossValidationTrainer, EarlyStopping
from src import utils, evaluate


warnings.simplefilter("ignore")


class BayOptLoss:
    loss = None
    r2 = None
    number = 0


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def bayopt_trainer(
    output_dir: Path,
    logger: Logger,
    n_features: int,
    bayopt_bounds: dict,
    graph_data: GraphData,
    gnn_type="GAT",
    num_of_outputs=1,
    bayopt_n_epochs=10,
    bayopt_n_trials=10,
    seed=42,
    device="cpu"
):
    bayopt_dir = output_dir.joinpath("bayes_opt")
    if bayopt_dir.exists():
        shutil.rmtree(bayopt_dir)
    bayopt_dir.mkdir()
    # device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # Optunaの学習用関数を内部に作成
    optuna.logging.enable_propagation()

    def _objective(trial: optuna.trial.Trial):
        lr = trial.suggest_float(
            "learning_rate",
            float(bayopt_bounds["learning_rate"][0]),
            float(bayopt_bounds["learning_rate"][1]),
            log=True,
        )

        opt_model = make_opt_model(
            bayopt_bounds,
            n_features,
            gnn_type,
            trial,
            num_of_outputs
        )
        # criterion = nn.MSELoss()
        # optimizer = optim.AdamW(opt_model.parameters(), lr=lr)
        trial_path = bayopt_dir.joinpath(f"trial_{trial.number}")
        trial_path.mkdir(exist_ok=True)
        if graph_data.validation_method == "holdout":
            trainer = HoldOutTrainer(
                graph_data,
                trial_path,
                learning_rate=lr,
                scheduler=None,
                n_epochs=bayopt_n_epochs,
                early_stopping=None,
                device=device,
            )
        else:
            trainer = CrossValidationTrainer(
                graph_data,
                trial_path,
                learning_rate=lr,
                scheduler=None,
                n_epochs=bayopt_n_epochs,
                early_stopping=None,
                device=device,
            )
        trainer.fit(opt_model)
        if BayOptLoss.loss is None:
            BayOptLoss.loss = trainer.loss
        else:
            if BayOptLoss.loss > trainer.loss:
                BayOptLoss.loss = trainer.loss
                BayOptLoss.number = trial.number
        return trainer.loss

    # ハイパーパラメータの探索の開始
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=seed)
    )
    study.optimize(_objective, n_trials=bayopt_n_trials, n_jobs=1)
    # 探索のうち、一番損失が少なかった条件でのハイパーパラメータを保存
    trial = study.best_trial
    logger.info(
        f"Best Trial: {trial.number} with RMSE value: {trial.value}"
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

    opt_model = MolecularGNN(
        n_features,
        n_conv_hidden_layer,
        n_dense_hidden_layer,
        2**n_graph_dim,
        2**n_dense_dim,
        drop_rate,
        gnn_type,
        num_of_outputs
    )
    return opt_model


def training_model(
    training_model: MolecularGNN,
    output_dir: Path,
    graph_data: GraphData,
    learning_rate: float,
    n_epochs=100,
    early_stopping_patience=0,
    device="cpu",
):
    training_dir = output_dir.joinpath("training")
    training_dir.mkdir()
    if early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            early_stopping_patience,
            output_dir.joinpath("model"),
        )
    else:
        early_stopping = None
    if graph_data.validation_method == "holdout":
        trainer = HoldOutTrainer(
            graph_data,
            training_dir,
            learning_rate,
            scheduler=None,
            n_epochs=n_epochs,
            early_stopping=early_stopping,
            device=device
        )
    else:
        trainer = CrossValidationTrainer(
            graph_data,
            training_dir,
            learning_rate,
            scheduler=None,
            n_epochs=n_epochs,
            early_stopping=early_stopping,
            device=device,
        )
    trainer.fit(training_model)


def run(
    config_filepath: str,
    # n_gpus=1,
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
    bayopt_n_epochs = config["train"]["bayopt_n_epochs"]
    bayopt_n_trials = config["train"]["bayopt_n_trials"]

    validation_method = config["train"]["validation_method"]
    kfold_n_splits = config["train"]["kfold_n_splits"]
    batch_size = config["train"]["batch_size"]
    n_epochs = config["train"]["n_epochs"]
    early_stopping_patience = config["train"]["early_stopping_patience"]
    tf16 = config["train"]["tf16"]
    seed = config["train"]["seed"]
    scaling = config["train"]["scaling"]

    dataset_filepath = config["dataset"]["filepath"]
    output_dir = config["dataset"]["output_dir"]
    smiles_col_name = config["dataset"]["smiles_col_name"]
    prop_col_name = config["dataset"]["prop_col_name"]
    dataset_ratio = config["dataset"]["dataset_ratio"]

    computable_atoms = config["computable_atoms"]
    chirality = config["chirality"]
    stereochemistry = config["stereochemistry"]
    gnn_type = config["gnn_type"]

    if type(prop_col_name) is str:
        prop_col_name = [prop_col_name]

    num_of_outputs = len(prop_col_name)
    dataset = pl.read_csv(dataset_filepath)
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

    logger = utils.log_setup(log_dir, "training", verbose=True)
    logger.info(f"OS: {platform.system()}")

    if tf16:
        precision = "16"
        logger.info(f"precision is {precision}")
        torch.set_float32_matmul_precision("medium")
    else:
        precision = "32"
        logger.info(f"precision is {precision}")
        torch.set_float32_matmul_precision("high")
    seed_everything(seed)

    if dataset[smiles_col_name][0].count("*") == 0:
        poly_flag = False
    else:
        poly_flag = True
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    mol2graph = Mol2Graph(
        computable_atoms, poly_flag, chirality, stereochemistry
    )
    graph_data = GraphData(
        smiles=dataset[smiles_col_name].to_numpy(),
        y=dataset[*prop_col_name].to_numpy(),
        mol2graph=mol2graph,
        validation_method=validation_method,
        kfold_n_splits=kfold_n_splits,
        batch_size=batch_size,
        dataset_ratio=dataset_ratio,
        random_state=seed,
        scaling=scaling,
    )
    # graph_data.get_graphs_datasets()
    if graph_data.scaler is not None:
        with open(str(model_dir.joinpath("scaler.pkl")), mode="wb") as f:
            pickle.dump(graph_data.scaler, f)
        logger.info("Scaling output.")
    logger.info(f"Dava validation method: {validation_method}.")
    if validation_method == "kfold":
        logger.info(f"Num of fold: {kfold_n_splits}")
    n_features = len(graph_data.graph_datasets["train"][0]["x"][0])
    # *****************************************
    # ハイパーパラメータの最適化
    # *****************************************
    if bayopt_on:
        # optuna.logger.disable_default_handler()
        best_hparams = bayopt_trainer(
            output_dir,
            logger,
            n_features,
            bayopt_bounds,
            graph_data,
            gnn_type,
            num_of_outputs,
            bayopt_n_epochs=bayopt_n_epochs,
            bayopt_n_trials=bayopt_n_trials,
            seed=seed,
            device=device
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
    logger.info("Best Params")
    logger.info(f"GNN Type            |{best_hparams['gnn_type']}")
    logger.info(f"Conv hidden layers  |{best_hparams['n_conv_hidden_layer']}")
    logger.info(f"Dense hidden layers |{best_hparams['n_dense_hidden_layer']}")
    logger.info(f"Graph dim           |{best_hparams['graph_dim']}")
    logger.info(f"Dense dim           |{best_hparams['dense_dim']}")
    logger.info(f"Drop rate           |{best_hparams['drop_rate']}")
    logger.info(f"learning rate       |{best_hparams['learning_rate']}")
    logger.info("")
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
    with open(model_dir.joinpath("all_params.yaml"), mode="w") as f:
        yaml.dump(config, f)
    model = MolecularGNN(
        n_features=best_hparams["n_features"],
        n_conv_hidden_layer=best_hparams["n_conv_hidden_layer"],
        n_dense_hidden_layer=best_hparams["n_dense_hidden_layer"],
        graph_dim=best_hparams["graph_dim"],
        dense_dim=best_hparams["dense_dim"],
        gnn_type=best_hparams["gnn_type"],
        num_of_outputs=best_hparams["num_of_outputs"],
    )
    lr = best_hparams["learning_rate"]
    logger.info("***Training of the best model.***")
    training_model(
        model,
        output_dir,
        graph_data,
        lr,
        n_epochs,
        early_stopping_patience,
        device
    )
    logger.info("Training Finishued!!!")
    evaluate.model_evaluation(
        model,
        logger,
        output_dir,
        graph_data,
        device,
    )
