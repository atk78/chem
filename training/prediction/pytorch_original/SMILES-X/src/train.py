import shutil
import os
import platform
import argparse
import warnings
import random
from logging import Logger
from pathlib import Path

import yaml
import polars as pl
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.model import SmilesX
from src.trainer import HoldOutTrainer, CrossValidationTrainer, EarlyStopping
from src import token, utils, data, evaluate

warnings.simplefilter("ignore")


class BayOptLoss:
    loss = None
    r2 = None
    number = 0


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def bayopt_training(
    output_dir: Path,
    logger: Logger,
    vocab_size: int,
    bayopt_bounds: dict,
    smilesX_data: data.SmilesXData,
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
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
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
            vocab_size,
            bayopt_bounds,
            trial,
            num_of_outputs
        )
        criterion = F.mse_loss
        optimizer = optim.AdamW(opt_model.parameters(), lr=lr)
        trial_path = bayopt_dir.joinpath(f"trial_{trial.number}")
        trial_path.mkdir(exist_ok=True)
        if smilesX_data.validation_method == "holdout":
            trainer = HoldOutTrainer(
                smilesX_data,
                trial_path,
                criterion,
                optimizer,
                scheduler=None,
                n_epochs=bayopt_n_epochs,
                early_stopping=None,
                device=device,
            )
        else:
            trainer = CrossValidationTrainer(
                smilesX_data,
                trial_path,
                criterion,
                optimizer,
                scheduler=None,
                n_epochs=bayopt_n_epochs,
                early_stopping=None,
                device=device,
            )
        _opted_model = trainer.fit(opt_model)
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
        "vocab_size": int(vocab_size),
        "lstm_dim": 2 ** trial.params["n_lstm_dim"],
        "dense_dim": 2 ** trial.params["n_dense_dim"],
        "embedding_dim": 2 ** trial.params["n_embedding_dim"],
        "learning_rate": trial.params["learning_rate"],
        "num_of_outputs": num_of_outputs,
    }

    return best_hparams


def make_opt_model(
    vacab_size: int,
    bayopt_bounds: dict,
    trial: optuna.trial.Trial,
    num_of_outputs=1,
):
    n_lstm_dim = trial.suggest_int(
        "n_lstm_dim",
        bayopt_bounds["lstm_dim"][0],
        bayopt_bounds["lstm_dim"][1],
    )
    n_dense_dim = trial.suggest_int(
        "n_dense_dim",
        bayopt_bounds["dense_dim"][0],
        bayopt_bounds["dense_dim"][1],
    )
    n_embedding_dim = trial.suggest_int(
        "n_embedding_dim",
        bayopt_bounds["embedding_dim"][0],
        bayopt_bounds["embedding_dim"][1],
    )
    opt_model = SmilesX(
        vocab_size=vacab_size,
        lstm_dim=n_lstm_dim,
        dense_dim=n_dense_dim,
        embedding_dim=n_embedding_dim,
        return_proba=False,
        num_of_outputs=num_of_outputs
    )
    return opt_model


def training_model(
    output_dir: Path,
    best_hparams: dict,
    smilesX_data: data.SmilesXData,
    n_epochs=100,
    early_stopping_patience=0,
    device="cpu",
):
    training_dir = output_dir.joinpath("training")
    training_dir.mkdir()
    training_model = SmilesX(
        vocab_size=best_hparams["vocab_size"],
        lstm_dim=best_hparams["lstm_dim"],
        dense_dim=best_hparams["dense_dim"],
        embedding_dim=best_hparams["embedding_dim"],
        num_of_outputs=best_hparams["num_of_outputs"]
    )
    lr = best_hparams["learning_rate"]
    criterion = F.mse_loss
    optimizer = optim.AdamW(training_model.parameters(), lr=lr)
    if early_stopping_patience > 0:
        early_stopping = EarlyStopping(
            early_stopping_patience,
            training_dir.joinpath("model.pth"),
        )
    else:
        early_stopping = None
    if smilesX_data.validation_method == "holdout":
        trainer = HoldOutTrainer(
            smilesX_data,
            training_dir,
            criterion,
            optimizer,
            scheduler=None,
            n_epochs=n_epochs,
            early_stopping=early_stopping,
            device=device
        )
    else:
        trainer = CrossValidationTrainer(
            smilesX_data,
            training_dir,
            criterion,
            optimizer,
            scheduler=None,
            n_epochs=n_epochs,
            early_stopping=early_stopping,
            device=device,
        )
    best_model = trainer.fit(training_model)
    best_model_path = output_dir.joinpath("model/best_model.pth")
    torch.save(best_model.state_dict(), str(best_model_path))
    return best_model


def run(config_filepath=""):
    with open(config_filepath) as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    bayopt_on = config["train"]["bayopt_on"]
    bayopt_n_epochs = config["train"]["bayopt_n_epochs"]
    bayopt_n_trials = config["train"]["bayopt_n_trials"]
    if bayopt_on:
        bayopt_bounds = config["bayopt_bounds"]
    else:
        lstm_dim_ref = config["ref_hyperparam"]["lstm_dim"]
        dense_dim_ref = config["ref_hyperparam"]["dense_dim"]
        embedding_dim_ref = config["ref_hyperparam"]["embedding_dim"]
        learning_rate_ref = config["ref_hyperparam"]["learning_rate"]

    augmentation = config["train"]["augmentation"]
    validation_method = config["train"]["validation_method"]
    kfold_n_splits = config["train"]["kfold_n_splits"]
    batch_size = config["train"]["batch_size"]
    n_epochs = config["train"]["n_epochs"]
    early_stopping_patience = config["train"]["early_stopping_patience"]
    tf16 = config["train"]["tf16"]
    seed = config["train"]["seed"]
    scaling = config["train"]["scaling"]

    dataset_filepath = config["dataset"]["filepath"]
    smiles_col_name = config["dataset"]["smiles_col_name"]
    prop_col_name = config["dataset"]["prop_col_name"]
    output_dir = config["dataset"]["output_path"]
    dataset_ratio = config["dataset"]["dataset_ratio"]

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
    img_dir = output_dir.joinpath("images")
    img_dir.mkdir()

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

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    logger.info("***Sampling and splitting of the dataset.***")
    smilesX_data = data.SmilesXData(
        smiles=dataset[smiles_col_name].to_numpy(),
        y=dataset[*prop_col_name].to_numpy(),
        augmentation=augmentation,
        validation_method=validation_method,
        kfold_n_splits=kfold_n_splits,
        batch_size=batch_size,
        dataset_ratio=dataset_ratio,
        random_state=seed,
        scaling=scaling
    )

    if smilesX_data.original_datasets["train"][0][0].count("*") == 0:
        logger.info("Setup Molecule Tokens.")
    else:
        logger.info("Setup Polymer Tokens.")

    if augmentation:
        logger.info("***Data augmentation is True.***")
        logger.info("Augmented SMILES data size:")
    else:
        logger.info("***No data augmentation has been required.***")
        logger.info("SMILES data size:")
    all_smlies_list = [
        data[0] for data in smilesX_data.original_datasets.values()
    ]
    # all_tokenized_smiles = token.get_tokens(
    #     list(dataset.select(smiles_col_name).to_numpy().flatten())
    # )
    all_smlies_list = sum(all_smlies_list, [])
    all_tokenized_smiles = token.get_tokens(all_smlies_list)
    tokens = token.extract_vocab(all_tokenized_smiles)
    vocab_size = len(tokens)
    logger.info("Tokens:")
    logger.info(tokens)

    for phase, datas in smilesX_data.original_datasets.items():
        logger.info(f"{phase} set\t| {len(datas[0])}")

    # token.check_unique_tokens(
    #     tokenized_smiles_train, tokenized_smiles_valid, tokenized_smiles_test
    # )
    logger.info(f"Full vocabulary: {tokens}")
    logger.info(f"Of size: {vocab_size}")
    tokens, vocab_size = token.add_extract_tokens(tokens, vocab_size)
    max_length = max([len(i_smiles) for i_smiles in all_tokenized_smiles])
    max_length += 2  # ["pad"]の分
    logger.info(
        f"Maximum length of tokenized SMILES: {max_length} tokens"
        "(termination spaces included)"
    )
    smilesX_data.tensorize(max_length, tokens)
    if bayopt_on:
        # optuna.logger.disable_default_handler()
        best_hparams = bayopt_training(
            output_dir,
            logger,
            vocab_size,
            bayopt_bounds,
            smilesX_data,
            num_of_outputs,
            bayopt_n_epochs,
            bayopt_n_trials,
            seed,
            device
        )
    else:
        best_hparams = {
            "vocab_size": vocab_size,
            "lstm_dim": lstm_dim_ref,
            "dense_dim": dense_dim_ref,
            "embedding_dim": embedding_dim_ref,
            "learning_rate": learning_rate_ref,
        }
    logger.info("Best Params")
    logger.info(f"LSTM dim       |{best_hparams['lstm_dim']}")
    logger.info(f"Dense dim      |{best_hparams['dense_dim']}")
    logger.info(f"Embedding dim  |{best_hparams['embedding_dim']}")
    logger.info(f"learning rate    |{best_hparams['learning_rate']}")
    logger.info("")

    config["hyper_parameters"] = {
        "model": {
            "vocab_size": best_hparams["vocab_size"],
            "lstm_dim": best_hparams["lstm_dim"],
            "dense_dim": best_hparams["dense_dim"],
            "embedding_dim": best_hparams["embedding_dim"],
            "num_of_outputs": best_hparams["num_of_outputs"],
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
    with open(model_dir.joinpath("all_params.yaml"), mode="w") as f:
        yaml.dump(config, f)

    logger.info("***Training of the best model.***")
    best_model = training_model(
        output_dir,
        best_hparams,
        smilesX_data,
        n_epochs,
        early_stopping_patience,
        device
    )
    logger.info("Training Finished !!!")
    evaluate.model_evaluation(
        best_model,
        logger,
        img_dir,
        smilesX_data,
        device,
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
