import shutil
import os

import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger
import optuna


from SMILESX.model import LSTMAttention
from SMILESX import datas


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
    precision="16-mixed",
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

        train_dataloader = datas.make_dataloader(
            x_train, y_train, batch_size=2**n_batch_size, train=True
        )
        valid_dataloader = datas.make_dataloader(
            x_valid,
            y_valid,
            batch_size=(
                2 ** n_batch_size if len(y_valid) > 2 ** n_batch_size else len(y_valid)
            ),
        )
        if os.path.exists(os.path.join(save_dir, "bays_opt")):
            shutil.rmtree(os.path.join(save_dir, "bays_opt"))
        logger = CSVLogger(save_dir, name="bays_opt")
        opt_model = LSTMAttention(
            token_size=max_length + 1,
            learning_rate=lr,
            lstm_units=2**n_lstm_units,
            dense_units=2**n_dense_units,
            embedding_dim=2**n_embedding_dim,
            log_flag=False
        )
        trainer = pl.Trainer(
            max_epochs=bayopt_n_epochs,
            precision=precision,
            logger=logger,
            enable_checkpointing=False,
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
        max_length + 1,
        2 ** trial.params["n_lstm_units"],
        2 ** trial.params["n_dense_units"],
        2 ** trial.params["n_embedding_dim"],
        2 ** trial.params["n_batch_size"],
        trial.params["learning_rate"],
    ]

    return best_hyper_param
