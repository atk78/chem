import os
import shutil
import warnings
import platform
import logging

from tqdm import tqdm
import yaml
import polars as pl
import numpy as np
import torch
from torch import nn
from lightning import LightningModule
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, ProgressBar
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import train_test_split
from rdkit import RDLogger

from . import dataset, token, utils, plot
from .model import SmilesLSTM


warnings.simplefilter("ignore")
RDLogger.DisableLog("rdApp.*")


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
            loss = self.running_loss / self.total_train_batches
            self.bar.set_postfix(
                loss=f"{loss:.3f}", val_loss=f"{val_loss:.3f}"
            )
            self.bar.close()
            self.bar = None

    def disable(self):
        self.bar = None
        self.enabled = False


class LightningModel(LightningModule):
    def __init__(self, vocab, hidden_size, n_layers, learning_rate=1e-3):
        """SMILESをLSTMで学習するためのモデルの初期化

        Parameters
        ----------
        vocab : int
            入力データの次元（=|X|）
        hidden_size : int
            隠れ状態の次元
        n_layers : int
            層数
        """
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.model = SmilesLSTM(vocab, hidden_size, n_layers)
        self.lr = learning_rate
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

    def configure_optimizers(self):
        return torch.optim.RAdam(self.parameters(), lr=self.lr)

    def forward(self, in_seq):
        return self.model(in_seq)

    def loss(self, in_seq, out_seq):
        return self.loss_func(in_seq, out_seq)

    def training_step(self, batch, batch_idx):
        in_seq, out_seq = batch
        # forwardメソッドの出力は(バッチサイズ)×(系列長)×(語彙サイズ)
        # nn.CrossEntropyLossは(バッチサイズ)×(語彙サイズ)×(系列長)を想定
        # transpose(1, 2)で入れ替え
        in_seq = self.forward(in_seq).transpose(1, 2)
        train_loss = self.loss(in_seq, out_seq).mean()
        self.log(
            name="train_loss",
            value=train_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        in_seq, out_seq = batch
        in_seq = self.forward(in_seq).transpose(1, 2)
        valid_loss = self.loss(in_seq, out_seq).mean()
        self.log(
            "valid_loss",
            valid_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )
        return valid_loss


def main(config_filepath):
    with open(config_filepath) as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)
    epochs = config["epochs"]
    hidden_size = config["hidden_size"]
    n_layers = config["n_layers"]
    batch_size = config["batch_size"]
    lr = float(config["learning_rate"])
    output_dir = config["output_dir"]
    training_dataset = config["training_dataset"]
    smiles_col_name = config["smiles_col_name"]
    test_size = config["test_size"]
    n_log_step = config["n_log_step"]
    seed = config["seed"]
    seed_everything(seed)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    training_dir = os.path.join(output_dir, "training")
    img_dir = os.path.join(output_dir, "images")
    log_dir = os.path.join(output_dir, "log")
    os.makedirs(training_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    smiles = pl.read_csv(training_dataset)
    smiles = list(smiles[smiles_col_name])
    train_smiles, valid_smiles = train_test_split(smiles, test_size=test_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _ = utils.log_setup(log_dir, "training", verbose=True)
    logging.info(f"OS: {platform.system()}")
    logging.info(f"Training Dataset filepath: {training_dataset}")
    tokenizer = token.Token()
    tokenized_train_smiles = tokenizer.get_tokens(train_smiles)
    tokenized_valid_smiles = tokenizer.get_tokens(valid_smiles)
    all_tokenized_smiles = tokenized_train_smiles + tokenized_valid_smiles
    tokens = tokenizer.extract_vocab(all_tokenized_smiles)
    tokens = tokenizer.add_extra_tokens(tokens)
    max_length = np.max([len(i_smiles) for i_smiles in all_tokenized_smiles])
    max_length += 2  # eosとsosの分
    logging.info("Loading training SMILES data.")
    train_smiles_tensor = tokenizer.batch_update(
        tokenized_smiles_list=tokenized_train_smiles,
        max_length=max_length,
        tokens=tokens
    )
    logging.info(f"Data size: {train_smiles_tensor.shape}")
    logging.info("")
    logging.info("Loading validation SMILES data.")
    valid_smiles_tensor = tokenizer.batch_update(
        tokenized_smiles_list=tokenized_valid_smiles,
        max_length=max_length,
        tokens=tokens
    )
    logging.info(f"Data size: {valid_smiles_tensor.shape}")
    logging.info("")
    logging.info("All vocabulary: ")
    logging.info(tokens)
    logging.info("Shows the initial molecular SMILES of the training data.")
    logging.info(tokenizer.seq2smiles(train_smiles_tensor[0], tokens))
    logging.info("Convert to token.")
    test_smiles = tokenizer.seq2smiles(train_smiles_tensor[0], tokens)
    test_tokenized_smiles = tokenizer.get_tokens([test_smiles])[0]
    logging.info(tokenizer.smiles2seq(test_tokenized_smiles, tokens))
    datamodule = dataset.DataModule(
        train_smiles=train_smiles_tensor,
        valid_smiles=valid_smiles_tensor,
        batch_size=batch_size
    )
    parameters = {
        "vocab": tokens,
        "hidden_size": hidden_size,
        "n_layers": n_layers
    }
    with open(os.path.join(training_dir, "params.yaml"), mode="w") as f:
        yaml.dump(parameters, f)
    model = LightningModel(learning_rate=lr, **parameters)
    model_checkpoint = ModelCheckpoint(
        dirpath=training_dir,
        filename="best_weights",
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    logger = CSVLogger(output_dir, name="training")
    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[model_checkpoint, LitProgressBar()],
        default_root_dir=training_dir,
        log_every_n_steps=n_log_step
    )
    trainer.fit(model, datamodule=datamodule)
    logging.info("Training Finished!!!")
    best_lightning_model = LightningModel.load_from_checkpoint(
        checkpoint_path=os.path.join(training_dir, "best_weights.ckpt"),
        **parameters
    )
    torch.save(
        obj=best_lightning_model.model.state_dict(),
        f=os.path.join(training_dir, "best_weights.pth")
    )
    model = SmilesLSTM(**parameters)
    model.load_state_dict(
        torch.load(
            os.path.join(training_dir, "best_weights.pth"),
            map_location="cpu"
        )
    )
    model = model.to(device)
    generated_smiles_list = model.generate(sample_size=10000)
    logging.info(f"success rate: {utils.valid_ratio(generated_smiles_list)}")
    metrics_df = pl.read_csv(
        os.path.join(training_dir, "version_0/metrics.csv")
    )
    train_loss = (
        metrics_df.select("step", "train_loss_step")
        .drop_nulls()
        .cast(pl.Float32)
    )
    valid_loss = (
        metrics_df.select("step", "valid_loss")
        .drop_nulls()
        .cast(pl.Float32)
    )
    plot.plot_minibatch_loss(train_loss, valid_loss, img_dir)
