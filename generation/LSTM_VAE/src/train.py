import os
import shutil
import warnings
import pickle
from tqdm import tqdm

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from rdkit import RDLogger

from . import dataset, token, utils, plot
from .generate import generate
from .models.SmilesLSTM import LightningModel


warnings.simplefilter("ignore")
RDLogger.DisableLog("rdApp.*")


class LitProgressBar(pl.callbacks.ProgressBar):
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
            loss = self.running_loss / self.total_train_batches
            self.bar.set_postfix(loss=f"{loss:.3f}", val_loss=f"{val_loss:.3f}")
            self.bar.close()
            self.bar = None

    def disable(self):
        self.bar = None
        self.enabled = False


def main(train_smiles, valid_smlies, epochs=2, output_dir="", lr=1e-3, batch_size=128):
    training_dir = os.path.join(output_dir, "training")
    if os.path.exists(training_dir):
        shutil.rmtree(training_dir)
        os.makedirs(training_dir)
    else:
        os.makedirs(training_dir)

    smiles_vocab = token.SmilesVocabulary()
    print("Loading training SMILES data.")
    train_smiles_tensor = smiles_vocab.batch_update(train_smiles)
    print(f"Data size: {train_smiles_tensor.shape}\n")
    print("Loading validation SMILES data.")
    valid_smiles_tensor = smiles_vocab.batch_update(valid_smlies)
    print(f"Data size: {valid_smiles_tensor.shape}\n")
    print("Shows the initial molecular SMILES of the training data.")
    print(smiles_vocab.seq2smiles(train_smiles_tensor[0]))
    print("Convert to token.")
    print(smiles_vocab.smiles2seq(smiles_vocab.seq2smiles(train_smiles_tensor[0])))

    datamodule = dataset.DataModule(train_smiles_tensor, valid_smiles_tensor, batch_size=batch_size)
    vocab_size = len(smiles_vocab.char_list)
    model = LightningModel(
        vocab_size=vocab_size, hidden_size=512, n_layers=3, learning_rate=lr
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=training_dir,
        filename="best_weights",
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    logger = CSVLogger(output_dir, name="training")
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[model_checkpoint, LitProgressBar()],
        default_root_dir=training_dir,
    )
    trainer.fit(model, datamodule=datamodule)
    print("Training Finished!!!")

    # generated_smiles_list = model.generate(smiles_vocab, sample_size=10000)
    generated_smiles_list = generate(model, smiles_vocab, sample_size=1000)
    with open(os.path.join(output_dir, "smiles_vocab.pkl"), mode="wb") as f:
        pickle.dump(smiles_vocab, f)
    print(f"success rate: {utils.valid_ratio(generated_smiles_list)}")
    img_dir = os.path.join(output_dir, "images")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    metrics_df = pd.read_csv(os.path.join(training_dir, "version_0/metrics.csv"))
    train_loss = metrics_df[["step", "train_loss_step"]].dropna()
    valid_loss = metrics_df[["step", "valid_loss"]].dropna()
    plot.plot_minibatch_loss(train_loss, valid_loss, img_dir)
