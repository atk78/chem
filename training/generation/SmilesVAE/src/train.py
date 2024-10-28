import os
import shutil
import warnings
import platform
import logging

import yaml
from tqdm import tqdm
import polars as pl
import torch
from torch import nn
import numpy as np
from lightning import LightningModule, seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ProgressBar
from lightning.pytorch.loggers import CSVLogger
from rdkit import RDLogger
from sklearn.model_selection import train_test_split

from . import dataset, token, utils, plot
from .model import SmilesVAE


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
    def __init__(
        self,
        vocab,
        latent_dim,
        embedding_dim=128,
        max_len=100,
        encoder_params={
            "hidden_size": 128,
            "num_layers": 1,
            "dropout": 0.0,
            "bidirectional": False
        },
        decoder_params={
            "hidden_size": 128,
            "num_layers": 1,
            "dropout": 0.0,
            "bidirectional": False
        },
        encoder_out_dim_list=[128, 128],
        learning_rate=1e-3,
        # beta_schedule=None,
        device="cpu",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.vocab = vocab
        # self.beta_schedule = beta_schedule
        self.save_hyperparameters(logger=True)
        self.loss_func = nn.CrossEntropyLoss(reduction="none")
        # self.beta = 1.0
        self.lr = learning_rate
        self.model = SmilesVAE(
            vocab,
            latent_dim,
            embedding_dim,
            max_len,
            encoder_params,
            decoder_params,
            encoder_out_dim_list,
            device=device
        )

    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters(), lr=self.lr)
        return torch.optim.RAdam(self.parameters(), lr=self.lr)

    def loss(self, out_seq_logit, mu, logvar, out_seq):
        """KL情報量を最小化にするための損失関数

        Parameters
        ----------
        out_seq_logit : decoderにより出力された
            _description_
        mu : _type_
            _description_
        logvar : _type_
            _description_
        out_seq : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # out_seq_logit, mu, logvar = self.forward(in_seq, out_seq)
        neg_likelihood = self.loss_func(
            out_seq_logit.transpose(1, 2), out_seq[:, 1:]
        )
        neg_likelihood = neg_likelihood.sum(axis=1).mean()
        kl_div = -0.5 * (1.0 + logvar - mu**2 - torch.exp(logvar)).sum(axis=1).mean()
        return neg_likelihood + 0.1 * kl_div
        # return neg_likelihood + self.beta * kl_div

    def forward(self, in_seq, out_seq=None):
        return self.model(in_seq, out_seq)

    def on_train_epoch_start(self) -> None:
        # try:
        #     self.beta = self.beta_schedule[self.trainer.current_epoch]
        # except:
        #     self.beta = 1.0
        pass

    def training_step(self, batch, batch_idx):
        in_seq, out_seq = batch
        out_seq_logit, mu, logvar = self.model.forward(in_seq, out_seq)
        train_loss = self.loss(out_seq_logit, mu, logvar, out_seq)
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
        out_seq_logit, mu, logvar = self.model.forward(in_seq, out_seq)
        valid_loss = self.loss(out_seq_logit, mu, logvar, out_seq)
        success = self.model.reconstruct(in_seq=in_seq, verbose=False)
        reconstruct_rate = sum(success) / len(success)
        self.log_dict(
            dictionary={
                "valid_loss": valid_loss,
                "success_rate": reconstruct_rate
            },
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        # self.log("valid_loss", valid_loss, prog_bar=True, on_step=False, on_epoch=True)
        return valid_loss


def main(
    config_filepath
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(config_filepath) as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = float(config["learning_rate"])
    latent_dim = config["latent_dim"]
    embedding_dim = config["embedding_dim"]
    encoder_params = config["encoder_params"]
    decoder_params = config["decoder_params"]
    encoder_out_dim_list = config["encoder_out_dim_list"]
    output_dir = config["output_dir"]
    training_dataset = config["training_dataset"]
    smiles_col_name = config["smiles_col_name"]
    test_size = float(config["test_size"])
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
        train_smiles_tensor, valid_smiles_tensor, batch_size=batch_size
    )

    # vocab_size = len(smiles_vocab.char_list)
    max_len = valid_smiles_tensor.shape[1]

    parameters = {
        "vocab": tokens,
        "latent_dim": latent_dim,
        "embedding_dim": embedding_dim,
        "max_len": max_len,
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
        "encoder_out_dim_list": encoder_out_dim_list
    }
    with open(os.path.join(training_dir, "params.yaml"), mode="w") as f:
        yaml.dump(parameters, f)

    lightning_model = LightningModel(
        learning_rate=lr,
        device=device,
        **parameters
    )
    model_checkpoint = ModelCheckpoint(
        dirpath=training_dir,
        filename="best_weights",
        monitor="valid_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    training_logger = CSVLogger(output_dir, name="training")
    trainer = Trainer(
        max_epochs=epochs,
        logger=training_logger,
        enable_checkpointing=True,
        callbacks=[model_checkpoint, LitProgressBar()],
        default_root_dir=training_dir,
        log_every_n_steps=n_log_step,
    )

    trainer.fit(lightning_model, datamodule=datamodule)
    logging.info("Training Finished!!!")

    best_lightning_model = LightningModel.load_from_checkpoint(
        checkpoint_path=os.path.join(training_dir, "best_weights.ckpt"),
        device=device,
        **parameters
    )
    torch.save(
        obj=best_lightning_model.model.state_dict(),
        f=os.path.join(training_dir, "best_weights.pth")
    )

    model = SmilesVAE(device=device, **parameters)
    model.load_state_dict(
        torch.load(
            os.path.join(training_dir, "best_weights.pth"),
            map_location=torch.device("cpu")
        )
    )
    model = model.to(device)
    generated_smiles_list = model.generate(sample_size=1000)
    logging.info(f"success rate: {utils.valid_ratio(generated_smiles_list)}")

    metrics_df = pl.read_csv(
        os.path.join(training_dir, "version_0/metrics.csv")
    )
    train_loss = (
        metrics_df.select("step", "train_loss_step")
        .drop_nulls()
        .cast(dtypes=pl.Float32)
    )
    valid_loss = (
        metrics_df.select("step", "valid_loss")
        .drop_nulls()
        .cast(dtypes=pl.Float32)
    )
    reconstruction_rate = (
        metrics_df.select("step", "success_rate")
        .drop_nulls()
        .cast(dtypes=pl.Float32)
    )
    plot.plot_minibatch_loss(train_loss, valid_loss, img_dir)
    plot.plot_reconstruction_rate(reconstruction_rate, img_dir)
