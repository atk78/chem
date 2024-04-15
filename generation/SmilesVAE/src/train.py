import os
import shutil
import warnings
import random
import platform
import logging

import yaml
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ProgressBar
from lightning.pytorch.loggers import CSVLogger
from rdkit import RDLogger

from . import dataset, token, utils, plot
from .models.SmilesVAE import SmilesVAE
from .models.lightning_model import LightningModel


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


def run(
    train_smiles,
    valid_smlies,
    config_filepath
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
    n_log_step = config["n_log_step"]
    seed = config["seed"]
    seed_everything(seed)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    training_dir = os.path.join(output_dir, "training")
    img_dir = os.path.join(output_dir, "images")
    log_dir = os.path.join(output_dir, "log")
    os.mkdir(training_dir)
    os.mkdir(img_dir)
    os.mkdir(log_dir)

    _, _ = utils.log_setup(log_dir, "training", verbose=True)
    logging.info(f"OS: {platform.system()}")

    logging.info("Loading training SMILES data.")
    train_smiles_tensor = token.batch_update(train_smiles)
    logging.info(f"Data size: {train_smiles_tensor.shape}\n")
    logging.info("Loading validation SMILES data.")
    valid_smiles_tensor = token.batch_update(valid_smlies)
    vocab = token.Token.char_list
    logging.info(f"Data size: {valid_smiles_tensor.shape}\n")
    logging.info("Shows the initial molecular SMILES of the training data.")
    logging.info(token.seq2smiles(train_smiles_tensor[0], vocab))
    logging.info("Convert to token.")
    logging.info(token.smiles2seq(token.seq2smiles(train_smiles_tensor[0], vocab), vocab))
    logging.info(token.Token.char_list)

    datamodule = dataset.DataModule(
        train_smiles_tensor, valid_smiles_tensor, batch_size=batch_size
    )

    # vocab_size = len(smiles_vocab.char_list)
    max_len = valid_smiles_tensor.shape[1]

    model = LightningModel(
        vocab=vocab,
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
        max_len=max_len,
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        encoder_out_dim_list=encoder_out_dim_list,
        learning_rate=lr,
        device=device
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
    trainer = L.Trainer(
        max_epochs=epochs,
        logger=logger,
        enable_checkpointing=True,
        callbacks=[model_checkpoint, LitProgressBar()],
        default_root_dir=training_dir,
        log_every_n_steps=n_log_step,
    )

    trainer.fit(model, datamodule=datamodule)
    logging.info("Training Finished!!!")

    model = LightningModel.load_from_checkpoint(
        checkpoint_path=os.path.join(training_dir, "best_weights.ckpt"),
        vocab=vocab,
        latent_dim=latent_dim,
        embedding_dim=embedding_dim,
        max_len=max_len,
        encoder_params=encoder_params,
        decoder_params=decoder_params,
        encoder_out_dim_list=encoder_out_dim_list,
        learning_rate=lr,
        device=device
    )
    torch.save(
        obj=model.model.state_dict(),
        f=os.path.join(training_dir, "best_weights.pth")
    )

    parameters = {
        "vocab": vocab,
        "latent_dim": latent_dim,
        "embedding_dim": embedding_dim,
        "max_len": max_len,
        "encoder_params": encoder_params,
        "decoder_params": decoder_params,
        "encoder_out_dim_list": encoder_out_dim_list
    }
    with open(os.path.join(training_dir, "params.yaml"), mode="w") as f:
        yaml.dump(parameters, f)

    model = SmilesVAE(
        device=device,
        **parameters
    )
    model.load_state_dict(
        torch.load(
            os.path.join(training_dir, "best_weights.pth"),
            map_location=torch.device("cpu")
        )
    )
    model = model.to(device)
    generated_smiles_list = model.generate(sample_size=1000)
    logging.info(f"success rate: {utils.valid_ratio(generated_smiles_list)}")

    metrics_df = pd.read_csv(
        os.path.join(training_dir, "version_0/metrics.csv")
    )
    train_loss = metrics_df[["step", "train_loss_step"]].dropna()
    valid_loss = metrics_df[["step", "valid_loss"]].dropna()
    reconstruction_rate = metrics_df[["step", "success_rate"]].dropna()
    plot.plot_minibatch_loss(train_loss, valid_loss, img_dir)
    plot.plot_reconstruction_rate(reconstruction_rate, img_dir)


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
