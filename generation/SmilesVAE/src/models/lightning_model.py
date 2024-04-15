import torch
from torch import nn
import lightning as L

from src.models.SmilesVAE import SmilesVAE


class LightningModel(L.LightningModule):
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
