import sys
from video_module.data_loader import DataSetLoader

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer

torch.cuda.empty_cache()

sys.path.append("home/gomni/Documents/InstColorization/")

from video_module.models.lstm_model.encoder_decoder import ArchitectureConvLSTM


class ConvLSTMMain(pl.LightningModule):
    def __init__(self, hparams=None, model=None):
        super(ConvLSTMMain, self).__init__()

        self.path = './video_module/data/training/'

        self.normalize = False
        self.model = model

        self.log_images = False

        self.criterion = torch.nn.MSELoss()
        self.batch_size = 2
        self.n_steps_past = 10

        self.learning_rate = 1e-4
        self.decay_rate_1 = 0.9
        self.decay_rate_2 = 0.98

        self.n_hidden_dim = 64

    def forward(self, x):
        output = self.model(x, future_seq=self.n_steps_ahead)
        return output

    def training_step(self, batch, batch_idx):

        x, y = batch[:, 0, 0:self.n_steps_past, :, :, :], batch[:, 1, self.n_steps_past, :, :, :].unsqueeze(1)
        x = x.permute(0, 1, 4, 2, 3)
        y = y.permute(0, 1, 4, 2, 3)
        y = y.squeeze()

        y_hat = self.forward(x).squeeze()
        loss = self.criterion(y_hat, y)

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': self.criterion(y_hat, y)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(self.decay_rate_1, self.decay_rate_2))

    def train_dataloader(self):
        a = DataSetLoader()
        train_loader = torch.utils.data.DataLoader(dataset=a, batch_size=self.batch_size, shuffle=False)
        return train_loader

    def test_dataloader(self):
        a = DataSetLoader(data_root='./video_module/data/testing')
        train_loader = torch.utils.data.DataLoader(dataset=a, batch_size=self.batch_size, shuffle=False)
        return train_loader


def run_trainer(n_steps_ahead=1, epochs=200, in_channel=3):
    conv_lstm_model = ArchitectureConvLSTM(nf=n_steps_ahead, in_chan=in_channel)
    model = ConvLSTMMain(model=conv_lstm_model)
    trainer = Trainer(max_epochs=epochs)
    trainer.fit(model)


def run_trainer(n_steps_ahead=1, epochs=200, in_channel=3):
    conv_lstm_model = ArchitectureConvLSTM(nf=n_steps_ahead, in_chan=in_channel)
    model = ConvLSTMMain(model=conv_lstm_model)
    trainer = Trainer(max_epochs=epochs)
    output = trainer.evaluate(model)


if __name__ == '__main__':
    run_trainer()
    # run_tester()
