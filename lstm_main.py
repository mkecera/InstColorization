import sys
from video_module.data_loader import DataSetLoader
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import time
torch.cuda.empty_cache()

sys.path.append("home/gomni/Documents/InstColorization/")

from video_module.models.lstm_model.encoder_decoder import ArchitectureConvLSTM
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store_true',)
opt = parser.parse_args()

args, leftovers = parser.parse_known_args()


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

    def save_picture(self, image):
        file_name = './video_module/models/lstm_model/results/' + str(time.strftime("%Y%m%d-%H%M%S")) + '.png'
        cv2.imwrite(file_name, image)

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
        self.save_picture(y_hat)
        return {'test_loss': self.criterion(y_hat, y)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(self.decay_rate_1, self.decay_rate_2))

    def train_dataloader(self):
        a = DataSetLoader()
        train_loader = torch.utils.data.DataLoader(dataset=a, batch_size=self.batch_size, shuffle=False)
        return train_loader

    def test_dataloader(self):
        a = DataSetLoader()
        test_loader = torch.utils.data.DataLoader(dataset=a, batch_size=self.batch_size, shuffle=False)
        return test_loader


def run_trainer(n_steps_ahead=1, epochs=200, in_channel=3):
    conv_lstm_model = ArchitectureConvLSTM(nf=n_steps_ahead, in_chan=in_channel)
    model = ConvLSTMMain(model=conv_lstm_model)
    trainer = Trainer(max_epochs=epochs)
    trainer.fit(model)


def run_tester(n_steps_ahead=1, epochs=200, in_channel=3):
    conv_lstm_model = ArchitectureConvLSTM(nf=n_steps_ahead, in_chan=in_channel)
    model = ConvLSTMMain(model=conv_lstm_model)
    trainer = Trainer()
    trainer.test(model)


if __name__ == '__main__':
    if args.train is not None:
        run_trainer()
    else:
        run_tester()
