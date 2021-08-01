import torch.nn as nn
import torch
from video_module.models.lstm_model.conv_lstm import ConvLSTMCell


class ArchitectureConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(ArchitectureConvLSTM, self).__init__()

        self.encoder_1 = ConvLSTMCell(input_dim=in_chan,
                                      hidden_dim=nf,
                                      kernel_size=(3, 3),
                                      bias=True)

        self.encoder_2 = ConvLSTMCell(input_dim=nf,
                                      hidden_dim=nf,
                                      kernel_size=(3, 3),
                                      bias=True)

        self.decoder_1 = ConvLSTMCell(input_dim=nf,  # nf + 1
                                      hidden_dim=nf,
                                      kernel_size=(3, 3),
                                      bias=True)

        self.decoder_2 = ConvLSTMCell(input_dim=nf,
                                      hidden_dim=nf,
                                      kernel_size=(3, 3),
                                      bias=True)

        self.decoder_CNN_output = nn.Conv3d(in_channels=nf,
                                            out_channels=3,
                                            kernel_size=(1, 3, 3),
                                            padding=(0, 1, 1))

    def autoencoder(self, x, seq_len, future_step, hidden_t, cell_t, hidden_t2, cell_t2, hidden_t3, cell_t3, hidden_t4, cell_t4):

        outputs = []

        for t in range(seq_len):
            hidden_t, cell_t = self.encoder_1(input_tensor=x[:, t, :, :],
                                              cur_state=[hidden_t, cell_t])
            hidden_t2, cell_t2 = self.encoder_2(input_tensor=hidden_t,
                                                cur_state=[hidden_t2, cell_t2])

        encoded_vector = hidden_t2

        for t in range(future_step):
            hidden_t3, cell_t3 = self.decoder_1(input_tensor=encoded_vector,
                                                cur_state=[hidden_t3, cell_t3])
            hidden_t4, cell_t4 = self.decoder_2(input_tensor=hidden_t3,
                                                cur_state=[hidden_t4, cell_t4])
            encoded_vector = hidden_t4
            outputs += [hidden_t4]

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN_output(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x, future_seq=0, hidden_state=None):

        b, seq_len, _, h, w = x.size()
        h_t, c_t = self.encoder_1.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2.init_hidden(batch_size=b, image_size=(h, w))

        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4)

        return outputs