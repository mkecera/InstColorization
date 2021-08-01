import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):

        super(ConvLSTMCell, self).__init__()

        self.input_size = input_dim
        self.hidden_size = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.convolutional_layer = nn.Conv2d(in_channels=self.input_size + self.hidden_size,
                                             out_channels=4 * self.hidden_size,
                                             kernel_size=self.kernel_size,
                                             padding=self.padding,
                                             bias=self.bias)

    def forward(self, input_tensor, cur_state):
        hidden_current, cell_cur = cur_state
        combined_hidden_input = torch.cat([input_tensor, hidden_current], dim=1)

        combined_convolution = self.convolutional_layer(combined_hidden_input)
        cc_input, cc_forget, cc_output, cc_hidden_g = torch.split(combined_convolution, self.hidden_size, dim=1)
        input = torch.sigmoid(cc_input)
        forget = torch.sigmoid(cc_forget)
        output = torch.sigmoid(cc_output)
        hidden_g = torch.tanh(cc_hidden_g)

        cell_next = forget * cell_cur + input * hidden_g
        hidden_next = output * torch.tanh(cell_next)

        return hidden_next, cell_next

    def init_hidden(self, batch_size, image_size):
        return (torch.zeros(batch_size, self.hidden_size, image_size[0], image_size[1], device=self.convolutional_layer.weight.device),
                torch.zeros(batch_size, self.hidden_size, image_size[0], image_size[1], device=self.convolutional_layer.weight.device))
