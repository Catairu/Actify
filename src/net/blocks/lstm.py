import torch
import torch.nn as nn


class LSTMBlock(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
        use_layer_norm=False,
    ):
        super(LSTMBlock, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_layer_norm = use_layer_norm

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        output_dim = hidden_size * 2 if bidirectional else hidden_size

        if self.use_layer_norm:
            self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)

        if self.bidirectional:
            final_hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        else:
            final_hn = hn[-1, :, :]

        if self.use_layer_norm:
            final_hn = self.ln(final_hn)

        return final_hn
