from torch import nn
from math import ceil
from emotion_id.wavenet import Conv1dMasked, Conv1dSamePadding, ResidualStack
from util import GlobalNormalization


class MLPEmotionIDModel(nn.Module):
    def __init__(self, input_dim, output_classes, no_layers=2, hidden_size=1024):
        super().__init__()
        assert no_layers > 1
        blocks = [
            GlobalNormalization(input_dim, scale=False),
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(no_layers - 2):
            blocks.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        blocks.append(nn.Linear(hidden_size, output_classes))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class LinearEmotionIDModel(nn.Module):
    def __init__(self, in_c, output_classes):
        super().__init__()
        self.normalize = GlobalNormalization(in_c, scale=False)
        self.output_classes = output_classes
        self.linear_1 = nn.Linear(in_c, output_classes)

    def forward(self, x):
        x = self.normalize(x)
        x = self.linear_1(x)
        return x


class RecurrentEmotionIDModel(nn.Module):
    def __init__(
        self,
        feat_dim,
        num_emotions,
        hidden_size=512,
        num_layers=2,
        bidirectional=False,
        dropout=0,
    ):
        super().__init__()
        num_directions = 2 if bidirectional else 1

        self.normalize = GlobalNormalization(feat_dim, scale=False)
        self.hidden_state = None
        self.gru = nn.GRU(
            feat_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(hidden_size * num_directions, num_emotions)

    def forward(self, x):
        x = self.normalize(x)

        if self.hidden_state is not None:
            self.hidden_state = self.hidden_state.detach()

            if x.shape[0] != self.hidden_state.shape[0]:
                self.hidden_state = None

        x, self.hidden_state = self.gru(x, self.hidden_state)
        return self.linear(x)


class WaveNetEmotionIDModel(nn.Module):
    def __init__(
        self,
        in_c,
        output_classes,
        hidden_size=64,
        dilation_depth=6,
        n_repeat=5,
        kernel_size=2,
        masked=True,
    ):
        super().__init__()
        self.normalize = GlobalNormalization(in_c, scale=False)
        self.output_classes = output_classes
        ConvModule = Conv1dMasked if masked else Conv1dSamePadding

        dilations = [kernel_size ** i for i in range(dilation_depth)] * n_repeat
        self.receptive_field = sum(dilations)
        self.max_padding = ceil((dilations[-1] * (kernel_size - 1)) / 2)

        blocks = [
            ConvModule(in_c, hidden_size, kernel_size=1),
            nn.ReLU(),
            ResidualStack(
                hidden_size,
                hidden_size,
                dilations,
                kernel_size=kernel_size,
                conv_module=ConvModule,
            ),
            ConvModule(hidden_size, output_classes, kernel_size=1),
        ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.normalize(x)
        x = x.permute(0, 2, 1)
        x = self.blocks(x)
        return x.permute(0, 2, 1)
