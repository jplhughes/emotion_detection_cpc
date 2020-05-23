from torch import nn


class MLPEmotionIDModel(nn.Module):
    def __init__(self, input_dim, output_classes, hidden_size=1024):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_classes)

    def forward(self, x):
        x = self.linear_1(x)
        x = nn.ReLU()(x)
        return self.linear_2(x)


class LinearEmotionIDModel(nn.Module):
    def __init__(self, in_c, output_classes):
        super().__init__()
        self.output_classes = output_classes
        self.linear_1 = nn.Linear(in_c, output_classes)

    def forward(self, x):
        x = self.linear_1(x)
        return x
