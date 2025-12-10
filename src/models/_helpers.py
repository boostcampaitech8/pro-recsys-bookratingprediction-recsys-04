import torch
import torch.nn as nn
from numpy import cumsum


# ==============================
# FeaturesEmbedding
# ==============================
class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims: list, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = [0, *cumsum(field_dims)[:-1]]

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)  # (batch_size, num_fields, embed_dim)


# ==============================
# FeaturesLinear
# ==============================
class FeaturesLinear(nn.Module):
    def __init__(self, field_dims: list, output_dim: int = 1, bias: bool = True):
        super().__init__()
        self.feature_dims = sum(field_dims)
        self.output_dim = output_dim
        self.offsets = [0, *cumsum(field_dims)[:-1]]

        self.fc = nn.Embedding(self.feature_dims, self.output_dim)
        if bias:
            self.bias = nn.Parameter(
                torch.zeros((self.output_dim,)), requires_grad=True
            )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)

    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        out = torch.sum(self.fc(x), dim=1)
        if hasattr(self, "bias"):
            out = out + self.bias
        return out


# ==============================
# FM Layer Dense
# ==============================
class FMLayer_Dense(nn.Module):
    def __init__(self):
        super().__init__()

    def square(self, x: torch.Tensor):
        return torch.pow(x, 2)

    def forward(self, x):
        square_of_sum = self.square(torch.sum(x, dim=1))
        sum_of_square = torch.sum(self.square(x), dim=1)
        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)


# ==============================
# FM Layer Sparse
# ==============================
class FMLayer_Sparse(nn.Module):
    def __init__(self, field_dims: list, factor_dim: int):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, factor_dim)
        self.fm = FMLayer_Dense()

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.fm(x)
        return x


# ==============================
# MLP Base
# ==============================
class MLP_Base(nn.Module):
    def __init__(
        self, input_dim, embed_dims, batchnorm=True, dropout=0.2, output_layer=False
    ):
        super().__init__()
        layers = []
        for idx, embed_dim in enumerate(embed_dims):
            layers.append(nn.Linear(input_dim, embed_dim))
            if batchnorm:
                layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        return self.mlp(x)


# ==============================
# CNN Base (★ 수정 완료 버전 ★)
# ==============================
class CNN_Base(nn.Module):
    def __init__(
        self,
        input_size=(3, 64, 64),
        channel_list=[8, 16, 32],
        kernel_size=3,
        stride=2,
        padding=1,
        dropout=0.2,
        batchnorm=True,
    ):
        super().__init__()

        self.cnn = nn.Sequential()
        in_channel_list = [input_size[0]] + channel_list[:-1]

        for idx, (in_channel, out_channel) in enumerate(
            zip(in_channel_list, channel_list)
        ):
            self.cnn.add_module(
                f"conv{idx}",
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )
            if batchnorm:
                self.cnn.add_module(f"batchnorm{idx}", nn.BatchNorm2d(out_channel))

            # 🔥 ReLU → LeakyReLU 로 변경
            self.cnn.add_module(f"leakyrelu{idx}", nn.LeakyReLU(negative_slope=0.01))

            if dropout > 0:
                self.cnn.add_module(f"dropout{idx}", nn.Dropout(p=dropout))

            if idx % 2 == 1:
                self.cnn.add_module(
                    f"maxpool{idx}", nn.MaxPool2d(kernel_size=2, stride=2)
                )

        self.output_dim = self.compute_output_shape((1, *input_size))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # ReLU에서 LeakyReLU로 바뀜 → kaiming init mode도 맞춰줌
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity="leaky_relu")
                nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)

    def compute_output_shape(self, input_shape):
        x = torch.rand(input_shape)
        for layer in self.cnn:
            x = layer(x)
        return x.size()

    def forward(self, x):
        return self.cnn(x)
