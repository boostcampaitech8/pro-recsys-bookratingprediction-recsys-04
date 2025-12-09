import torch
import torch.nn as nn
from ._helpers import FeaturesEmbedding, MLP_Base


# user와 item의 latent factor를 활용하여 GMF를 구현합니다.
# 그리고 MLP결과와 concat하여 NCF 모델을 구현하고 최종 결과를 도출합니다.
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data["field_dims"]
        self.user_field_idx = [0]
        self.item_field_idx = [1]
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        self.mlp = MLP_Base(
            self.embed_output_dim, args.mlp_dims, args.batchnorm, args.dropout
        )
        self.fc = nn.Linear(args.mlp_dims[-1] + args.embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        user_x = x[:, self.user_field_idx].squeeze(1)
        item_x = x[:, self.item_field_idx].squeeze(1)
        gmf = user_x * item_x
        x = self.mlp(x.view(-1, self.embed_output_dim))
        x = torch.cat([gmf, x], dim=1)
        x = self.fc(x).squeeze(1)
        return x


# user와 item의 latent factor를 활용하여 GMF를 구현합니다.
# 그리고 MLP결과와 concat하여 NCF 모델을 구현하고 최종 결과를 도출합니다.
class NeuralCollaborativeFilteringWithBias(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data["field_dims"]
        self.user_field_idx = [0]
        self.item_field_idx = [1]
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)

        # bias를 추가하기 위한 임베딩 레이어
        self.user_bias = nn.Embedding(self.field_dims[0], 1)
        self.item_bias = nn.Embedding(self.field_dims[1], 1)

        # bias 초기값을 0으로 설정
        self.user_bias.weight.data.fill_(0.0)
        self.item_bias.weight.data.fill_(0.0)

        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        self.mlp = MLP_Base(
            self.embed_output_dim, args.mlp_dims, args.batchnorm, args.dropout
        )
        self.fc = nn.Linear(args.mlp_dims[-1] + args.embed_dim, 1)
        # self.fc = nn.Linear(args.mlp_dims[-1], 1)

    def forward(self, x):
        user_idx = x[:, self.user_field_idx].long()
        item_idx = x[:, self.item_field_idx].long()

        # bias 항 추가
        b_u = self.user_bias(user_idx).squeeze()
        b_i = self.item_bias(item_idx).squeeze()

        x = self.embedding(x)
        user_x = x[:, self.user_field_idx].squeeze(1)
        item_x = x[:, self.item_field_idx].squeeze(1)

        gmf = user_x * item_x
        x = self.mlp(x.view(-1, self.embed_output_dim))
        # x = torch.cat([gmf, x], dim=1)
        x = torch.cat([gmf, x], dim=1)
        x = self.fc(x).squeeze(1)

        # bias 추가 하여 예측
        x = x + b_u + b_i
        return x
