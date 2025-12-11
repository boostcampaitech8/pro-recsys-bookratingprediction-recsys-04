import torch
import torch.nn as nn
from ._helpers import FeaturesEmbedding, MLP_Base


# user와 item의 latent factor를 활용하여 GMF를 구현합니다.
# 그리고 MLP결과와 concat하여 NCF 모델을 구현하고 최종 결과를 도출합니다.
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.user_field_idx = [0]
        self.item_field_idx = [1]
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        self.mlp = MLP_Base(self.embed_output_dim, args.mlp_dims, args.batchnorm, args.dropout)
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
        self.field_dims = data['field_dims']
        self.user_field_idx = [0]
        self.item_field_idx = [1]
        
        # gmf를 위한 임베딩
        self.gmf_embedding = FeaturesEmbedding(self.field_dims, args.gmf_embed_dim)
        
        # mlp를 위한 임베딩
        self.mlp_embedding = FeaturesEmbedding(self.field_dims, args.mlp_embed_dim)
        
        # bias를 추가하기 위한 임베딩 레이어
        self.user_bias = nn.Embedding(self.field_dims[0], 1)
        self.item_bias = nn.Embedding(self.field_dims[1], 1)
        
        # bias 초기값을 0으로 설정
        self.user_bias.weight.data.fill_(0.)
        self.item_bias.weight.data.fill_(0.)
        
        
        self.gmf_embed_output_dim = args.gmf_embed_dim
        self.mlp_embed_output_dim = len(self.field_dims) * args.mlp_embed_dim
        
        self.mlp = MLP_Base(self.mlp_embed_output_dim, args.mlp_dims, args.batchnorm, args.dropout)
        
        #(gmf || MLP)의 결과역시 정규화를 하기위한 BatchNorm 레이어 추가
        self.concat_out_bn = nn.BatchNorm1d(self.gmf_embed_output_dim + args.mlp_dims[-1])
        
        self.fc = nn.Linear(args.mlp_dims[-1] + self.gmf_embed_output_dim, 1)
        
        # fc의 bias를 전체 평균으로 사용
        self.fc.bias.data.fill_(7.0)
        
        # dropout 층(과적합 방지)
        self.dropout = nn.Dropout(p=args.dropout)


    def forward(self, x):
        user_idx = x[:, self.user_field_idx].long()
        item_idx = x[:, self.item_field_idx].long()
        
        # bias 항 추가
        b_u = self.user_bias(user_idx).view(-1)
        b_i = self.item_bias(item_idx).view(-1)
        
        # gmf 과정
        gmf_embedding = self.gmf_embedding(x)
        user_x = gmf_embedding[:, self.user_field_idx].squeeze(1)
        item_x = gmf_embedding[:, self.item_field_idx].squeeze(1)
        gmf = user_x * item_x
        # dropout 적용(과적합 방지)
        gmf = self.dropout(gmf)
        
        # mlp 과정
        mlp_embedding = self.mlp_embedding(x)
        mlp = self.mlp(mlp_embedding.view(-1, self.mlp_embed_output_dim))
        
        concat_out = torch.cat([gmf, mlp], dim=1)
        
        bn_out = self.concat_out_bn(concat_out)
        
        # dropout 적용(과적합 방지)
        bn_out = self.dropout(bn_out)
        
        fc_out = self.fc(bn_out).squeeze(1)
        
        # bias 추가 하여 예측
        output = fc_out + b_u + b_i
        return output
