import numpy as np
import torch
import torch.nn as nn
from ._helpers import FeaturesEmbedding, MLP_Base

# Netflix 방식의 Embedding bag 모듈
class NetflixEmbeddingBag(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=0):
        super().__init__()
        # [핵심 1] mode='sum'으로 설정하여 일단 벡터들을 다 더합니다.
        self.embedding = nn.EmbeddingBag(
            num_embeddings, 
            embedding_dim, 
            mode='sum', 
            padding_idx=padding_idx
        )
        self.padding_idx = padding_idx
        self.init_weights()

    def forward(self, input):
        # input shape: (batch_size, seq_len)
        
        # 1. 임베딩 합계 계산 (Sum)
        embed_sum = self.embedding(input)
        
        # 2. 각 유저가 본 아이템 개수(N) 계산
        # 패딩(0)이 아닌 것만 카운트합니다.
        # (True=1, False=0 이므로 sum하면 개수가 나옴)
        n_items = (input != self.padding_idx).sum(dim=1, keepdim=True).float()
        
        # 3. 0으로 나누는 것 방지 (최소값을 1.0으로 설정)
        # 아이템을 하나도 안 본 유저가 있을 경우를 대비함
        n_items = n_items.clamp(min=1.0)
        
        # [핵심 2] 합계(Sum)를 개수의 제곱근(Sqrt N)으로 나눔
        # SVD++ 논문 수식: Sum / sqrt(|N(u)|)
        norm_embed = embed_sum / torch.sqrt(n_items)
        
        return norm_embed

    def init_weights(self):
        # 1. Xavier 초기화
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        
        # 2. Padding Index 0으로 초기화 (안전장치)
        with torch.no_grad():
            self.embedding.weight[self.padding_idx].fill_(0)

# user와 item의 latent factor를 활용하여 GMF를 구현합니다.
class MFSVDPlusModel(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.user_field_idx = [0]
        self.item_field_idx = [1]
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        
        # Implicit History Embedding
        self.history_embedding = NetflixEmbeddingBag(
            self.field_dims[1] + 1,
            args.embed_dim,
            padding_idx=0
        )
        
        # bias를 추가하기 위한 임베딩 레이어
        self.user_bias = nn.Embedding(self.field_dims[0], 1)
        self.item_bias = nn.Embedding(self.field_dims[1], 1)
        
        # bias 초기값을 0으로 설정
        self.user_bias.weight.data.fill_(0.)
        self.item_bias.weight.data.fill_(0.)
        
        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        
        self.fc = nn.Linear(args.embed_dim, 1)
        self.dropout = nn.Dropout(p=0.5)
        
        self.__init_weights(data)

    def __init_weights(self, data):
        try:
            y_train = data.get('y_train')
            if y_train is not None:
                if isinstance(y_train, torch.Tensor):
                    global_mean = y_train.float().mean().item()
                else:
                    global_mean = float(np.mean(y_train))
            else:
                global_mean = 0.0
        except Exception as e:
            print(f"Warning: Could not calculate global mean: {e}")
            global_mean = 0.0

        print(f"[Init] Global Bias (FC) initialized to: {global_mean:.4f}")
        
        # FC Weight는 Xavier, Bias는 평균 값으로 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(global_mean)
        
            
        # Bias Embedding 0으로 초기화
        self.user_bias.weight.data.fill_(0.)
        self.item_bias.weight.data.fill_(0.)
        

    def forward(self, x, history):
        user_idx = x[:, self.user_field_idx].long()
        item_idx = x[:, self.item_field_idx].long()
        
        # bias 항 추가
        b_u = self.user_bias(user_idx).view(-1)
        b_i = self.item_bias(item_idx).view(-1)
        
        x = self.embedding(x)
        user_x = x[:, self.user_field_idx].squeeze(1)
        item_x = x[:, self.item_field_idx].squeeze(1)
        
        implicit_u = self.history_embedding(history)
        
        user_x = user_x + implicit_u
    
        gmf = user_x * item_x
        
        gmf = self.dropout(gmf)
        
        out = self.fc(gmf).view(-1)
        
        # bias 추가 하여 예측
        out = out + b_u + b_i
        return out
