import numpy as np
import torch
import torch.nn as nn
from ._helpers import FeaturesEmbedding, MLP_Base


# user와 item의 latent factor를 활용하여 GMF를 구현합니다.
class DualHybridGMFModel(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        # 예: [user_idx 수, item_idx 수, user_feats..., item_feats...]
        
        # 인덱스 정의
        self.user_field_idx = 0
        self.item_field_idx = 1
        # 유저 메타 피처가 시작되는 인덱스
        self.feature_field_start_idx = 2
        
        # 유저 메타데이터 장부 등록
        # 데이터 로더에서 만든 텐서를 여기서 받습니다.
        if 'user_meta_tensor' in data:
            self.register_buffer('user_features', data['user_meta_tensor'])
        else:
            self.user_features = None
            
        if 'user_numeric_meta_tensor' in data:
            self.register_buffer('user_numeric_features', data['user_numeric_meta_tensor'])
        else:
            self.user_numeric_features = None
        
        # 아이템 메타데이터 장부 등록
        # 데이터 로더에서 만든 텐서를 여기서 받습니다.
        if 'item_meta_tensor' in data:
            self.register_buffer('item_features', data['item_meta_tensor'])
        else:
            self.item_features = None
        
        if 'item_numeric_meta_tensor' in data:
            self.register_buffer('item_numeric_features', data['item_numeric_meta_tensor'])
        else:
            self.item_numeric_features = None
        
        
        # 슬라이싱을 위한 각 특성의 개수 저장
        self.num_users_feats = len(data.get('user_feature_dims', []))
        self.num_items_feats = len(data.get('item_feature_dims', []))
        
        # 수치형 피쳐의 개수
        num_user_numeric_feature = data.get('num_user_numeric_feature')
        num_item_numeric_feature = data.get('num_item_numeric_feature')
        
        total_numeric_dim = num_user_numeric_feature + num_item_numeric_feature
        
        
        # 모든 피처 별 통합 임베딩 사용(임베딩 차원은 모두 동일하기때문에 가능)
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        
        ############### Projection Layer ##############
        user_feat_concat_dim = args.embed_dim * self.num_users_feats
        # self.user_feat_proj = nn.Linear(user_feat_concat_dim, args.embed_dim)
        self.user_feat_proj = nn.Sequential(
            nn.Linear(user_feat_concat_dim, user_feat_concat_dim),
            nn.BatchNorm1d(user_feat_concat_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(user_feat_concat_dim, args.embed_dim)
        )
        
        item_feat_concat_dim = args.embed_dim * self.num_items_feats
        self.item_feat_proj = nn.Sequential(
            nn.Linear(item_feat_concat_dim, item_feat_concat_dim),
            nn.BatchNorm1d(item_feat_concat_dim),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(item_feat_concat_dim, args.embed_dim)
        )
        #nn.Linear(item_feat_concat_dim, args.embed_dim)
        ############# Gate MLP ################
        if total_numeric_dim > 0:
            self.gate_mlp = nn.Sequential(
                nn.Linear(total_numeric_dim, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
                nn.Sigmoid()
            )
        else:
            self.gate_mlp = None
        
        ############## Bias & Output #################
        
        # bias를 추가하기 위한 임베딩 레이어
        self.user_bias = nn.Embedding(self.field_dims[0], 1)
        self.item_bias = nn.Embedding(self.field_dims[1], 1)
        self.feature_bias = FeaturesEmbedding(self.field_dims[2:], 1)
        
        # ID_GMF 결과 + Feature_GMF 결과 = 2 * dim
        self.fc = nn.Linear(args.embed_dim * 2, 1)
        self.dropout = nn.Dropout(args.dropout)
        
        self.__init_weights(data)

    def __init_weights(self, data):
        
        # FC Weight는 Xavier, Bias는 평균 값으로 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)
        
        # 유저 or 아이템 차원축소 weight 초기화
        # torch.nn.init.xavier_uniform_(self.user_feat_proj.weight)
        # torch.nn.init.xavier_uniform_(self.item_feat_proj.weight)
        for m in self.user_feat_proj.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
                
        for m in self.item_feat_proj.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
        
        # global bias 설정
        global_mean = 0.0
        try:
            if 'y_train' in data:
                global_mean = data['y_train'].float().mean().item()
        except:
            pass
        
        self.fc.bias.data.fill_(global_mean)
        
        # Bias Embedding 0으로 초기화
        self.user_bias.weight.data.fill_(0.)
        self.item_bias.weight.data.fill_(0.)
        for m in self.feature_bias.modules():
            if isinstance(m, nn.Embedding):
                nn.init.constant_(m.weight.data, 0)
        

    def forward(self, x):
                #history):
        # 입력데이터: x: [Batch, 2] -> (user_idx, item_idx) 
        user_idx = x[:, self.user_field_idx].long()
        item_idx = x[:, self.item_field_idx].long()
        
        ######### 입력 데이터 확장(유저 메타정보 주입)
        indices_list = [x]
        
        if self.user_features is not None:
            # 장부에서 피처 조회: [Batch, Num_Features]
            indices_list.append(self.user_features[user_idx])
        
        if self.item_features is not None:
            indices_list.append(self.item_features[item_idx])
        
        all_indices = torch.cat(indices_list, dim=1)
        
        ######### 임베딩 조회 및 슬라이싱
        all_embs = self.embedding(all_indices)
        
        user_id_vec = all_embs[:, self.user_field_idx]  # [Batch, dim]
        item_id_vec = all_embs[:, self.item_field_idx]     # [Batch, dim]
        
        feature_vecs = all_embs[:, self.feature_field_start_idx:]   # [Batch, n_feats, dim]
        
        user_feat_vec = feature_vecs[:, :self.num_users_feats]
        item_feat_vec = feature_vecs[:, self.num_users_feats:]
        
        ######### ID GMF
        gmf_id = user_id_vec * item_id_vec
        
        ######## Feature GMF
        
        # 유저 특성압축
        user_feat_flat = user_feat_vec.reshape(x.size(0), -1)
        user_feat_summary = self.user_feat_proj(user_feat_flat)
        # user_feat_summary = nn.Tanh(user_feat_summary)
        
        # 아이템 특성 압축
        item_feat_flat = item_feat_vec.reshape(x.size(0), -1)
        item_feat_summary = self.item_feat_proj(item_feat_flat)
        # item_feat_summary = nn.Tanh(user_feat_summary)
        # 비선형 추가는 선택
        
        gmf_feat = user_feat_summary * item_feat_summary
        
        ############ Gating #############
        if self.gate_mlp is not None:
            u_num = self.user_numeric_features[user_idx]
            i_num = self.item_numeric_features[item_idx]
            
            numeric_features_concat = torch.cat([u_num, i_num], dim=1)
            gates = self.gate_mlp(numeric_features_concat)
            alpha = gates[:, 0].unsqueeze(1)
            beta = gates[:, 1].unsqueeze(1)
            
            weighted_gmf_id = alpha * gmf_id
            weighted_gmf_feat = beta * gmf_feat
        else:
            weighted_gmf_id = 0.5 * gmf_id
            weighted_gmf_feat = 0.5 * gmf_feat
            
        
        ########## Fusion & Prediction
        concat_vec = torch.cat([weighted_gmf_id, weighted_gmf_feat], dim=1)
        
        # 과적합 방지
        concat_vec = self.dropout(concat_vec)
        
        out = self.fc(concat_vec).view(-1)
        
        ############ Bias 추가
        b_u = self.user_bias(user_idx).view(-1)
        b_i = self.item_bias(item_idx).view(-1)
        
        feat_indices = all_indices[:, 2:]
        b_feat_sum = self.feature_bias(feat_indices).sum(dim=1).squeeze(-1)
        
        # bias 추가 하여 예측
        out = out + b_u + b_i + b_feat_sum
        return out
