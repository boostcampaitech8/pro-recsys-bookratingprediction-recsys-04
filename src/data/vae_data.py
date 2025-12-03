import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix

def vae_data_load(args):
    """
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser
    
    Returns
    -------
    data : dict
        학습 및 테스트 데이터가 담긴 사전 형식의 데이터를 반환합니다
    """

    ######################## DATA LOAD
    train_df = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test_df = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    all_df = pd.concat([train_df, test_df], axis=0)
    
    sparse_cols = ['user_id', 'isbn']

    # 라벨 인코딩하고 인덱스 정보를 저장
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('unknown')
        unique_labels = all_df[col].astype("category").cat.categories
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        train_df[col] = train_df[col].fillna('unknown').map(label2idx[col])
        test_df[col] = test_df[col].fillna('unknown').map(label2idx[col])

    data = {
            'train':train_df,
            'test':test_df.drop(['rating'], axis=1),
            'label2idx':label2idx,
            'idx2label':idx2label,
            'sub':sub,
            }


    return data

def vae_data_split(args, data):
    """
    Parameters
    ----------
    args.dataset.valid_ratio : float
        Train/Valid split 비율을 입력합니다.
    args.seed : int
        데이터 셔플 시 사용할 seed 값을 입력합니다.

    Returns
    -------
    data : dict
        data 내의 학습 데이터를 학습/검증 데이터(희소 행렬 형태)로 나누어 추가한 후 반환합니다.
    """
    num_users = data['label2idx']['user_id'].max() + 1
    num_items = data['label2idx']['isbn'].max() + 1
    
    if args.dataset.valid_ratio == 0:
        train_csr = csr_matrix((data['train']['rating'], (data['train']['user_id'], data['train']['isbn'])),
                               shape=(num_users, num_items))
        data['train_csr'] = train_csr

    else:
        train_df, valid_df = train_test_split(
                                            data['train'],
                                            test_size=args.dataset.valid_ratio,
                                            random_state=args.seed,
                                            stratify=data['train']['user_id'],  # 한 유저에서 적절하게 분배되도록 stratify
                                            shuffle=True
                                            )

        train_csr = csr_matrix((train_df['rating'], (train_df['user_id'], train_df['isbn'])),
                               shape=(num_users, num_items))
        valid_csr = csr_matrix((valid_df['rating'], (valid_df['user_id'], valid_df['isbn'])),
                               shape=(num_users, num_items))
        data['train_csr'] = train_csr
        data['valid_csr'] = valid_csr

    test_csr = csr_matrix((data['test']['rating'], (data['test']['user_id'], data['test']['isbn'])),
                                  shape=(num_users, num_items))
    data['test_csr'] = test_csr

    return data

class SparseDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 학습하기 위해 csr matrix를 dense matrix로 변환
        dense_matrix = self.data[idx].toarray().squeeze()
        dense_matrix = torch.tensor(dense_matrix, dtype=torch.float32)

        return dense_matrix

def vae_data_loader(args, data):
    """
    Parameters
    ----------
    args.dataloader.batch_size : int
        데이터 batch에 사용할 데이터 사이즈
    args.dataloader.shuffle : bool
        data shuffle 여부
    args.dataloader.num_workers: int
        dataloader에서 사용할 멀티프로세서 수
    args.dataset.valid_ratio : float
        Train/Valid split 비율로, 0일 경우에 대한 처리를 위해 사용합니다.
    data : dict
        basic_data_split 함수에서 반환된 데이터
    
    Returns
    -------
    data : dict
        DataLoader가 추가된 데이터를 반환합니다.
    """

    train_dataset = SparseDataset(data['train_csr'])
    valid_dataset = SparseDataset(data['valid_csr']) if args.dataset.valid_ratio != 0 else None
    test_dataset = SparseDataset(data['test_csr'])

    train_dataloader = DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers) if args.dataset.valid_ratio != 0 else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data