import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import torch
from torch.utils.data import TensorDataset, DataLoader

def basic_data_load(args):
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
    train_df = pd.read_csv(args.dataset.data_path + "train_ratings.csv")
    test_df = pd.read_csv(args.dataset.data_path + "test_ratings.csv")
    sub = pd.read_csv(args.dataset.data_path + "sample_submission.csv")

    # 처리를 위해 잠시 합칩니다 (나중에 분리할 때 원본 보존을 위해 copy 사용 추천하지만, 여기선 흐름상 진행)
    all_df = pd.concat([train_df, test_df], axis=0)

    sparse_cols = ["user_id", "isbn"]

    # 라벨 인코딩하고 인덱스 정보를 저장
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        # 1. 전체 데이터 기준으로 결측치 처리 및 유니크 라벨 추출
        all_df[col] = all_df[col].fillna("unknown")
        unique_labels = all_df[col].astype("category").cat.categories

        # 2. 사전 생성
        label2idx[col] = {label: idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx: label for idx, label in enumerate(unique_labels)}

        # 3. [중요] 생성된 사전을 이용해 train/test 각각 매핑 (일관성 유지)
        # 데이터프레임 원본에도 fillna를 해줘야 매핑 시 에러가 안 납니다.
        train_df[col] = train_df[col].fillna("unknown").map(label2idx[col])
        test_df[col] = test_df[col].fillna("unknown").map(label2idx[col])

    field_dims = [len(label2idx[col]) for col in sparse_cols]

    data = {
            'train':train_df,
            'test':test_df.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'label2idx':label2idx,
            'idx2label':idx2label,
            'sub':sub,
            }


    return data


def basic_data_split(args, data):
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
        data 내의 학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다.
    """
    if args.dataset.valid_ratio == 0:
        data['X_train'] = data['train'].drop('rating', axis=1)
        data['y_train'] = data['train']['rating']

    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
                                                            data['train'].drop(['rating'], axis=1),
                                                            data['train']['rating'],
                                                            test_size=args.dataset.valid_ratio,
                                                            random_state=args.seed,
                                                            shuffle=True
                                                            )
        data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    
    return data

def basic_data_loader(args, data):
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

    train_dataset = TensorDataset(
        torch.LongTensor(data["X_train"].values),
        torch.LongTensor(data["y_train"].values),
    )
    valid_dataset = (
        TensorDataset(
            torch.LongTensor(data["X_valid"].values),
            torch.LongTensor(data["y_valid"].values),
        )
        if args.dataset.valid_ratio != 0
        else None
    )
    test_dataset = TensorDataset(torch.LongTensor(data["test"].values))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.dataloader.batch_size,
        shuffle=args.dataloader.shuffle,
        num_workers=args.dataloader.num_workers,
    )
    valid_dataloader = (
        DataLoader(
            valid_dataset,
            batch_size=args.dataloader.batch_size,
            shuffle=False,
            num_workers=args.dataloader.num_workers,
        )
        if args.dataset.valid_ratio != 0
        else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.dataloader.batch_size,
        shuffle=False,
        num_workers=args.dataloader.num_workers,
    )

    data["train_dataloader"], data["valid_dataloader"], data["test_dataloader"] = (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
    )

    return data


def generate_kfold_indices(args, data):
    """
    user_id를 기준으로 Stratified K-Fold 인덱스 리스트 생성
        
    Parameters
    ----------
    args.kfold.n_splits : int
        fold의 수
    args.kfold.shuffle : bool
        fold를 만들기 전 데이터 셔플 여부
    args.seed: int
        랜덤 시드
    
    Returns
    -------
    data : list
        각 폴드별 train_idx 리스트와, valid_idx 리스트를 튜플로 묶은 리스트를 반환합니다
    """
    
    # user_id 의 분포를 유지하며 kfold
    stratify_target = data['train']['user_id'].values
    
    # split 함수를 위한 더미 데이터
    X_dummy = np.zeros(len(stratify_target))
    
    # K-fold 객체 생성
    skf = KFold(n_splits=args.kfold.n_splits, shuffle=args.kfold.shuffle, random_state=args.seed)
    
    # 인덱스 리스트 생성
    folds_indices = []
    
    # skf.split은 인덱스(정수)를 반환
    for train_idx, valid_idx in skf.split(X_dummy, stratify_target):
        folds_indices.append((train_idx, valid_idx))
    
    return folds_indices

def basic_kfold_split(all_X, all_y, train_idx, valid_idx):
    """
    전체 Tensor(all_X, all_y)를 인덱스 배열을 사용해 슬라이싱
    DataFrame이 아닌 Tensor 상태에서 자르므로 속도 향상

    Parameters
    ----------
    all_X: Tensor
        data tensor
    all_y: Tensor
        label tensor
    train_idx: list
        각 폴드별 train indices
    valid_idx: list
        각 폴드별 valid indices
    """
    
    # 학습용 데이터 슬라이싱
    X_train = all_X[train_idx]
    y_train = all_y[train_idx]
    
    # 검증용 데이터 슬라이싱
    X_valid = all_X[valid_idx]
    y_valid = all_y[valid_idx]
    
    return X_train, y_train, X_valid, y_valid

def basic_kfold_loader(args, X_train, y_train, X_valid, y_valid):
    
    # TensorDataset 생성
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.dataloader.batch_size,
        shuffle=True,
        num_workers=args.dataloader.num_workers
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.dataloader.batch_size,
        shuffle=False, # 검증 데이터는 섞을 필요 없음
        num_workers=args.dataloader.num_workers
    )
    
    return train_loader, valid_loader
    
    


    
