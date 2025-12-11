import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder

class ImplicitDataset(torch.utils.data.Dataset):
    def __init__(self, X, y
                 #user_history_dict
        ):
        self.X = X
        self.y = y
        # self.user_history_dict = user_history_dict
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        user_id = self.X[idx, 0].item()
        target_item_id = self.X[idx, 1].item()
        
        # history = list(self.user_history_dict.get(user_id, []))
        
        # target_shifted = target_item_id + 1
        
        # if target_shifted in history:
        #     history.remove(target_shifted)
        
        # max_len = 10
        # if len(history) > max_len:
        #     history = history[-max_len:]
            
        
        # if len(history) == 0:
        #     history_tensor = torch.tensor([0], dtype=torch.long)
        # else:
        #     history_tensor = torch.tensor(history, dtype=torch.long)
        
        return self.X[idx], self.y[idx]#, history_tensor

def process_user_metadata(users_path, user_encoder):
    """
    users.csv 파일을 읽어서 모델이 참조하게될 Lookup Tensor를 만듭니다.
    """
    categorical_feature_cols = ['age_group', 'location_country', 'location_cluster' ]
    numeric_feature_cols = ['user_review_count_log']
    load_cols = ['user_id'] + categorical_feature_cols + numeric_feature_cols
    
    user_df = pd.read_csv(users_path, usecols=load_cols, dtype=str)
    
    # 인코더의 순서(0, 1, 2...)대로 정렬된 뼈대 DataFrame 생성
    # all_active_users: Train + Test에 존재하는 모든 유저 ID
    # 이 순서가 모델의 User Embedding 인덱스와 일치해야 합니다.
    sorted_user_df = pd.DataFrame({'user_id': user_encoder.classes_})
    
    target_dtype = sorted_user_df['user_id'].dtype
    
    user_df['user_id'] = user_df['user_id'].astype(target_dtype)
    
    # 유저 id에 유저 정보 붙이기
    sorted_user_df = pd.merge(sorted_user_df, user_df, on='user_id', how='left')
    
    # 결측치 처리
    for col in categorical_feature_cols:
        # 각 피쳐별 결측치 처리
        sorted_user_df[col] = sorted_user_df[col].fillna("unknown")

        # 데이터 타입을 문자열로 통일 (인코딩 에러 방지용)
        sorted_user_df[col] = sorted_user_df[col].astype(str)
    
    # 피처 인코딩(각 피처를 임베딩 가능한 정수 형태로 변환)
    #       각 컬럼별로 LabelEncoder가 필요
    encoded_features = []
    feature_dims = []   # 각 피처별로 몇 개의 카테고리를 갖는지 저장
    for col in categorical_feature_cols:
        label_encoder = LabelEncoder()
        
        # 인코딩 수행
        encoded_col = label_encoder.fit_transform(sorted_user_df[col])
        
        encoded_features.append(encoded_col)
        feature_dims.append(len(label_encoder.classes_))
    
    numeric_features = []
    for col in numeric_feature_cols:
        # 수치형 데이터 0으로 설정
        sorted_user_df[col] = pd.to_numeric(sorted_user_df[col], errors='coerce')
        
        count_values = sorted_user_df[col].fillna(0.0).values
        numeric_features.append(count_values)
    
    user_meta_tensor = torch.LongTensor(np.array(encoded_features)).T
    user_numeric_meta_tensor = torch.FloatTensor(np.array(numeric_features)).T
    
    return user_meta_tensor, user_numeric_meta_tensor, feature_dims
        
def process_item_metadata(books_path, item_encoder):
    """
    books.csv 파일을 읽어서 모델이 참조하게 될 Item Lookup Tensor를 만듭니다.
    """
    # 1. 사용할 컬럼 정의 (데이터셋의 실제 컬럼명 확인 필요)
    #    예: 작가(author), 출판사(publisher), 연도(year) 등
    categorical_feature_cols = ['book_title', 'title_cluster', 'category_cluster', 'summary_cluster', 'book_author']
    numeric_feature_cols = ['publisher_book_count_log']
    load_cols = ['isbn'] + categorical_feature_cols + numeric_feature_cols
    
    # 2. 데이터 로드 (에러 방지 처리)
    try:
        # csv 파일 읽기 (컬럼이 없으면 에러나므로 확인 필수)
        item_df = pd.read_csv(books_path, usecols=load_cols, dtype=str) 
        # dtype=str: 연도 등이 숫자로 읽히면 나중에 인코딩할 때 꼬일 수 있어 문자로 통일
    except ValueError:
        # 혹시 컬럼명이 다를 경우를 대비해 전체 로드
        item_df = pd.read_csv(books_path, dtype=str)
        # 필요한 컬럼만 남기기 (여기서 컬럼명 수정 필요할 수 있음)
        item_df = item_df[load_cols]

    # 3. 뼈대 생성 (가장 중요!)
    #    item_encoder.classes_ 순서대로(0, 1, 2...) 정렬된 DataFrame 생성
    sorted_item_df = pd.DataFrame({'isbn': item_encoder.classes_})
    
    target_dtype = sorted_item_df['isbn'].dtype
    item_df['isbn'] = item_df['isbn'].astype(target_dtype)
    
    # 4. 정보 붙이기 (Left Join)
    #    ratings에는 있는데 books.csv에 없는 책은 NaN이 됨
    sorted_item_df = pd.merge(sorted_item_df, item_df, on='isbn', how='left')
    
    # 5. 결측치 처리 및 전처리
    for col in categorical_feature_cols:
        # 결측치는 'unknown'으로 채움
        sorted_item_df[col] = sorted_item_df[col].fillna("unknown")
        
        # 데이터 타입을 문자열로 통일 (인코딩 에러 방지용)
        sorted_item_df[col] = sorted_item_df[col].astype(str)
        
    
    # 6. 피처 인코딩 (LabelEncoder)
    encoded_features = []
    feature_dims = [] 
    
    for col in categorical_feature_cols:
        label_encoder = LabelEncoder()
        # fit_transform으로 숫자로 변환
        encoded_col = label_encoder.fit_transform(sorted_item_df[col])
        
        encoded_features.append(encoded_col)
        feature_dims.append(len(label_encoder.classes_))
    
    numeric_features = []
    for col in numeric_feature_cols:
        sorted_item_df[col] = pd.to_numeric(sorted_item_df[col], errors='coerce')
        count_values = sorted_item_df[col].fillna(0.0).values
        numeric_features.append(count_values)
    
    # 7. 텐서 변환 [Num_Items, Num_Features]
    item_meta_tensor = torch.LongTensor(np.array(encoded_features).T)
    item_numeric_meta_tensor = torch.FloatTensor(np.array(numeric_features).T)
    
    return item_meta_tensor, item_numeric_meta_tensor, feature_dims

def implicit_data_load(args):
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

    ######################## User ID Labeling
    # Train + Test 합집합 유저 추출
    all_users = pd.concat([train_df['user_id'], test_df['user_id']]).unique()
    
    # user_id 라벨 인코더 생성 및 학습
    user_encoder = LabelEncoder()
    user_encoder.fit(all_users)
    
    # 데이터프레임의 user_id를 라벨링한 인덱스로 변환
    train_df['user_idx'] = user_encoder.transform(train_df['user_id'])
    test_df['user_idx'] = user_encoder.transform(test_df['user_id'])
    
    num_users = len(user_encoder.classes_)
    
    
    ####################### ISBN Labeling
    # Train + Test 합집합 ISBN 추출
    all_isbns = pd.concat([train_df['isbn'], test_df['isbn']]).unique()
    
    # item_encoder 생성 및 학습
    item_encoder = LabelEncoder()
    item_encoder.fit(all_isbns)
    
    # 데이터프레임의 isbn을 라벨링한 인덱스로 변환
    train_df['item_idx'] = item_encoder.transform(train_df['isbn'])
    test_df['item_idx'] = item_encoder.transform(test_df['isbn'])
    
    num_items = len(item_encoder.classes_)
    
    
    ####################### User Metadata Tensor 생성
    user_meta_tensor, user_numeric_meta_tensor, user_feature_dims = process_user_metadata(
        args.dataset.data_path + "./users_v5.csv",
        user_encoder
    )
    
    ####################### Item Metadata Tensor 생성
    item_meta_tensor, item_numeric_meta_tensor, item_feature_dims = process_item_metadata(
        args.dataset.data_path + "./books_v5.csv",
        item_encoder
    )
    
    # 유저 수와 책의 수
    print(f"[INFO] User Count: {num_users}, Item Count: {num_items}")
    
    # 유저 메타 정보
    print(f"[Info] Meta Tensor: {user_meta_tensor.shape}")
    print(f"[Info] Meta Numeric Tensor: {user_numeric_meta_tensor.shape}")
    
    # 아이템 메타 정보
    print(f"[Info] Meta Tensor: {item_meta_tensor.shape}")
    print(f"[Info] Meta Numeric Tensor: {item_numeric_meta_tensor.shape}")
    
    # 각 피쳐별 카테고리 수
    field_dims = [num_users, num_items] + user_feature_dims + item_feature_dims
    
    # 수치형 피쳐의 개수
    num_user_numeric_feature = user_numeric_meta_tensor.shape[1]
    num_item_numeric_feature = item_numeric_meta_tensor.shape[1]
    
    ####################### User history dictionary 생성
    # print("Generating User History Dictionary...")
    
    # # 0번은 padding을 위해 1씩 쉬프트함
    # train_df['item_idx_shifted'] = train_df['item_idx'] + 1
    
    # # GroupBy로 리스트 묶기
    # # 라벨링한 user_idx를 키로 사용해야함
    # user_history_group = train_df.groupby('user_idx')['item_idx_shifted'].agg(list)

    # # 딕셔너리로 변환
    # user_history_dict = user_history_group.to_dict()
    
    # print(f"History generated for {len(user_history_dict)} users.")
    
    data = {
        'train':train_df,
        'test':test_df.drop(['rating'], axis=1),
        'field_dims':field_dims,
        
        # SVD++용 히스토리 (Item Index가 +1 시프트된 딕셔너리)
        # 'user_history_dict': user_history_dict,
        
        # 메타 정보 텐서
        'user_meta_tensor' : user_meta_tensor,
        'user_numeric_meta_tensor' : user_numeric_meta_tensor,
        'item_meta_tensor' : item_meta_tensor,
        'item_numeric_meta_tensor' : item_numeric_meta_tensor,
        
        # 모델이 슬라이싱할 때 필요한 정보
        'user_feature_dims': user_feature_dims,
        'item_feature_dims': item_feature_dims,
        
        # 수치형 데이터 필드 수 알려주는 정보
        'num_user_numeric_feature': num_user_numeric_feature,
        'num_item_numeric_feature': num_item_numeric_feature,
        
        # 원래 ID로 복원을 위한 인코더 객체    
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
            
        'sub': sub
    }

    return data


def implicit_data_split(args, data):
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
    
    use_columns = ['user_idx', 'item_idx']
    
    target_column = 'rating'
    
    if args.dataset.valid_ratio == 0:
        data['X_train'] = data['train'][use_columns]
        data['y_train'] = data['train'][target_column]

    else:
        X_train, X_valid, y_train, y_valid = train_test_split(
            data['train'][use_columns],
            data['train'][target_column],
            test_size=args.dataset.valid_ratio,
            random_state=args.seed,
            shuffle=True,
            stratify=data['train'][target_column]
        )
        data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    
    return data

def collate_fn(batch):
    X_list = [item[0] for item in batch]
    y_list = [item[1] for item in batch]
    # hist_list = [item[2] for item in batch]
    
    X_batch = torch.stack(X_list)
    y_batch = torch.stack(y_list)
    
    # hist_batch = pad_sequence(hist_list, batch_first=True, padding_value=0)
    
    return X_batch, y_batch#, hist_batch


def implicit_data_loader(args, data):
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
    
    # user_history_dict = data['user_history_dict']
    
    
    # ---- Train Loader ----
    train_dataset = ImplicitDataset(
        torch.LongTensor(data["X_train"].values),
        torch.LongTensor(data["y_train"].values),
        # user_history_dict
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.dataloader.batch_size,
        shuffle=args.dataloader.shuffle,
        num_workers=args.dataloader.num_workers,
        drop_last=True
        #collate_fn=collate_fn
    )
    
    # ---- Valid Loader ----
    valid_dataloader = None
    if args.dataset.valid_ratio != 0:
        valid_dataset = ImplicitDataset(
            torch.LongTensor(data["X_valid"].values),
            torch.LongTensor(data["y_valid"].values),
            # user_history_dict
        )
    
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=args.dataloader.batch_size,
            shuffle=False,
            num_workers=args.dataloader.num_workers,
            #collate_fn=collate_fn
        )
        
        
    # ---- Test Loader ----
    use_cols = ['user_idx', 'item_idx']    
    
    test_X = torch.LongTensor(data["test"][use_cols].values)
    dummy_y = torch.zeros(len(test_X))
    
    test_dataset = ImplicitDataset(
        test_X,
        dummy_y,
        # user_history_dict
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.dataloader.batch_size,
        shuffle=False,
        num_workers=args.dataloader.num_workers,
        #collate_fn=collate_fn
    )

    data["train_dataloader"], data["valid_dataloader"], data["test_dataloader"] = (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
    )

    return data


# def generate_kfold_indices(args, data):
#     """
#     user_id를 기준으로 Stratified K-Fold 인덱스 리스트 생성
        
#     Parameters
#     ----------
#     args.kfold.n_splits : int
#         fold의 수
#     args.kfold.shuffle : bool
#         fold를 만들기 전 데이터 셔플 여부
#     args.seed: int
#         랜덤 시드
    
#     Returns
#     -------
#     data : list
#         각 폴드별 train_idx 리스트와, valid_idx 리스트를 튜플로 묶은 리스트를 반환합니다
#     """
    
#     # user_id 의 분포를 유지하며 kfold
#     stratify_target = data['train']['user_id'].values
    
#     # split 함수를 위한 더미 데이터
#     X_dummy = np.zeros(len(stratify_target))
    
#     # K-fold 객체 생성
#     skf = KFold(n_splits=args.kfold.n_splits, shuffle=args.kfold.shuffle, random_state=args.seed)
    
#     # 인덱스 리스트 생성
#     folds_indices = []
    
#     # skf.split은 인덱스(정수)를 반환
#     for train_idx, valid_idx in skf.split(X_dummy, stratify_target):
#         folds_indices.append((train_idx, valid_idx))
    
#     return folds_indices

# def basic_kfold_split(all_X, all_y, train_idx, valid_idx):
#     """
#     전체 Tensor(all_X, all_y)를 인덱스 배열을 사용해 슬라이싱
#     DataFrame이 아닌 Tensor 상태에서 자르므로 속도 향상

#     Parameters
#     ----------
#     all_X: Tensor
#         data tensor
#     all_y: Tensor
#         label tensor
#     train_idx: list
#         각 폴드별 train indices
#     valid_idx: list
#         각 폴드별 valid indices
#     """
    
#     # 학습용 데이터 슬라이싱
#     X_train = all_X[train_idx]
#     y_train = all_y[train_idx]
    
#     # 검증용 데이터 슬라이싱
#     X_valid = all_X[valid_idx]
#     y_valid = all_y[valid_idx]
    
#     return X_train, y_train, X_valid, y_valid

# def basic_kfold_loader(args, X_train, y_train, X_valid, y_valid):
    
#     # TensorDataset 생성
#     train_dataset = TensorDataset(X_train, y_train)
#     valid_dataset = TensorDataset(X_valid, y_valid)
    
#     # DataLoader 생성
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.dataloader.batch_size,
#         shuffle=True,
#         num_workers=args.dataloader.num_workers
#     )
    
#     valid_loader = DataLoader(
#         valid_dataset,
#         batch_size=args.dataloader.batch_size,
#         shuffle=False, # 검증 데이터는 섞을 필요 없음
#         num_workers=args.dataloader.num_workers
#     )
    
#     return train_loader, valid_loader
    
    


    
