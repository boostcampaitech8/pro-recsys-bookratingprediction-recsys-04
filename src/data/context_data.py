import numpy as np
import pandas as pd
import regex
import torch
from torch.utils.data import TensorDataset, DataLoader
from .basic_data import basic_data_split


def str2list(x: str) -> list:
    """문자열을 리스트로 변환하는 함수"""
    return x[1:-1].split(", ")


def split_location(x: str) -> list:
    """
    Parameters
    ----------
    x : str
        location 데이터

    Returns
    -------
    res : list
        location 데이터를 나눈 뒤, 정제한 결과를 반환합니다.
        순서는 country, state, city, ... 입니다.
    """
    res = x.split(",")
    res = [i.strip().lower() for i in res]
    res = [regex.sub(r"[^a-zA-Z/ ]", "", i) for i in res]  # remove special characters
    res = [i if i not in ["n/a", ""] else np.nan for i in res]  # change 'n/a' into NaN
    res.reverse()  # reverse the list to get country, state, city, ... order

    for i in range(len(res) - 1, 0, -1):
        if (res[i] in res[:i]) and (
            not pd.isna(res[i])
        ):  # remove duplicated values if not NaN
            res.pop(i)

    return res


def process_context_data(users, books):
    """
    users.csv와 books.csv를 전처리하여 반환합니다.
    (location 문자열 컬럼이 없으므로 country/state/city만 사용)
    """

    users_ = users.copy()
    books_ = books.copy()

    # -----------------------------
    # 📌 BOOKS 전처리
    # -----------------------------
    books_["category"] = books_["category"].apply(
        lambda x: str2list(x)[0] if not pd.isna(x) else np.nan
    )
    books_["language"] = books_["language"].fillna(books_["language"].mode()[0])
    books_["publication_range"] = books_["year_of_publication"].apply(
        lambda x: x // 10 * 10
    )

    # -----------------------------
    # 📌 USERS 전처리
    # -----------------------------
    users_["age"] = users_["age"].fillna(users_["age"].mode()[0])
    users_["age_range"] = users_["age"].apply(lambda x: x // 10 * 10)

    # 📌 location_country/state/city는 이미 users.csv에 존재함
    # → 여기서는 결측치만 단순 처리
    if "location_country" not in users_.columns:
        users_["location_country"] = "unknown"

    if "location_state" not in users_.columns:
        users_["location_state"] = "unknown"

    if "location_city" not in users_.columns:
        users_["location_city"] = "unknown"

    users_["location_country"] = users_["location_country"].fillna("unknown")
    users_["location_state"] = users_["location_state"].fillna("unknown")
    users_["location_city"] = users_["location_city"].fillna("unknown")

    # 📌 기존 location 문자열 기반 파싱은 제거
    # (users_["location_list"], split_location 등 삭제)

    return users_, books_


# def process_context_data(users, books):
#     """
#     Parameters
#     ----------
#     users : pd.DataFrame
#         users.csv를 인덱싱한 데이터
#     books : pd.DataFrame
#         books.csv를 인덱싱한 데이터
#     ratings1 : pd.DataFrame
#         train 데이터의 rating
#     ratings2 : pd.DataFrame
#         test 데이터의 rating

#     Returns
#     -------
#     label_to_idx : dict
#         데이터를 인덱싱한 정보를 담은 딕셔너리
#     idx_to_label : dict
#         인덱스를 다시 원래 데이터로 변환하는 정보를 담은 딕셔너리
#     train_df : pd.DataFrame
#         train 데이터
#     test_df : pd.DataFrame
#         test 데이터
#     """

#     users_ = users.copy()
#     books_ = books.copy()

#     # 데이터 전처리 (전처리는 각자의 상황에 맞게 진행해주세요!)
#     books_["category"] = books_["category"].apply(
#         lambda x: str2list(x)[0] if not pd.isna(x) else np.nan
#     )
#     books_["language"] = books_["language"].fillna(books_["language"].mode()[0])
#     books_["publication_range"] = books_["year_of_publication"].apply(
#         lambda x: x // 10 * 10
#     )  # 1990년대, 2000년대, 2010년대, ...

#     users_["age"] = users_["age"].fillna(users_["age"].mode()[0])
#     users_["age_range"] = users_["age"].apply(
#         lambda x: x // 10 * 10
#     )  # 10대, 20대, 30대, ...

#     users_["location_list"] = users_["location"].apply(lambda x: split_location(x))
#     users_["location_country"] = users_["location_list"].apply(lambda x: x[0])
#     users_["location_state"] = users_["location_list"].apply(
#         lambda x: x[1] if len(x) > 1 else np.nan
#     )
#     users_["location_city"] = users_["location_list"].apply(
#         lambda x: x[2] if len(x) > 2 else np.nan
#     )
#     for idx, row in users_.iterrows():
#         if (not pd.isna(row["location_state"])) and pd.isna(row["location_country"]):
#             fill_country = users_[users_["location_state"] == row["location_state"]][
#                 "location_country"
#             ].mode()
#             fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
#             users_.loc[idx, "location_country"] = fill_country
#         elif (not pd.isna(row["location_city"])) and pd.isna(row["location_state"]):
#             if not pd.isna(row["location_country"]):
#                 fill_state = users_[
#                     (users_["location_country"] == row["location_country"])
#                     & (users_["location_city"] == row["location_city"])
#                 ]["location_state"].mode()
#                 fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
#                 users_.loc[idx, "location_state"] = fill_state
#             else:
#                 fill_state = users_[users_["location_city"] == row["location_city"]][
#                     "location_state"
#                 ].mode()
#                 fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
#                 fill_country = users_[users_["location_city"] == row["location_city"]][
#                     "location_country"
#                 ].mode()
#                 fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
#                 users_.loc[idx, "location_country"] = fill_country
#                 users_.loc[idx, "location_state"] = fill_state

#     users_ = users_.drop(["location"], axis=1)

#     return users_, books_


def context_data_load(args):
    """
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser

    Returns
    -------
    data : dict
        학습 및 테스트 데이터가 담긴 사전 형식의 데이터를 반환합니다.
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.dataset.data_path + "users.csv")
    books = pd.read_csv(args.dataset.data_path + "books.csv")
    train = pd.read_csv(args.dataset.data_path + "train_ratings.csv")
    test = pd.read_csv(args.dataset.data_path + "test_ratings.csv")
    sub = pd.read_csv(args.dataset.data_path + "sample_submission.csv")

    users_, books_ = process_context_data(users, books)

    # 유저 및 책 정보를 합쳐서 데이터 프레임 생성
    # 사용할 컬럼을 user_features와 book_features에 정의합니다. (단, 모두 범주형 데이터로 가정)
    # 베이스라인에서는 가능한 모든 컬럼을 사용하도록 구성하였습니다.
    # NCF를 사용할 경우, idx 0, 1은 각각 user_id, isbn이어야 합니다.
    user_features = [
        "user_id",
        "age_range",
        "location_country",
        "location_state",
        "location_city",
    ]
    book_features = [
        "isbn",
        "book_title",
        "book_author",
        "publisher",
        "language",
        "category",
        "publication_range",
    ]
    sparse_cols = (
        ["user_id", "isbn"]
        + list(set(user_features + book_features) - {"user_id", "isbn"})
        if args.model == "NCF"
        else user_features + book_features
    )

    # 선택한 컬럼만 추출하여 데이터 조인
    train_df = train.merge(users_, on="user_id", how="left").merge(
        books_, on="isbn", how="left"
    )[sparse_cols + ["rating"]]
    test_df = test.merge(users_, on="user_id", how="left").merge(
        books_, on="isbn", how="left"
    )[sparse_cols]
    all_df = pd.concat([train_df, test_df], axis=0)

    # feature_cols의 데이터만 라벨 인코딩하고 인덱스 정보를 저장
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        # 1. 전체 데이터 기준으로 유니크 라벨 추출 및 사전 생성
        all_df[col] = all_df[col].fillna("unknown")
        unique_labels = all_df[col].astype("category").cat.categories
        label2idx[col] = {label: idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx: label for idx, label in enumerate(unique_labels)}

        # 2. [수정됨] 생성된 사전을 이용해 train/test 일관성 있게 매핑
        # 주의: merge 과정에서 생길 수 있는 NaN을 위해 여기서도 fillna를 해줍니다.
        train_df[col] = train_df[col].fillna("unknown").map(label2idx[col])
        test_df[col] = test_df[col].fillna("unknown").map(label2idx[col])

    # field_dims 계산 시 rating 컬럼 제외하고 정확히 계산
    # (train_df.columns에는 rating이 포함되어 있으므로 주의)
    field_dims = [len(label2idx[col]) for col in sparse_cols]

    data = {
        "train": train_df,
        "test": test_df,
        "field_names": sparse_cols,
        "field_dims": field_dims,
        "label2idx": label2idx,
        "idx2label": idx2label,
        "sub": sub,
    }

    return data


def context_data_split(args, data):
    """data 내의 학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다."""
    return basic_data_split(args, data)


def context_data_loader(args, data):
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
        context_data_load 함수에서 반환된 데이터

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
