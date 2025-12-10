# src/data/catboost_data.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def catboost_data_load(args):
    """
    CatBoost 전용 데이터 로더.
    - categorical은 raw string으로 변환
    - numeric feature 포함
    - label encoding 절대 하지 않음
    """

    base_path = "/data/ephemeral/home/data/features/v3/"
    user_base_path = "/data/ephemeral/home/data/features/v4/"


    # Load data
    users = pd.read_parquet(user_base_path + "users/user_features.parquet")
    books = pd.read_parquet(base_path + "books/book_features.parquet")


    train = pd.read_csv(args.dataset.data_path + "train_ratings.csv")
    test = pd.read_csv(args.dataset.data_path + "test_ratings.csv")
    sub = pd.read_csv(args.dataset.data_path + "sample_submission.csv")

    users_, books_ = users.copy(), books.copy()

    # -----------------------------
    # 1) Feature 정의
    # -----------------------------
    categorical_features = [
        # ID-like categorical
        "user_id", "isbn",

        # user categorical
        "age_group",
        "location_country", "location_state", "location_city",
        "user_review_count_log_bin",
        "user_top_genre",
        "user_top_author",
        "user_top_publisher",
        "location_cluster",

        # book categorical
        "book_title", "book_author", "publisher", "language",
        "category", "category_missing_flag", "category_cluster",
        "summary_cluster",

        # canonical / normalized text categorical
        "title_norm", "author_norm", "canonical_id",

        # binned categorical features
        "publisher_book_count_bin",
        "publisher_review_count_bin",
        "author_book_count_bin",
        "author_review_count_bin",

        # clustering categorical
        "title_cluster",
    ]




    numeric_features = [
        # book numeric
        
        "year_of_publication_clipped",
        "age_of_book",

        # user numeric
        "age",
        "user_review_count",
        "user_review_count_log",
        "user_genre_variety",
        "user_genre_entropy",
        "user_author_variety",
        "user_mean_book_age",

        # publisher / author numeric
        "publisher_book_count",
        "publisher_book_count_log",
        "publisher_review_count",
        "publisher_review_count_log",

        "author_book_count",
        "author_book_count_log",
        "author_review_count",
        "author_review_count_log",

        # title numeric
        "title_count",
    ]

    all_features = categorical_features + numeric_features

    # -----------------------------
    # 2) Merge
    # -----------------------------
    train_df = (
        train.merge(users_, on='user_id', how='left')
             .merge(books_, on='isbn', how='left')
    )[all_features + ['rating']]

    test_df = (
        test.merge(users_, on='user_id', how='left')
            .merge(books_, on='isbn', how='left')
    )[all_features]

    # -----------------------------
    # 3) 결측치 처리
    # -----------------------------
    for col in categorical_features:
        train_df[col] = train_df[col].fillna("unknown")
        test_df[col] = test_df[col].fillna("unknown")

    for col in numeric_features:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        test_df[col] = test_df[col].fillna(train_df[col].median())

    # -----------------------------
    # 4) categorical 컬럼을 전부 문자열로 변환 (CatBoost 필수)
    # -----------------------------
    for col in categorical_features:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

    # -----------------------------
    # 5) numeric이 categorical에 섞여 있지 않도록 자동 필터링
    # -----------------------------
    categorical_features = [
        col for col in categorical_features
        if not np.issubdtype(train_df[col].dtype, np.number)
    ]

    data = {
        "train_df": train_df,
        "test_df": test_df,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "all_features": all_features,
        "sub": sub
    }

    return data



def catboost_data_split(args, data):
    """
    CatBoost는 DataFrame 형태에서 바로 split 해야 함.
    """
    train_df = data["train_df"]

    if args.dataset.valid_ratio == 0:
        data["X_train"] = train_df[data["all_features"]]
        data["y_train"] = train_df["rating"]
        return data

    X_train, X_valid, y_train, y_valid = train_test_split(
        train_df[data["all_features"]],
        train_df["rating"],
        test_size=args.dataset.valid_ratio,
        random_state=args.seed,
        shuffle=True
    )

    data["X_train"] = X_train
    data["X_valid"] = X_valid
    data["y_train"] = y_train
    data["y_valid"] = y_valid

    return data


def catboost_data_loader(args, data):
    """
    CatBoost는 DataLoader를 사용하지 않음.
    대신 DataFrame 그대로 반환한다.
    """
    data["train_dataloader"] = None
    data["valid_dataloader"] = None
    data["test_dataloader"] = None
    return data
