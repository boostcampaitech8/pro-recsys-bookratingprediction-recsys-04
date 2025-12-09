import numpy as np
import pandas as pd
import regex
import torch
from torch.utils.data import TensorDataset, DataLoader
from .basic_data import basic_data_split
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# -------------------------------
# 1. Helper 함수
# -------------------------------
def str2list(x: str) -> list:
    return x[1:-1].split(",")

def split_location(x: str) -> list:
    res = x.split(",")
    res = [i.strip().lower() for i in res]
    res = [regex.sub(r"[^a-zA-Z/ ]", "", i) for i in res]
    res = [i if i not in ["n/a", ""] else np.nan for i in res]
    res.reverse()
    for i in range(len(res)-1, 0, -1):
        if (res[i] in res[:i]) and (not pd.isna(res[i])):
            res.pop(i)
    return res

# -------------------------------
# 2. Category 처리 & 클러스터링
# -------------------------------
def preprocess_category(df, col='category', min_len=3):
    df[col] = df[col].fillna('unknown')
    df[col] = df[col].str.strip().str.lower()
    df.loc[df[col].str.len() < min_len, col] = 'other'
    return df

def augment_category_text(df, col='category'):
    df['category_text'] = df[col].apply(lambda x: f"Category: {x}")
    return df

def embed_categories(df, text_col='category_text', model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    categories = df[text_col].tolist()
    embeddings = model.encode(categories, batch_size=64, show_progress_bar=True)
    return np.array(embeddings)

def cluster_categories(embeddings, n_clusters=200, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

# -------------------------------
# 3. User age_group 처리
# -------------------------------
def process_user_age(user_df):
    # Winsorizing
    lower_bound = user_df['age'].quantile(0.05)
    upper_bound = user_df['age'].quantile(0.95)
    user_df['age_winsor'] = user_df['age'].clip(lower=lower_bound, upper=upper_bound)

    # Age group
    bins = [17, 26, 36, 46, 56, 61]
    labels = ['17-25', '26-35', '36-45', '46-55', '56-61']
    user_df['age_group'] = pd.cut(user_df['age_winsor'], bins=bins, labels=labels, right=True)
    return user_df

# -------------------------------
# 4. 전체 데이터 전처리
# -------------------------------
def process_context_data(users, books, n_clusters=200):
    users_ = users.copy()
    books_ = books.copy()

    # --- User
    users_['age'] = users_['age'].fillna(users_['age'].mode()[0])
    users_ = process_user_age(users_)

    users_["location_list"] = users_["location"].apply(lambda x: split_location(x))
    users_["location_country"] = users_["location_list"].apply(lambda x: x[0])
    users_["location_state"] = users_["location_list"].apply(lambda x: x[1] if len(x)>1 else np.nan)
    users_["location_city"] = users_["location_list"].apply(lambda x: x[2] if len(x)>2 else np.nan)

    for idx, row in users_.iterrows():
        if (not pd.isna(row["location_state"])) and pd.isna(row["location_country"]):
            fill_country = users_[users_["location_state"] == row["location_state"]]["location_country"].mode()
            users_.loc[idx, "location_country"] = fill_country[0] if len(fill_country)>0 else np.nan
        elif (not pd.isna(row["location_city"])) and pd.isna(row["location_state"]):
            if not pd.isna(row["location_country"]):
                fill_state = users_[(users_["location_country"]==row["location_country"]) & (users_["location_city"]==row["location_city"])]["location_state"].mode()
                users_.loc[idx, "location_state"] = fill_state[0] if len(fill_state)>0 else np.nan
            else:
                fill_state = users_[users_["location_city"]==row["location_city"]]["location_state"].mode()
                fill_country = users_[users_["location_city"]==row["location_city"]]["location_country"].mode()
                users_.loc[idx, "location_state"] = fill_state[0] if len(fill_state)>0 else np.nan
                users_.loc[idx, "location_country"] = fill_country[0] if len(fill_country)>0 else np.nan

    users_ = users_.drop(['location'], axis=1)

    # --- Book
    books_ = preprocess_category(books_, col='category', min_len=3)
    books_ = augment_category_text(books_, col='category')
    embeddings = embed_categories(books_, text_col='category_text')
    books_['category_cluster'] = cluster_categories(embeddings, n_clusters=n_clusters)

    # 안전: year_of_publication 숫자 변환 (변환 실패는 NaN)
    books_['year_of_publication'] = pd.to_numeric(books_['year_of_publication'], errors='coerce')

    # ----------------------------------------
    # 1) IQR 기반 정상 범위 계산 (이상치 판정용)
    # ----------------------------------------
    Q1, Q3 = books_['year_of_publication'].quantile([0.25, 0.75])
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    print("정상 출판년도 범위:", int(lower), "~", int(upper))

    # ----------------------------------------
    # 2) Clip 방식 적용 (원본 유지 + clipped 컬럼 추가)
    # ----------------------------------------
    books_['year_of_publication_clipped'] = books_['year_of_publication'].clip(lower, upper)

    # ----------------------------------------
    # 3) publication_range (십 년대) 생성 — 결측은 그대로 np.nan 유지
    # ----------------------------------------
    def to_decade_year(x):
        if pd.isna(x):
            return np.nan
        return int((x // 10) * 10)

    books_['publication_range'] = books_['year_of_publication_clipped'].apply(to_decade_year)

    # 기존 category 컬럼은 필요하면 유지, cluster를 feature로 사용
    return users_, books_

# -------------------------------
# 5. Data Load
# -------------------------------
def context_data_load(args):
    users = pd.read_csv(args.dataset.data_path + "users.csv")
    books = pd.read_csv(args.dataset.data_path + "books.csv")
    train = pd.read_csv(args.dataset.data_path + "train_ratings.csv")
    test = pd.read_csv(args.dataset.data_path + "test_ratings.csv")
    sub = pd.read_csv(args.dataset.data_path + "sample_submission.csv")

    users_, books_ = process_context_data(users, books, n_clusters=200)

    user_features = ["user_id", "age_group", "location_country", "location_state", "location_city"] + ["age"]
    book_features = ["isbn", "book_title", "book_author", "publisher", "language", "category_cluster", "publication_range"] +["category"]

    sparse_cols = user_features + book_features

    # Train/test join
    train_df = train.merge(users_, on="user_id", how="left").merge(books_, on="isbn", how="left")[sparse_cols + ["rating"]]
    test_df = test.merge(users_, on="user_id", how="left").merge(books_, on="isbn", how="left")[sparse_cols]

    all_df = pd.concat([train_df, test_df], axis=0)

    label2idx, idx2label = {}, {}
        # --- 안전한 fillna 처리
    categorical_fillna = ["age_group", "location_country", "location_state", "location_city",
                          "book_title", "book_author", "publisher", "language", "category_cluster"]
    # 수치형/ID: fillna 유지 (NaN) / ID는 결측 없다고 가정

    for col in sparse_cols:
        all_df[col] = all_df[col].astype('category')

        # 카테고리형은 "unknown" 추가
        if col in categorical_fillna:
            if "unknown" not in all_df[col].cat.categories:
                all_df[col] = all_df[col].cat.add_categories(["unknown"])
        
        # label2idx 생성
        unique_labels = all_df[col].cat.categories
        label2idx[col] = {label: idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx: label for idx, label in enumerate(unique_labels)}

        # train/test 매핑
        if col in categorical_fillna:
                # train/test 컬럼도 Categorical로 변환
                train_df[col] = train_df[col].astype("category")
                test_df[col] = test_df[col].astype("category")

                # unknown 추가
                if "unknown" not in train_df[col].cat.categories:
                    train_df[col] = train_df[col].cat.add_categories(["unknown"])
                if "unknown" not in test_df[col].cat.categories:
                    test_df[col] = test_df[col].cat.add_categories(["unknown"])

                # fillna 후 매핑
                train_df[col] = train_df[col].fillna("unknown").map(label2idx[col])
                test_df[col] = test_df[col].fillna("unknown").map(label2idx[col])
        else:
            train_df[col] = train_df[col].map(label2idx[col])
            test_df[col] = test_df[col].map(label2idx[col])



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

# -------------------------------
# 6. Data Split / Loader
# -------------------------------
def context_data_split(args, data):
    return basic_data_split(args, data)

def context_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data["X_train"].values),
                                  torch.LongTensor(data["y_train"].values))
    valid_dataset = TensorDataset(torch.LongTensor(data["X_valid"].values),
                                  torch.LongTensor(data["y_valid"].values)) if args.dataset.valid_ratio != 0 else None
    test_dataset = TensorDataset(torch.LongTensor(data["test"].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.dataloader.batch_size,
                                  shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.dataloader.batch_size,
                                  shuffle=False, num_workers=args.dataloader.num_workers) if args.dataset.valid_ratio != 0 else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.dataloader.batch_size,
                                 shuffle=False, num_workers=args.dataloader.num_workers)

    data["train_dataloader"], data["valid_dataloader"], data["test_dataloader"] = train_dataloader, valid_dataloader, test_dataloader

    return data
