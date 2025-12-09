import pandas as pd
import numpy as np
import regex
import os

# ----------------------
# Helper functions
# ----------------------

def str2list(x: str) -> list:
    if pd.isna(x):
        return []
    return x[1:-1].split(', ')


def split_location(x: str) -> list:
    if pd.isna(x):
        return [np.nan, np.nan, np.nan]

    res = x.split(',')
    res = [i.strip().lower() for i in res]
    res = [regex.sub(r'[^a-zA-Z/ ]', '', i) for i in res]  
    res = [i if i not in ['n/a', ''] else np.nan for i in res]
    res.reverse()

    # remove duplicated values
    for i in range(len(res)-1, 0, -1):
        if (res[i] in res[:i]) and (not pd.isna(res[i])):
            res.pop(i)

    # ensure at least 3 slots: country/state/city
    while len(res) < 3:
        res.append(np.nan)

    return res[:3]


# ----------------------
# User preprocessing
# ----------------------

def preprocess_users(users):
    df = users.copy()

    # baseline age 처리(mode -> median)
    df["age"] = df["age"].fillna(df["age"].median())
    df["age_group"] = df["age"].apply(lambda x: x // 10 * 10)

    # location split
    df["location_list"] = df["location"].apply(split_location)
    df["location_country"] = df["location_list"].apply(lambda x: x[0])
    df["location_state"] = df["location_list"].apply(lambda x: x[1])
    df["location_city"] = df["location_list"].apply(lambda x: x[2])

    # -----------------------------------------
    # 🔥 1) state는 있는데 country 없는 경우 → mode country로 보정
    # -----------------------------------------
    mask1 = df["location_state"].notna() & df["location_country"].isna()

    fill_country_map = (
        df.groupby("location_state")["location_country"]
          .agg(lambda x: x.mode()[0] if len(x.mode()) else np.nan)
    )

    df.loc[mask1, "location_country"] = df.loc[mask1, "location_state"].map(fill_country_map)

    # -----------------------------------------
    # 🔥 2) city는 있는데 state 없는 경우
    # -----------------------------------------
    mask2 = df["location_city"].notna() & df["location_state"].isna()

    # country가 있는 경우에는 country 기준 city→state 채우기
    mask2a = mask2 & df["location_country"].notna()

    fill_state_map = (
        df.groupby(["location_country", "location_city"])["location_state"]
          .agg(lambda x: x.mode()[0] if len(x.mode()) else np.nan)
    )

    df.loc[mask2a, "location_state"] = df.loc[mask2a].set_index(
        ["location_country", "location_city"]
    ).index.map(fill_state_map)

    # country도 없는 경우 city 기준 전체 mode로 보정
    mask2b = mask2 & df["location_country"].isna()

    fill_state_city = (
        df.groupby("location_city")["location_state"]
          .agg(lambda x: x.mode()[0] if len(x.mode()) else np.nan)
    )

    fill_country_city = (
        df.groupby("location_city")["location_country"]
          .agg(lambda x: x.mode()[0] if len(x.mode()) else np.nan)
    )

    df.loc[mask2b, "location_state"]  = df.loc[mask2b, "location_city"].map(fill_state_city)
    df.loc[mask2b, "location_country"] = df.loc[mask2b, "location_city"].map(fill_country_city)

    # -----------------------------------------
    # 🔥 깔끔 마무리
    # -----------------------------------------
    df = df.drop(columns=["location"])

    return df



# ----------------------
# Book preprocessing
# ----------------------

def preprocess_books(books):
    df = books.copy()

    # df["category"] = df["category"].apply(lambda x: str2list(x)[0] if not pd.isna(x) else np.nan)
    df["language"] = df["language"].fillna(df["language"].mode()[0])
    # df["publication_range"] = df["year_of_publication"].apply(lambda x: x // 10 * 10) -> baseline 방식.
    # -------------------------------
    # 1) 이상치 clip 처리 (IQR)
    # -------------------------------
    Q1 = df["year_of_publication"].quantile(0.25)
    Q3 = df["year_of_publication"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df["year_of_publication_clipped"] = df["year_of_publication"].clip(
        lower=lower_bound, 
        upper=upper_bound
    )

    # -------------------------------
    # 2) 결측치 처리
    # -------------------------------
    mode_year = df["year_of_publication_clipped"].mode()[0]
    df["year_of_publication_clipped"] = df["year_of_publication_clipped"].fillna(mode_year)

    return df


# ----------------------
# Save processed data
# ----------------------

def save_processed(users_df, books_df, base_path="/data/ephemeral/home/data/processed/v1"):
    os.makedirs(f"{base_path}/users", exist_ok=True)
    os.makedirs(f"{base_path}/books", exist_ok=True)

    users_df.to_parquet(f"{base_path}/users/users.parquet", index=False)
    books_df.to_parquet(f"{base_path}/books/books.parquet", index=False)

    print("[INFO] Processed data saved successfully.")


# ----------------------
# Main pipeline
# ----------------------

def run_preprocessing(user_path = "/data/ephemeral/home/data/users.csv", book_path = "/data/ephemeral/home/data/books.csv"):
    users = pd.read_csv(user_path)
    books = pd.read_csv(book_path)

    users_proc = preprocess_users(users)
    books_proc = preprocess_books(books)

    save_processed(users_proc, books_proc)

    return 


run_preprocessing()