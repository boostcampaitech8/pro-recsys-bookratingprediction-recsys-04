import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ============================================================
# Helper: Entropy
# ============================================================
def compute_entropy(series):
    counts = series.value_counts(normalize=True)
    return -(counts * np.log(counts + 1e-12)).sum()


# ============================================================
# Step 1: Location Cluster (user_df 기반)
# ============================================================
def build_user_location_cluster(user_df, n_clusters=20):

    df = user_df.copy()

    # 결측치 0 대체를 위한 frequency encoding
    for col in ["location_country", "location_state", "location_city"]:
        freq = df[col].value_counts().to_dict()
        df[f"{col}_freq"] = df[col].map(freq)

    loc_mat = df[["location_country_freq", "location_state_freq", "location_city_freq"]].fillna(0)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df["location_cluster"] = kmeans.fit_predict(loc_mat)

    return df[["user_id", "location_cluster"]]


# ============================================================
# Step 2: Full_df 기반 유저 행동 feature
# ============================================================
def build_user_behavior_features(full_df):

    df = full_df.copy()

    # -------------------------------
    # 리뷰 수
    # -------------------------------
    user_review_count = (
        df.groupby("user_id")["isbn"].count().rename("user_review_count")
    )

    user_review_count_log = np.log1p(user_review_count)
    user_review_count_log.name = "user_review_count_log"

    user_review_count_log_bin = pd.qcut(
        user_review_count_log,
        q=10,
        labels=False,
        duplicates="drop"
    ).rename("user_review_count_log_bin")

    # -------------------------------
    # 장르 다양성
    # -------------------------------
    user_genre_variety = (
        df.groupby("user_id")["category_cluster"]
          .nunique()
          .rename("user_genre_variety")
    )

    user_top_genre = (
        df.groupby("user_id")["category_cluster"]
          .agg(lambda x: x.value_counts().idxmax())
          .rename("user_top_genre")
    )

    user_genre_entropy = (
        df.groupby("user_id")["category_cluster"]
          .agg(lambda x: compute_entropy(x))
          .rename("user_genre_entropy")
    )

    # -------------------------------
    # 작가 기반 feature
    # -------------------------------
    user_author_variety = (
        df.groupby("user_id")["book_author"]
          .nunique()
          .rename("user_author_variety")
    )

    user_top_author = (
        df.groupby("user_id")["book_author"]
          .agg(lambda x: x.value_counts().idxmax())
          .rename("user_top_author")
    )

    # -------------------------------
    # 출판사 기반 feature
    # -------------------------------
    user_top_publisher = (
        df.groupby("user_id")["publisher"]
          .agg(lambda x: x.value_counts().idxmax())
          .rename("user_top_publisher")
    )

    # -------------------------------
    # 평균 book age
    # -------------------------------
    user_mean_book_age = (
        df.groupby("user_id")["age_of_book"]
          .mean()
          .rename("user_mean_book_age")
    )

    # -------------------------------
    # 병합
    # -------------------------------
    behavior = pd.concat([
        user_review_count,
        user_review_count_log,
        user_review_count_log_bin,
        user_genre_variety,
        user_top_genre,
        user_genre_entropy,
        user_author_variety,
        user_top_author,
        user_top_publisher,
        user_mean_book_age,
    ], axis=1)

    return behavior.reset_index()


# ============================================================
# Step 3: Main user feature build
# ============================================================
def build_user_features(user_df, full_df):

    # 1) location cluster (user_df 기반)
    loc = build_user_location_cluster(user_df)

    # 2) behavior feature (full_df 기반)
    behavior = build_user_behavior_features(full_df)

    # 3) 모든 feature merge
    user_final = (
        user_df.merge(loc, on="user_id", how="left")
               .merge(behavior, on="user_id", how="left")
    )

    return user_final


# ============================================================
# Step 4: Save
# ============================================================
def save_user_features(df, base_path="/data/ephemeral/home/data/features/v4/users"):
    os.makedirs(base_path, exist_ok=True)
    df.to_parquet(f"{base_path}/user_features.parquet", index=False)
    print(f"[INFO] Saved user features → {base_path}/user_features.parquet")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    print("[INFO] Loading data...")

    # (원본 user_df)
    user_df = pd.read_parquet("/data/ephemeral/home/data/features/v1/users/users_features.parquet")

    # (book feature + metadata)
    book_df = pd.read_parquet("/data/ephemeral/home/data/features/v3/books/book_features.parquet")

    # rating
    train_df = pd.read_csv("/data/ephemeral/home/data/train_ratings.csv")

    print("[INFO] Merging to full_df...")
    full_df = (
        train_df
        .merge(user_df, on="user_id", how="left")
        .merge(book_df, on="isbn", how="left")
    )

    print("[INFO] Building user features...")
    user_final = build_user_features(user_df, full_df)

    print("[INFO] Saving...")
    save_user_features(user_final)
