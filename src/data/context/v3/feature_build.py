import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import re


# ============================================================
# 0-0. 텍스트 정규화 함수 (canonical_id 생성용)
# ============================================================
def normalize_text(x):
    """title/author 정규화: 소문자 + 공백정리 + 특수문자 제거"""
    if pd.isna(x):
        return "unknown"
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9 ]", "", x)
    x = re.sub(r"\s+", " ", x)
    return x


# ============================================================
# 0-1. canonical_id 생성
# ============================================================
def add_canonical_id(df):
    """
    canonical_id = normalize(title) + ' | ' + normalize(author)
    누수 없음, 단순 그룹화용 feature
    """
    print("[INFO] Adding canonical_id feature...")

    df["title_norm"] = df["book_title"].apply(normalize_text)
    df["author_norm"] = df["book_author"].apply(normalize_text)

    df["canonical_id"] = df["title_norm"] + " | " + df["author_norm"]

    return df


# ============================================================
# 0-2. Publisher Book Count + Log + Bin
# ============================================================
def add_publisher_book_count(df, n_bins=5):

    print("[INFO] Adding publisher book count features...")

    pub_count = df.groupby("publisher")["isbn"].transform("nunique")
    df["publisher_book_count"] = pub_count
    df["publisher_book_count_log"] = np.log1p(pub_count)

    try:
        df["publisher_book_count_bin"] = pd.qcut(
            pub_count,
            q=n_bins,
            duplicates="drop"
        ).astype(str)
    except:
        df["publisher_book_count_bin"] = "single_bin"

    return df


# ============================================================
# 0-3. Author Book Count + Log + Bin
# ============================================================
def add_author_book_count(df, n_bins=5):

    print("[INFO] Adding author book count features...")

    author_count = df.groupby("book_author")["isbn"].transform("nunique")
    df["author_book_count"] = author_count
    df["author_book_count_log"] = np.log1p(author_count)

    try:
        df["author_book_count_bin"] = pd.qcut(
            author_count,
            q=n_bins,
            duplicates="drop"
        ).astype(str)
    except:
        df["author_book_count_bin"] = "single_bin"

    return df


# ============================================================
# 0-4. Publisher Review Count + Log + Bin
# ============================================================
def add_publisher_review_count(df, train_ratings, n_bins=5):

    print("[INFO] Adding publisher review count features...")

    merged = train_ratings.merge(df[["isbn", "publisher"]], on="isbn", how="left")

    pub_review = merged.groupby("publisher")["rating"].count()
    df["publisher_review_count"] = df["publisher"].map(pub_review).fillna(0)
    df["publisher_review_count_log"] = np.log1p(df["publisher_review_count"])

    try:
        df["publisher_review_count_bin"] = pd.qcut(
            df["publisher_review_count"],
            q=n_bins,
            duplicates="drop"
        ).astype(str)
    except:
        df["publisher_review_count_bin"] = "single_bin"

    return df


# ============================================================
# 0-5. Author Review Count + Log + Bin
# ============================================================
def add_author_review_count(df, train_ratings, n_bins=5):

    print("[INFO] Adding author review count features...")

    merged = train_ratings.merge(df[["isbn", "book_author"]], on="isbn", how="left")

    author_review = merged.groupby("book_author")["rating"].count()
    df["author_review_count"] = df["book_author"].map(author_review).fillna(0)
    df["author_review_count_log"] = np.log1p(df["author_review_count"])

    try:
        df["author_review_count_bin"] = pd.qcut(
            df["author_review_count"],
            q=n_bins,
            duplicates="drop"
        ).astype(str)
    except:
        df["author_review_count_bin"] = "single_bin"

    return df


# ============================================================
# 1. Count Encoding for title
# ============================================================
def add_title_count(df):

    print("[INFO] Adding title_count feature...")

    title_freq = df["book_title"].value_counts()
    df["title_count"] = df["book_title"].map(title_freq)

    return df


# ============================================================
# 2. Title Embedding + KMeans Clustering
# ============================================================
def build_title_clusters(df, K=250):

    print("[INFO] Building title clusters...")

    df["book_title_proc"] = df["book_title"].fillna("")
    df_nonnull = df[df["book_title_proc"] != ""].copy()

    if len(df_nonnull) == 0:
        df["title_cluster"] = "no_title"
        return df, None

    print("[INFO] Loading SBERT model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("[INFO] Encoding titles...")
    embeddings = model.encode(
        df_nonnull["book_title_proc"].tolist(),
        batch_size=64,
        show_progress_bar=True
    )
    embeddings = np.array(embeddings)

    print(f"[INFO] Performing KMeans clustering (K={K})...")
    kmeans = KMeans(n_clusters=K, random_state=42)
    df_nonnull["title_cluster"] = kmeans.fit_predict(embeddings)

    df = df.merge(
        df_nonnull[["isbn", "title_cluster"]],
        on="isbn",
        how="left"
    )

    df["title_cluster"] = df["title_cluster"].fillna("no_title").astype(str)

    return df, embeddings


# ============================================================
# 3. 전체 book features 생성 (canonical_id 포함)
# ============================================================
def build_book_title_features(book_df, train_ratings, K=250):

    df = book_df.copy()

    # ---------- canonical_id 추가 ----------
    df = add_canonical_id(df)

    # ---------- publisher ----------
    df = add_publisher_book_count(df)
    df = add_publisher_review_count(df, train_ratings)

    # ---------- author ----------
    df = add_author_book_count(df)
    df = add_author_review_count(df, train_ratings)

    # ---------- title ----------
    df = add_title_count(df)
    df, title_emb = build_title_clusters(df, K=K)

    return df, title_emb


# ============================================================
# 4. 저장 함수
# ============================================================
def save_title_features(df, embeddings, base_path="/data/ephemeral/home/data/features/v3/books"):

    os.makedirs(base_path, exist_ok=True)

    df.to_parquet(f"{base_path}/book_features.parquet", index=False)

    if embeddings is not None:
        np.save(f"{base_path}/title_embeddings.npy", embeddings)

    print("[INFO] Title + Publisher + Author + canonical_id features saved at:", base_path)


# ============================================================
# 5. 실행 예시
# ============================================================
if __name__ == "__main__":

    books = pd.read_parquet("/data/ephemeral/home/data/features/v2/books/book_features.parquet")
    train = pd.read_csv("/data/ephemeral/home/data/train_ratings.csv")

    books_feat, emb = build_book_title_features(books, train, K=250)
    save_title_features(books_feat, emb)
