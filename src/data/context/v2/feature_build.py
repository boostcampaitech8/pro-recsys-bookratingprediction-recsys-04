import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


# ============================================
# 1. age_of_book 생성
# ============================================
def add_age_of_book(df):
    """
    year_of_publication_clipped 기반으로 책의 나이(age_of_book) 생성
    Numeric Feature
    """
    df["age_of_book"] = 2025 - df["year_of_publication_clipped"]
    return df


# ============================================
# 2. summary embedding & clustering
# ============================================
def build_summary_clusters(df, K=50):
    """
    - 입력 df의 기존 컬럼은 절대 삭제/변형하지 않고 그대로 유지
    - summary 기반으로 summary_cluster, summary_missing_flag 컬럼만 추가
    """

    # 0) 원본 보호
    df = df.copy()

    # 1) 결측 플래그
    df["summary_missing_flag"] = df["summary"].isna().astype(int)

    # 2) summary 있는 행만 마스크
    mask = df["summary"].notna()
    df_nonnull = df.loc[mask, :].copy()
    print(f"[INFO] Summary 존재 비율: {mask.mean():.3f}")

    # summary가 하나도 없을 때: 그냥 no_summary로 통일
    if df_nonnull.empty:
        df["summary_cluster"] = "no_summary"
        return df, None

    # 3) SBERT 로드
    print("[INFO] Loading SBERT model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 4) Embedding (summary 있는 애들만)
    print("[INFO] Encoding summaries...")
    embeddings = model.encode(
        df_nonnull["summary"].tolist(),
        batch_size=64,
        show_progress_bar=True
    )
    embeddings = np.array(embeddings)

    # 5) KMeans
    print(f"[INFO] Running KMeans with K={K}...")
    kmeans = KMeans(n_clusters=K, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # 6) 클러스터 결과를 원래 df의 같은 인덱스에 매핑
    df.loc[mask, "summary_cluster"] = clusters.astype(str)
    df.loc[~mask, "summary_cluster"] = "no_summary"

    return df, embeddings


# ============================================
# 3. 전체 book feature 생성 함수
# ============================================
def build_book_features(book_df):
    print("[INFO] Building book features...")
    book_df = book_df.copy()

    # 1) 나이 feature
    book_df = add_age_of_book(book_df)

    # 2) summary 클러스터링
    book_df, emb = build_summary_clusters(book_df)

    print("[INFO] Book feature building complete!")
    print("[DEBUG] columns:", book_df.columns.tolist())

    return book_df, emb


# ============================================
# 4. 저장 함수
# ============================================
def save_book_features(df, embeddings, base_path="/data/ephemeral/home/data/features/v2/books"):
    os.makedirs(base_path, exist_ok=True)
    print("넣을================ df",df.columns)
    df.to_parquet(f"{base_path}/book_features.parquet", index=False)

    if embeddings is not None:
        np.save(f"{base_path}/summary_embeddings.npy", embeddings)

    print("[INFO] Book features saved at:", base_path)
    
    


# ============================================
# 5. 실행
# ============================================
if __name__ == "__main__":
    books = pd.read_parquet("/data/ephemeral/home/data/features/v1/books/book_features.parquet")

    books_feat, emb = build_book_features(books)
    save_book_features(books_feat, emb)


