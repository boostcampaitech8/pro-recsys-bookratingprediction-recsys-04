import os
import re
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


# ============================================================
# 1. Title2Genre 모델 로드
# ============================================================

def load_title2genre_model(model_name="BEE-spoke-data/roberta-large-title2genre"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    return tokenizer, model, device


# ============================================================
# 2. Category 결측치 채우기
# ============================================================

def fill_category_with_title2genre(book_df, tokenizer, model, device):
    df = book_df.copy()

    # 원래 결측치 flag
    df["category_missing_flag"] = df["category"].isna().astype(int)

    mask = df["category"].isna() | (df["category"] == "")
    df_missing = df[mask]

    if len(df_missing) == 0:
        print("[INFO] No missing categories. Skipping prediction.")
        return df

    titles = df_missing["book_title"].tolist()
    preds = []

    batch_size = 16
    for i in tqdm(range(0, len(titles), batch_size)):
        batch = titles[i:i + batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_ids = torch.argmax(probs, dim=1).cpu().tolist()

        pred_labels = [model.config.id2label[i] for i in pred_ids]
        preds.extend(pred_labels)

    # 결측치 채우기
    df.loc[mask, "category"] = preds

    print("[INFO] Category missing values filled.")
    return df


# ============================================================
# 3. Category 텍스트 정규화
# ============================================================

def normalize_category_text(x):
    if pd.isna(x):
        return "unknown"

    x = str(x).strip()

    # 리스트 문자열 처리
    if x.startswith("[") and x.endswith("]"):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return " ".join([str(p) for p in parsed])
        except:
            pass

    return x


def preprocess_category_text(df, col="category", min_len=3):
    df[col] = df[col].fillna("unknown").astype(str)

    def clean_cat(x):
        x = x.lower().strip()

        parts = re.split(r"[\/;:\|\>,\-\_\&\s]+", x)
        parts = [p for p in parts if p != ""]

        # 짧은 토큰 처리
        parts = [p if len(p) >= min_len else "other" for p in parts]

        return " ".join(parts) if len(parts) > 0 else "unknown"

    df[col] = df[col].apply(clean_cat)
    return df


# ============================================================
# 4. SBERT Embedding + Clustering
# ============================================================

def embed_category(df, text_col="category_text", model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = df[text_col].tolist()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    return embeddings


def cluster_categories(embeddings, n_clusters=200):
    km = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_ids = km.fit_predict(embeddings)
    return cluster_ids


# ============================================================
# 5. 전체 파이프라인
# ============================================================

def build_book_category_features(book_df, n_clusters=200):
    df = book_df.copy()

    # -------------------------------------
    # Step 1. Title2Genre 기반 결측치 채우기
    # -------------------------------------
    tokenizer, t2g_model, device = load_title2genre_model()
    df = fill_category_with_title2genre(df, tokenizer, t2g_model, device)

    # -------------------------------------
    # Step 2. 텍스트 정규화
    # -------------------------------------
    df["category"] = df["category"].apply(normalize_category_text)
    df = preprocess_category_text(df, col="category")

    # -------------------------------------
    # Step 3. SBERT embedding
    # -------------------------------------
    df["category_text"] = df["category"].apply(lambda x: f"Category: {x}")
    embeddings = embed_category(df, text_col="category_text")

    # -------------------------------------
    # Step 4. Clustering
    # -------------------------------------
    df["category_cluster"] = cluster_categories(embeddings, n_clusters=n_clusters)

    # 임시 컬럼 제거
    df = df.drop(columns=["category_text"])

    return df, embeddings


# ============================================================
# 6. 저장 함수
# ============================================================

def save_book_features(df, embeddings, base_path="/data/ephemeral/home/data/features/v1/books"):
    os.makedirs(base_path, exist_ok=True)

    df.to_parquet(f"{base_path}/book_features.parquet", index=False)
    np.save(f"{base_path}/category_embeddings.npy", embeddings)

    print("[INFO] Book features saved at:", base_path)



if __name__ == "__main__":
    books = pd.read_parquet("/data/ephemeral/home/data/processed/v1/books/books.parquet")
    books_feat, emb = build_book_category_features(books)
    save_book_features(books_feat, emb)

