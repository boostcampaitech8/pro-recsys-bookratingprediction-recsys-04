from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# GPU 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"

# 데이터 로드
data_path = "/data/ephemeral/home/data/"
books_df = pd.read_csv(data_path + "books.csv")
non_categorized = books_df["category"].isna()
non_categorized_books_df = books_df[non_categorized]


# 배치 처리를 위한 데이터셋 클래스
class BookTitleDataset(Dataset):
    def __init__(self, titles):
        self.titles = titles

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        return str(self.titles[idx])


def predict_genres_batch(titles, batch_size=64):
    # 모델과 토크나이저 로드
    model_name = "BEE-spoke-data/roberta-large-title2genre"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()  # 평가 모드로 설정

    # 데이터셋과 데이터로더 생성
    dataset = BookTitleDataset(titles)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_genres = []

    with torch.no_grad():  # 역전파 비활성화
        for batch in tqdm(dataloader, desc="장르 예측 중"):
            # 배치 토큰화
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=128,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 예측 수행
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

            # 예측 결과를 라벨로 변환
            for pred in predictions:
                all_genres.append(model.config.id2label[pred.item()].split()[-1])

    return all_genres


# 메인 실행 부분
if __name__ == "__main__":
    print("### 장르 예측을 시작합니다 ###\n")

    # 예측할 제목 목록 가져오기
    titles_to_predict = non_categorized_books_df["book_title"].tolist()

    # 배치로 예측 수행
    predicted_genres = predict_genres_batch(titles_to_predict)

    # 원본 df 업데이트
    books_df.loc[non_categorized, "category"] = predicted_genres

    # 결과 저장
    output_path = data_path + "books_with_categories.csv"
    books_df.to_csv(output_path, index=False)

    print(f"\n### 예측이 완료되었습니다. 결과가 {output_path}에 저장되었습니다. ###")
    print("\n### 예시 결과 (상위 10개) ###\n")
    for i, (title, genre) in enumerate(
        zip(titles_to_predict[:10], predicted_genres[:10])
    ):
        print(f"{i+1}. 제목: {title}")
        print(f"   예측 장르: {genre}\n")
