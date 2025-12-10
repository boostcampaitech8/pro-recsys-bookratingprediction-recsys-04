import os
import shutil
import zipfile

# 1) 원본 파일 경로들
file_paths = {
    "books.csv": "/data/ephemeral/home/min/pro-recsys-bookratingprediction-recsys-04/saved/data_share/v3_min_book.csv",
    "users.csv": "/data/ephemeral/home/min/pro-recsys-bookratingprediction-recsys-04/saved/data_share/v3_min_user.csv",
    "test_rating.csv": "/data/ephemeral/home/data/test_ratings.csv",
    "train_rating.csv": "/data/ephemeral/home/data/train_ratings.csv"
}

# 2) 압축 전 임시 폴더 만들기
bundle_dir = "/data/ephemeral/home/min/pro-recsys-bookratingprediction-recsys-04/saved/data_share/data_v3_minyou"
os.makedirs(bundle_dir, exist_ok=True)

# 3) 파일 복사 (이름 변경 포함)
for new_name, src_path in file_paths.items():
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"파일 없음: {src_path}")

    dst_path = os.path.join(bundle_dir, new_name)
    shutil.copy(src_path, dst_path)
    print(f"복사됨: {src_path} → {dst_path}")

# 4) ZIP 파일 생성
zip_path = f"{bundle_dir}.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for filename in os.listdir(bundle_dir):
        file_path = os.path.join(bundle_dir, filename)
        zipf.write(file_path, arcname=filename)

print("\n🎉 ZIP 생성 완료!")
print("ZIP 파일 경로:", zip_path)
