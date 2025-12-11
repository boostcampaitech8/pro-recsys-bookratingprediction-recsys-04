import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import os
import glob


def load_data(args):
    """
    저장된 OOF 파일과 Test 예측 파일을 불러와서 병합합니다.
    """
    submit_path = args.submit_path

    # 정답지(Target) 로드 (train_ratings.csv)
    train_df = pd.read_csv(os.path.join(args.data_path, "train_ratings.csv"))
    y_train = train_df["rating"].values

    # 메타 모델을 위한 데이터셋 초기화
    X_train_meta = []
    X_test_meta = []

    model_names = []

    print(f"Loading Models from {submit_path}...")

    # 입력받은 모델 이름(키워드)별로 최신 파일 찾기
    for model_name in args.models:
        # 1. OOF 파일 찾기 (*model_name*OOF.csv)
        oof_pattern = os.path.join(submit_path, f"*{model_name}*OOF.csv")
        oof_files = sorted(glob.glob(oof_pattern))

        # 2. Test 예측 파일 찾기 (*model_name*kfold*.csv)
        test_pattern = os.path.join(submit_path, f"*{model_name}*kfold*.csv")
        test_files_raw = sorted(glob.glob(test_pattern))

        # ⬇️ [핵심 수정]: 파일명에 '_OOF'가 포함되지 않은 파일만 필터링합니다. (순수한 Test 파일만 선택)
        test_files = [f for f in test_files_raw if "_OOF" not in os.path.basename(f)]
        # -------------------------------------------------------------------------------------

        if not oof_files or not test_files:
            print(
                f"[Warning] '{model_name}'에 대한 OOF 파일 또는 순수한 Test 파일을 찾을 수 없어 건너뜁니다."
            )
            continue

        # 가장 최근 파일 선택 (타임스탬프 기준)
        oof_file = oof_files[-1]
        test_file = test_files[-1]  # 필터링된 파일 중 가장 최신 파일

        print(f" -> [{model_name}] OOF: {os.path.basename(oof_file)}")
        print(f" -> [{model_name}] Test: {os.path.basename(test_file)}")

        # 데이터 로드
        oof_df = pd.read_csv(oof_file)
        test_df = pd.read_csv(test_file)

        # 예측값 추출 (순서가 섞이지 않았다고 가정)
        X_train_meta.append(oof_df["predict"].values)
        X_test_meta.append(test_df["rating"].values)
        model_names.append(model_name)

    if not X_train_meta:
        raise ValueError("병합할 모델 데이터가 없습니다.")

    # (N_samples, N_models) 형태로 변환
    X_train_meta = np.vstack(X_train_meta).T
    X_test_meta = np.vstack(X_test_meta).T

    return X_train_meta, y_train, X_test_meta, model_names


def main(args):
    # 1. 데이터 준비
    X_train, y_train, X_test, model_names = load_data(args)

    print(f"\nConstructed Meta-Dataset: Train {X_train.shape}, Test {X_test.shape}")

    # 2. 메타 모델 학습 (Linear Regression)
    # RMSE를 최소화하는 방향으로 가중치(계수)를 학습합니다.
    meta_model = LinearRegression(fit_intercept=False)

    meta_model.fit(X_train, y_train)

    # 3. 학습 결과 확인 (가중치)
    print("\n[Meta-Model Weights (Contribution)]")
    for name, weight in zip(model_names, meta_model.coef_):
        print(f" -> {name}: {weight:.4f}")

    # Training RMSE 확인
    train_pred = meta_model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    print(f"\n[Stacking Result] Train OOF RMSE: {train_rmse:.4f}")

    # 4. 최종 Test 예측
    final_pred = meta_model.predict(X_test)

    # 5. 결과 저장
    save_path = os.path.join(args.submit_path, "submission_stacking.csv")

    # 샘플 제출 파일 형식을 빌려옴
    sample_sub = pd.read_csv(os.path.join(args.data_path, "sample_submission.csv"))

    # ⬇️ [최종 점검] 예측값의 길이를 확인합니다.
    if len(sample_sub) != len(final_pred):
        print(
            f"\n[Error Check] Final prediction length ({len(final_pred)}) does not match sample submission length ({len(sample_sub)})."
        )
        print("Final prediction size will not match the submission file.")
        raise ValueError("Length mismatch during final submission creation.")

    sample_sub["rating"] = final_pred

    sample_sub.to_csv(save_path, index=False)
    print(f"\nFinal Stacking Submission Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="스태킹할 모델들의 이름 키워드 (예: Image_DeepFM NCF)",
    )
    parser.add_argument(
        "--submit_path",
        type=str,
        default="../saved/submit",
        help="OOF와 예측 파일이 있는 경로",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data",
        help="원본 데이터(train_ratings.csv)가 있는 경로",
    )

    args = parser.parse_args()
    main(args)
