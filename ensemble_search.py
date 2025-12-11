import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.optimize import minimize

# =================================================================
# 1. 설정: OOF 파일들의 경로를 리스트에 넣으세요
# =================================================================

OOF_FILE_LIST = [
    "../saved/submit/20251211_032012_CatBoost_kfold_5_OOF.csv",
    "../saved/submit/20251211_043128_MF_kfold_5_OOF.csv",
    "../saved/submit/20251211_082318_DeepFM_kfold_5_OOF.csv",
    # 여기에 계속 추가 가능...
]

TRUE_LABEL_COL = "rating"  # 정답 컬럼 (모든 파일 공통)
PRED_COL = "predict"  # 예측값 컬럼 (모든 파일 공통)

# =================================================================
# 2. 데이터 로드 및 정합성 체크
# =================================================================


def load_oof_predictions(file_paths):
    preds_matrix = []
    y_true = None

    print(f"📂 총 {len(file_paths)}개의 OOF 파일을 로드합니다...")

    for i, path in enumerate(file_paths):
        try:
            df = pd.read_csv(path)

            # ======================
            # 🔍 [추가] OOF 검증 코드
            # ======================
            print(f"\n=== 체크: {path} ===")

            # 1. row 수 출력
            print(" - Row 수:", len(df))

            # 2. predict 컬럼이 존재하는지 확인
            if PRED_COL not in df.columns:
                print(f" ❌ ERROR: '{PRED_COL}' 컬럼이 존재하지 않습니다!")
                return None, None

            # 3. predict 값의 variety 확인
            print(" - predict unique 개수:", df[PRED_COL].nunique())
            if df[PRED_COL].nunique() == 1:
                print(" ⚠️ Warning: predict 값이 전부 동일합니다. (고장 가능성)")

            # 4. RMSE sanity check
            tmp_rmse = sqrt(mean_squared_error(df[TRUE_LABEL_COL], df[PRED_COL]))
            print(f" - 개별 RMSE (sanity): {tmp_rmse:.5f}")

            # ======================
            # 원래 로직 진행
            # ======================

            if y_true is None:
                y_true = df[TRUE_LABEL_COL].values
            else:
                current_true = df[TRUE_LABEL_COL].values
                if not np.allclose(y_true, current_true, rtol=1e-5):
                    print(
                        f"⚠️ Warning: {path}의 정답 라벨 순서가 첫 번째 파일과 다릅니다!"
                    )
                    print("   (정렬 mismatch 가능성 매우 높음)")

            preds_matrix.append(df[PRED_COL].values)

            rmse = sqrt(mean_squared_error(y_true, df[PRED_COL].values))
            print(f"   [{i}] 파일 RMSE: {rmse:.5f}")

        except Exception as e:
            print(f"❌ Error loading {path}: {e}")
            return None, None

    return np.array(preds_matrix).T, y_true


# =================================================================
# 3. 최적화 함수 (RMSE 최소화)
# =================================================================


def calculate_rmse(weights, predictions, y_true):
    # 가중치 적용: (Predictions * Weights)의 합
    final_pred = np.dot(predictions, weights)
    mse = mean_squared_error(y_true, final_pred)
    return sqrt(mse)


def optimize_weights(predictions, y_true):
    num_models = predictions.shape[1]

    # 초기 가중치: 모두 1/N 로 시작 (균등 배분)
    initial_weights = [1.0 / num_models] * num_models

    # 제약 조건 (Constraints)
    # 1. 가중치의 합은 1이어야 한다.
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    # 범위 (Bounds)
    # 2. 각 가중치는 0보다 크고 1보다 작아야 한다. (음수 가중치 방지)
    bounds = [(0.0, 1.0)] * num_models

    print("\n⚖️ 최적의 가중치를 찾는 중입니다... (SLSQP Optimization)")

    result = minimize(
        fun=calculate_rmse,
        x0=initial_weights,
        args=(predictions, y_true),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        tol=1e-6,
    )

    return result


# =================================================================
# 4. 실행
# =================================================================


def main():
    # 데이터 로드
    X_preds, y_true = load_oof_predictions(OOF_FILE_LIST)

    if X_preds is None:
        print("데이터 로드 실패.")
        return

    # 모델 간 상관관계 확인 (다양성 체크)
    print("\n📊 모델 예측값 간 상관관계 (낮을수록 앙상블 효과 좋음):")
    df_corr = pd.DataFrame(
        X_preds, columns=[f"Model_{i}" for i in range(X_preds.shape[1])]
    )
    print(df_corr.corr())

    # 최적화 실행
    opt_res = optimize_weights(X_preds, y_true)

    if opt_res.success:
        best_weights = opt_res.x
        best_rmse = opt_res.fun

        print("\n" + "=" * 50)
        print(f"🎉 최적화 완료! Best RMSE: {best_rmse:.5f}")
        print("=" * 50)

        print("\n🏆 최적 가중치 조합:")
        for i, w in enumerate(best_weights):
            file_name = OOF_FILE_LIST[i].split("/")[-1]
            print(f"   Model {i} ({file_name}): {w:.4f} ({w*100:.1f}%)")

        print("\n🚀 팁: 이 가중치를 사용해 최종 제출 파일을 만드세요.")

    else:
        print("❌ 최적화 실패:", opt_res.message)


if __name__ == "__main__":
    main()
