# feature_importance.py

from catboost import CatBoostRegressor
import pandas as pd

# ------------------------------------
# 1) 모델 로드
# ------------------------------------
model = CatBoostRegressor()
model.load_model(
    "/data/ephemeral/home/sojin/saved/checkpoint/20251209_233917_CatBoost_fold2.cbm"
)

# ------------------------------------
# 2) 모델에서 feature 이름 자동 추출
# ------------------------------------
features = model.feature_names_
print("📌 모델에 저장된 feature 개수:", len(features))
print("📌 feature 목록:", features)

# ------------------------------------
# 3) importance type별 계산
# ------------------------------------
importance_types = [
    "FeatureImportance",  # split count
    "PredictionValuesChange",  # ⭐ 가장 유용
    # "LossFunctionChange",         # loss 감소 기여
]

importance_results = {}

for imp_type in importance_types:
    importances = model.get_feature_importance(type=imp_type)

    df = pd.DataFrame({"feature": features, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    importance_results[imp_type] = df

# ------------------------------------
# 4) 출력
# ------------------------------------
for t, df in importance_results.items():
    print(f"\n===============================")
    print(f"📌 Feature Importance Type: {t}")
    print("===============================\n")
    print(df)
