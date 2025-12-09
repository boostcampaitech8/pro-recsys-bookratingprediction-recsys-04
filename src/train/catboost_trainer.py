import numpy as np
import pandas as pd
from tqdm import tqdm


def train_catboost(args, model, data, logger, setting):
    """
    CatBoost 전용 training 함수
    - DataLoader 사용하지 않음
    - DataFrame 그대로 사용
    - categorical_features를 모델에 전달
    """
    print(f'--------------- CatBoost TRAINING ---------------')

    X_train = data["X_train"]
    y_train = data["y_train"]

    eval_set = None
    if args.dataset.valid_ratio != 0:
        eval_set = (data["X_valid"], data["y_valid"])

    categorical = data["categorical_features"]

    print("Training CatBoost model...")

    model.model.fit(
        X_train,
        y_train,
        cat_features=data["categorical_features"],
        eval_set=eval_set,
        verbose=100
    )

    # ---------------------------
    # 모델 저장
    # ---------------------------
    if args.train.save_best_model:
        import os
        os.makedirs(args.train.ckpt_dir, exist_ok=True)
        model_path = os.path.join(
            args.train.ckpt_dir, f"{setting.save_time}_{args.model}_best.cbm"
        )
        model.save_model(model_path)
        print(f"Model saved to {model_path}")

    return model



def test_catboost(args, model, data, setting, checkpoint_path=None):
    """
    CatBoost 전용 inference
    - DataFrame 그대로 입력
    """

    # 1) checkpoint load
    if checkpoint_path:
        print(f"Loading model from {checkpoint_path}")
        model.load_model(checkpoint_path)

    X_test = data["test_df"]
    all_features = data["all_features"]

    print("Making predictions...")
    predicts = model.predict(X_test[all_features])

    return predicts
