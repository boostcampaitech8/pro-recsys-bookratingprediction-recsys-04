import torch
import torch.nn as nn
from catboost import CatBoostRegressor


class CatBoost(nn.Module):
    """
    CatBoost 모델을 PyTorch 모듈처럼 래핑한 클래스.
    - PyTorch와 구조 통일성 유지
    - 실제 학습/예측은 CatBoostRegressor가 수행
    """

    def __init__(self, args, data, global_seed=None):
        super().__init__()

        self.args = args
        self.categorical_features = data["categorical_features"]
        self.all_features = data["all_features"]

        # Seed
        seed = (
            global_seed if global_seed is not None
            else getattr(args, "random_seed", 42)
        )

        # CatBoost model instance
        self.model = CatBoostRegressor(
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=seed,
            verbose=args.verbose if hasattr(args, "verbose") else False,
            task_type="GPU" if args.task_type == "GPU" else "CPU",
            devices=args.devices if hasattr(args, "devices") else "0",
            train_dir="saved/catboost_info",
        )

        self.is_fitted = False

    # ------------------------
    # FIT
    # ------------------------
    def fit(self, X, y, eval_set=None):
        """
        CatBoost 모델 학습
        """

        # CatBoostRegressor에 직접 전달
        self.model.fit(
            X,
            y,
            cat_features=self.categorical_features,
            eval_set=eval_set,
            use_best_model=True if eval_set is not None else False,
            early_stopping_rounds=50 if eval_set is not None else None,
            verbose=100
        )

        self.is_fitted = True

    # ------------------------
    # PREDICT
    # ------------------------
    def predict(self, X):
        """
        CatBoost 예측
        """
        return self.model.predict(X)

    # ------------------------
    # FORWARD (PyTorch 호환)
    # ------------------------
    def forward(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        preds = self.model.predict(X)
        return torch.tensor(preds, dtype=torch.float32)

    # ------------------------
    # SAVE / LOAD
    # ------------------------
    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)
        self.is_fitted = True

    # ------------------------
    # FEATURE IMPORTANCE
    # ------------------------
    def get_feature_importance(self):
        return self.model.get_feature_importance()
