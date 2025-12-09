import torch
import torch.nn as nn
from catboost import CatBoostRegressor


class CatBoost(nn.Module):
    """
    CatBoost 모델을 PyTorch 모듈처럼 래핑한 클래스입니다.

    CatBoost는 범주형 변수 처리에 최적화된 gradient boosting 모델로,
    별도의 전처리 없이 범주형 변수를 직접 다룰 수 있습니다.

    특징:
    - 범주형 변수를 자동으로 처리 (원-핫 인코딩 불필요)
    - Ordered boosting으로 overfitting 방지
    - GPU 학습 지원
    - 결측치 자동 처리
    """

    def __init__(self, args, data, global_seed=None):
        super().__init__()
        self.args = args
        self.cat_features = data["field_names"]

        # global_seed가 전달되지 않으면 args에서 찾고, 그것도 없으면 기본값 42 사용
        random_seed = (
            global_seed
            if global_seed is not None
            else (args.random_seed if hasattr(args, "random_seed") else 42)
        )

        # CatBoost 모델 초기화
        self.model = CatBoostRegressor(
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=random_seed,
            cat_features=self.cat_features,
            verbose=args.verbose if hasattr(args, "verbose") else False,
            task_type=(
                "GPU"
                if args.task_type == "GPU" and torch.cuda.is_available()
                else "CPU"
            ),
            devices=args.devices if hasattr(args, "devices") else "0",
            train_dir="saved/catboost_info",  # catboost_info 디렉토리를 saved 폴더 아래에 생성
        )

        # 학습 여부를 추적하는 플래그
        self.is_fitted = False

    def fit(self, X, y, eval_set=None, **kwargs):
        """
        CatBoost 모델을 학습합니다.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            학습 데이터
        y : np.ndarray or pd.Series
            타겟 값
        eval_set : tuple, optional
            검증 데이터 (X_val, y_val)
        """

        if eval_set is not None:
            self.model.fit(
                X,
                y,
                eval_set=eval_set,
                use_best_model=kwargs["use_best_model"],
                early_stopping_rounds=kwargs[
                    "early_stopping_rounds"
                ],  # 50 iteration 동안 개선 없으면 조기 종료
                verbose=(
                    kwargs["verbose"]
                    if (self.args.verbose if hasattr(self.args, "verbose") else False)
                    else False
                ),  # 100 iteration마다 출력
            )
        else:
            self.model.fit(
                X,
                y,
                verbose=self.args.verbose if hasattr(self.args, "verbose") else False,
            )

        self.is_fitted = True
        return self

    def forward(self, x):
        """
        PyTorch 모델과의 호환성을 위한 forward 메서드입니다.

        Parameters
        ----------
        x : torch.Tensor
            입력 데이터 (batch_size, num_fields)

        Returns
        -------
        predictions : torch.Tensor
            예측 값 (batch_size,)
        """
        # Tensor를 numpy로 변환
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x

        # CatBoost 예측
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before making predictions. Call fit() first."
            )

        predictions = self.model.predict(x_np)

        # numpy를 Tensor로 변환
        predictions = torch.tensor(
            predictions, dtype=torch.float32, device="cuda" if x.is_cuda else "cpu"
        )

        return predictions

    def predict(self, X):
        """
        예측을 수행합니다.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame or torch.Tensor
            입력 데이터

        Returns
        -------
        predictions : np.ndarray
            예측 값
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        return self.model.predict(X)

    def get_feature_importance(self):
        """
        Feature importance를 반환합니다.

        Returns
        -------
        importance : np.ndarray
            각 피처의 중요도
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before getting feature importance."
            )

        return self.model.get_feature_importance()
