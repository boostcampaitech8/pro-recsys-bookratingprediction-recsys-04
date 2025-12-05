import torch
import torch.nn as nn
from xgboost import XGBRegressor


class XGBoost(nn.Module):
    def __init__(self, args, data, global_seed=None):
        super().__init__()
        self.args = args
        self.field_dims = data['field_dims']
        random_seed = global_seed if global_seed is not None else (getattr(args, 'random_seed', None) or 42)
        use_gpu = getattr(args, 'task_type', 'CPU') == 'GPU' and torch.cuda.is_available()
        self.model = XGBRegressor(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=getattr(args, 'subsample', 1.0),
            colsample_bytree=getattr(args, 'colsample_bytree', 1.0),
            reg_lambda=getattr(args, 'reg_lambda', 1.0),
            reg_alpha=getattr(args, 'reg_alpha', 0.0),
            random_state=random_seed,
            tree_method='hist',
            predictor='gpu_predictor' if use_gpu else 'auto',
            verbosity=getattr(args, 'verbosity', 1),
            objective='reg:squarederror',
            eval_metric='rmse',
            n_jobs=getattr(args, 'n_jobs', 0),
        )
        self.is_fitted = False

    def fit(self, X, y, eval_set=None):
        evals = [eval_set] if eval_set is not None else None
        self.model.fit(X, y, eval_set=evals, verbose=getattr(self.args, 'verbose', False))
        self.is_fitted = True
        return self

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
        else:
            x_np = x
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions. Call fit() first.")
        preds = self.model.predict(x_np)
        return torch.tensor(preds, dtype=torch.float32, device='cuda' if (isinstance(x, torch.Tensor) and x.is_cuda) else 'cpu')

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        return self.model.predict(X)
