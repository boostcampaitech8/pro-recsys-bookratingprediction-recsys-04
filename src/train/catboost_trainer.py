import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

def train_catboost(args, model, data, logger, setting):
    """
    CatBoost 모델을 학습합니다.
    DataLoader를 사용하지 않고 DataFrame을 직접 주입합니다.
    """
    
    # 설정값 읽기 (옵션)
    use_optuna = False
    n_trials = 30
    k_folds = 1
    if hasattr(args, 'model_args') and args.model in args.model_args and 'use_optuna' in args.model_args[args.model]:
        use_optuna = bool(args.model_args[args.model]['use_optuna'])
    if hasattr(args, 'model_args') and args.model in args.model_args and 'optuna_n_trials' in args.model_args[args.model]:
        n_trials = int(args.model_args[args.model]['optuna_n_trials'])
    if hasattr(args, 'model_args') and args.model in args.model_args and 'k_folds' in args.model_args[args.model]:
        k_folds = int(args.model_args[args.model]['k_folds'])
    
    seed = args.seed if hasattr(args, 'seed') else 42

    # 공통 하이퍼파라미터 빌더 (Optuna 사용 여부와 무관하게 사용)
    def make_params(trial=None):
        if trial is None:
            p = dict(
                iterations=getattr(args.model_args[args.model], 'iterations', 5000),
                learning_rate=getattr(args.model_args[args.model], 'learning_rate', 0.03),
                depth=getattr(args.model_args[args.model], 'depth', 6),
                l2_leaf_reg=getattr(args.model_args[args.model], 'l2_leaf_reg', 0.03),
                bagging_temperature=getattr(args.model_args[args.model], 'bagging_temperature', 0.25),
                random_strength=getattr(args.model_args[args.model], 'random_strength', 6),
                border_count=getattr(args.model_args[args.model], 'border_count', 130),
                loss_function='RMSE',
                eval_metric='RMSE',
                random_seed=seed,
                task_type='GPU' if getattr(args.model_args[args.model], 'task_type', 'CPU') == 'GPU' and torch.cuda.is_available() else 'CPU',
                devices=getattr(args.model_args[args.model], 'devices', '0'),
                train_dir='saved/catboost_info',
            )
            return p
        p = dict(
            iterations=trial.suggest_int('iterations', 4500, 8000, step=500),
            learning_rate=trial.suggest_float('learning_rate', 0.02, 0.06, log=True),
            depth=trial.suggest_int('depth', 6, 9),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1e-2, 1.0, log=True),
            bagging_temperature=trial.suggest_float('bagging_temperature', 0.0, 0.5),
            random_strength=trial.suggest_float('random_strength', 5.0, 8.0),
            border_count=trial.suggest_int('border_count', 32, 255),
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=seed,
            task_type='GPU' if getattr(args.model_args[args.model], 'task_type', 'CPU') == 'GPU' and torch.cuda.is_available() else 'CPU',
            devices=getattr(args.model_args[args.model], 'devices', '0'),
            train_dir='saved/catboost_info',
        )
        return p

    # Optuna 또는 K-Fold 사용 시: 전체 train을 기반으로 CV 수행
    if use_optuna or k_folds > 1:
        df = data['train']
        X_all = df.drop('rating', axis=1)
        y_all = df['rating']

        X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, test_size=0.1, random_state=seed)

        def objective(trial):
            params = make_params(trial)
            estimator = CatBoostRegressor(**params, cat_features=getattr(model, 'cat_features', None), verbose=500)
            estimator.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True, early_stopping_rounds=100)
            pred = estimator.predict(X_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, pred))
            return rmse

        best_params = make_params(None)
        if use_optuna:
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials)
            best_params = make_params(study.best_trial)

        print(f'Training {args.model} with K-Fold={k_folds}...')
        print(best_params)
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
        fold_models = []
        oof_pred = np.zeros(len(X_all))
        for tr_idx, va_idx in kf.split(X_all):
            X_tr, X_va = X_all.iloc[tr_idx], X_all.iloc[va_idx]
            y_tr, y_va = y_all.iloc[tr_idx], y_all.iloc[va_idx]
            estimator = CatBoostRegressor(**best_params, cat_features=getattr(model, 'cat_features', None), verbose=True)
            estimator.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True, early_stopping_rounds=100, verbose=500)
            va_pred = estimator.predict(X_va)
            oof_pred[va_idx] = va_pred
            fold_models.append(estimator)

        oof_rmse = float(np.sqrt(mean_squared_error(y_all, oof_pred)))
        print(f'OOF RMSE: {oof_rmse:.6f}')
        pd.DataFrame({
            'rating': y_all,
            'predict': oof_pred
        }).to_csv('oof_pred.csv', index=False)
        setattr(model, 'fold_models', fold_models)
        setattr(model, 'oof_pred', oof_pred)
        setattr(model, 'oof_rmse', oof_rmse)
        setattr(model, 'is_fitted', True)
    else:
        # 1. 학습 데이터 추출 (DataFrame or Series)
        # boost_data_loader에서 반환한 딕셔너리에 이미 DF 형태로 들어있습니다.
        X_train = data['X_train']
        y_train = data['y_train']

        # 2. 검증 데이터 구성
        eval_set = None
        if args.dataset.valid_ratio != 0 and 'X_valid' in data:
            X_valid = data['X_valid']
            y_valid = data['y_valid']
            eval_set = (X_valid, y_valid)

        # 4. 학습 시작
        print(f'Training {args.model} model...')
        
        # 하이퍼파라미터 적용: CatBoost는 생성자 파라미터로 설정해야 하는 항목들이 있어
        # fit 전에 모델(혹은 내부 모델)을 재구성합니다.
        params = make_params(None)
        cat_features = getattr(model, 'cat_features', None)
        # Wrapper가 내부에 model(CatBoostRegressor)을 보유한 경우
        if hasattr(model, 'model') and isinstance(model.model, CatBoostRegressor):
            model.model = CatBoostRegressor(**params, cat_features=cat_features, verbose=100)
        # Wrapper가 아니라 바로 CatBoostRegressor인 경우
        elif isinstance(model, CatBoostRegressor):
            model = CatBoostRegressor(**params, cat_features=cat_features, verbose=100)
        # 기타 케이스: set_params를 지원하면 설정 시도
        elif hasattr(model, 'set_params') and callable(getattr(model, 'set_params')):
            try:
                model.set_params(**params)
            except Exception:
                pass
        
        # 모델 wrapper의 fit 메소드가 kwargs를 받아 CatBoost fit으로 넘겨준다고 가정합니다.
        # 만약 wrapper가 없다면 model.fit(...)에 직접 cat_features를 넣습니다.
        model.fit(
            X_train, 
            y_train, 
            eval_set=eval_set, 
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=100
        )

    # 5. 모델 저장
    if args.train.save_best_model:
        os.makedirs(args.train.ckpt_dir, exist_ok=True)
        
        if args.model == 'CatBoost':
            ext = '.cbm'
            # Fold 모델이 존재하면 각 fold를 저장
            if hasattr(model, 'fold_models'):
                for idx, fm in enumerate(model.fold_models):
                    model_path = os.path.join(args.train.ckpt_dir, f"{setting.save_time}_{args.model}_fold{idx}{ext}")
                    fm.save_model(model_path)
            else:
                model_path = os.path.join(args.train.ckpt_dir, f"{setting.save_time}_{args.model}_best{ext}")
                # Wrapper 클래스를 사용하는 경우 model.model 접근
                if hasattr(model, 'model'):
                    model.model.save_model(model_path)
                else: # Wrapper가 아닌 원본 객체인 경우
                    model.save_model(model_path)
                
        elif args.model == 'XGBoost':
            ext = '.json'
            model_path = os.path.join(args.train.ckpt_dir, f"{setting.save_time}_{args.model}_best{ext}")
            if hasattr(model, 'model'):
                model.model.save_model(model_path)
            else:
                model.save_model(model_path)
                
        else:
            # PyTorch 모델 등
            ext = '.pt'
            model_path = os.path.join(args.train.ckpt_dir, f"{setting.save_time}_{args.model}_best{ext}")
            torch.save(model.state_dict(), model_path)
            
        print(f'Model saved to {model_path}')

    return model


def test_catboost(args, model, data, setting, checkpoint_path=None):
    """
    CatBoost 모델로 예측을 수행합니다.
    DataLoader 반복 없이 DataFrame 전체를 한번에 예측합니다.
    """
    
    # 1. 모델 로드
    if checkpoint_path:
        print(f'Loading model from {checkpoint_path}')
        # Wrapper 처리
        if hasattr(model, 'model'):
            model.model.load_model(checkpoint_path)
        else:
            model.load_model(checkpoint_path)
        
        # 모델 wrapper에 is_fitted 플래그가 있다면 설정
        if hasattr(model, 'is_fitted'):
            model.is_fitted = True

    # 2. 테스트 데이터 준비
    # data['test']는 이미 DataFrame 형태입니다.
    X_test = data['test']

    # 3. 예측 수행
    print('Making predictions...')
    # Fold 모델이 존재하면 앙상블(평균)
    if hasattr(model, 'fold_models'):
        fold_preds = []
        for fm in model.fold_models:
            fold_preds.append(fm.predict(X_test))
        predicts = np.mean(np.vstack(fold_preds), axis=0)
    else:
        # DataFrame을 통째로 넣으면 CatBoost가 내부적으로 멀티스레딩을 활용해 매우 빠르게 처리합니다.
        predicts = model.predict(X_test)

    return predicts