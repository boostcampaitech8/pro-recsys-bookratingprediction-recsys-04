import numpy as np
import pandas as pd
from tqdm import tqdm


def train_catboost(args, model, data, logger, setting):
    """
    CatBoost 모델을 학습합니다.

    Parameters
    ----------
    args : OmegaConf
        학습 설정
    model : CatBoost
        CatBoost 모델 인스턴스
    data : dict
        학습/검증 데이터
    logger : Logger
        로깅 객체
    setting : Setting
        설정 객체

    Returns
    -------
    model : CatBoost
        학습된 모델
    """
    print(f"--------------- {args.model} TRAINING ---------------")

    # DataLoader에서 데이터 추출
    X_train_list, y_train_list = [], []
    for batch in tqdm(data['train_dataloader'], desc='Extracting training data'):
        if isinstance(batch, dict):
            X_train_list.append(batch['user_book_vector'].cpu().numpy())
            y_train_list.append(batch['rating'].cpu().numpy())
        else:
            X_train_list.append(batch[0].cpu().numpy())
            y_train_list.append(batch[1].cpu().numpy())

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    # 검증 데이터가 있는 경우
    eval_set = None
    if args.dataset.valid_ratio != 0 and 'valid_dataloader' in data:
        X_valid_list, y_valid_list = [], []
        for batch in tqdm(data['valid_dataloader'], desc='Extracting validation data'):
            if isinstance(batch, dict):
                X_valid_list.append(batch['user_book_vector'].cpu().numpy())
                y_valid_list.append(batch['rating'].cpu().numpy())
            else:
                X_valid_list.append(batch[0].cpu().numpy())
                y_valid_list.append(batch[1].cpu().numpy())

        X_valid = np.vstack(X_valid_list)
        y_valid = np.concatenate(y_valid_list)
        eval_set = (X_valid, y_valid)

    # CatBoost 학습
    print(f'Training {args.model} model...')
    model.fit(X_train, y_train, eval_set=eval_set)

    # 모델 저장
    if args.train.save_best_model:
        import os
        os.makedirs(args.train.ckpt_dir, exist_ok=True)
        if args.model == 'CatBoost':
            ext = '.cbm'
            model_path = os.path.join(args.train.ckpt_dir, f"{setting.save_time}_{args.model}_best{ext}")
            model.model.save_model(model_path)
        elif args.model == 'XGBoost':
            ext = '.json'
            model_path = os.path.join(args.train.ckpt_dir, f"{setting.save_time}_{args.model}_best{ext}")
            model.model.save_model(model_path)
        else:
            # default behavior
            ext = '.pt'
            model_path = os.path.join(args.train.ckpt_dir, f"{setting.save_time}_{args.model}_best{ext}")
            torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

    return model


def test_catboost(args, model, data, setting, checkpoint_path=None):
    """
    CatBoost 모델로 예측을 수행합니다.

    Parameters
    ----------
    args : OmegaConf
        예측 설정
    model : CatBoost
        CatBoost 모델 인스턴스
    data : dict
        테스트 데이터
    setting : Setting
        설정 객체
    checkpoint_path : str, optional
        불러올 모델 경로

    Returns
    -------
    predicts : np.ndarray
        예측 결과
    """
    # 체크포인트에서 모델 로드
    if checkpoint_path:
        print(f'Loading model from {checkpoint_path}')
        model.model.load_model(checkpoint_path)
        model.is_fitted = True

    # 테스트 데이터 추출
    X_test_list = []
    for batch in tqdm(data['test_dataloader'], desc='Extracting test data'):
        if isinstance(batch, dict):
            X_test_list.append(batch['user_book_vector'].cpu().numpy())
        else:
            X_test_list.append(batch[0].cpu().numpy())

    X_test = np.vstack(X_test_list)

    # 예측
    print('Making predictions...')
    predicts = model.predict(X_test)

    return predicts
