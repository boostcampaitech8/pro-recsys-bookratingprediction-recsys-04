import argparse
import ast
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.optim as optimizer_module
import torch.optim.lr_scheduler as scheduler_module
from src.utils import Logger, Setting
import src.data as data_module
from src.train import train, test
import src.models as model_module
import numpy as np
from sklearn.model_selection import KFold


def main(args, wandb=None):

    Setting.seed_everything(args.seed)

    ######################## LOAD DATA
    datatype = args.model_args[args.model].datatype
    data_load_fn = getattr(data_module, f"{datatype}_data_load")
    data_loader_fn = getattr(data_module, f"{datatype}_data_loader")
    data_split_fn = getattr(data_module, f"{datatype}_data_split")

    print(f"--------------- {args.model} Load Data ---------------")
    # 데이터 로드는 공통
    original_data = data_load_fn(args)

    ######################## MODE SELECTION
    # args 객체에 'kfold' 키가 없을 경우를 대비하여 None을 기본값으로 사용
    kfold_splits = getattr(args, "kfold", None)

    if kfold_splits is not None and kfold_splits > 1:

        # ==========================================
        # [MODE 1] K-Fold Ensemble Training Strategy
        # ==========================================
        n_splits = kfold_splits
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

        # Soft Voting을 위한 Test Set 예측값 리스트
        fold_predicts_list = []

        # [NEW] OOF 예측값을 담을 배열 초기화 (Train 데이터 길이만큼 0으로 채움)
        train_df = original_data["train"]
        oof_predictions = np.zeros(len(train_df))

        print(f">>> K-FOLD MODE ENABLED: {n_splits} Folds <<<")

        # 학습 모드일 때만 K-Fold Loop 진행
        if not args.predict:

            for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
                print(
                    f"\n\n=============== FOLD {fold_idx+1}/{n_splits} ==============="
                )

                # 1. Fold Split
                fold_train_data = train_df.iloc[train_idx]
                fold_valid_data = train_df.iloc[valid_idx]

                # 2. 메타데이터 포함하여 딕셔너리 구성 (original_data를 복사해서 시작)
                target = "rating"
                input_data = original_data.copy()

                # 3. K-Fold로 나눈 데이터로 train/valid 키를 덮어쓰거나 추가 (X, y 분리)
                input_data["X_train"] = fold_train_data.drop(columns=[target])
                input_data["y_train"] = fold_train_data[target]
                input_data["X_valid"] = fold_valid_data.drop(columns=[target])
                input_data["y_valid"] = fold_valid_data[target]

                # 4. DataLoader 생성
                # current_data: {'train': loader, 'valid': loader, 'test': loader ...}
                current_data = data_loader_fn(args, input_data)

                # 5. [안전장치] 모델 초기화에 필요한 메타데이터 복구
                for key in original_data:
                    if key not in current_data and key not in ["train"]:
                        current_data[key] = original_data[key]

                # 6. Logger Setup
                setting = Setting()
                log_path = setting.get_log_path(args)
                logger = Logger(args, log_path)
                logger.save_args()

                # 7. Model Init
                print(
                    f"--------------- INIT {args.model} (Fold {fold_idx+1}) ---------------"
                )
                model = getattr(model_module, args.model)(
                    args.model_args[args.model], current_data
                ).to(args.device)

                if args.train.resume:
                    model.load_state_dict(
                        torch.load(args.train.resume_path, weights_only=True)
                    )

                # 8. Train
                print(
                    f"--------------- {args.model} TRAINING (Fold {fold_idx+1}) ---------------"
                )
                model = train(args, model, current_data, logger, setting)

                # =========================================================
                # [NEW] OOF Prediction Logic (Valid Set 예측)
                # =========================================================
                print(
                    f"--------------- OOF PREDICT (Fold {fold_idx+1}) ---------------"
                )

                # 1) 기존 Test Loader를 잠시 백업 (키 이름 수정: 'test' -> 'test_dataloader')
                real_test_loader = current_data["test_dataloader"]

                # 2) Valid Loader를 Test 키에 할당 (키 이름 수정: 'valid' -> 'valid_dataloader' / 'test' -> 'test_dataloader')
                current_data["test_dataloader"] = current_data["valid_dataloader"]

                # 3) test 함수로 Valid Set 예측 수행 (test 함수가 'test_dataloader'를 사용한다고 가정)
                # [주의: test 함수가 어떤 키를 쓰는지에 따라 인자를 조정해야 할 수도 있음]
                valid_preds = test(args, model, current_data, setting)

                # 4) Test Loader 원상복구
                current_data["test_dataloader"] = real_test_loader

                # 5) OOF 배열에 예측값 저장
                oof_predictions[valid_idx] = valid_preds
                print(
                    f" -> Fold {fold_idx+1} OOF predictions saved. (Size: {len(valid_preds)})"
                )

                # =========================================================
                # 9. Predict (Inference on Real Test Set)
                # =========================================================
                print(
                    f"--------------- {args.model} PREDICT (Fold {fold_idx+1}) ---------------"
                )
                # 진짜 Test Set 예측
                predicts = test(args, model, current_data, setting)
                fold_predicts_list.append(predicts)

            # -----------------------------------------------------------
            # Loop 종료 후: Soft Voting & OOF Save
            # -----------------------------------------------------------

            # 1) Soft Voting (Test Set)
            print(
                f"--------------- Soft Voting Ensemble ({n_splits} Folds) ---------------"
            )
            avg_predicts = np.mean(fold_predicts_list, axis=0)

            # 2) Save Final Submission
            print(f"--------------- SAVE {args.model} ENSEMBLE PREDICT ---------------")
            submission = pd.read_csv(args.dataset.data_path + "sample_submission.csv")
            submission["rating"] = avg_predicts

            filename = setting.get_submit_filename(args)
            filename = filename.replace(".csv", f"_kfold_{n_splits}.csv")
            print(f"Save Predict: {filename}")
            submission.to_csv(filename, index=False)

            # 3) [NEW] Save OOF Predictions
            print(f"--------------- SAVE OOF PREDICT ---------------")
            # 원본 Train 데이터에 예측값을 붙여서 저장 (rating: 실제값, predict: 예측값)
            # 순서는 원본 train_df 순서 그대로 유지됨 (oof_predictions를 인덱스 맞춰서 채웠으므로)
            oof_df = train_df.copy()
            oof_df["predict"] = oof_predictions

            oof_filename = filename.replace(".csv", "_OOF.csv")
            print(f"Save OOF Predict: {oof_filename}")
            oof_df.to_csv(oof_filename, index=False)

        else:
            # 예측 모드인데 K-Fold를 킨 경우 (보통 5개 모델 로드해야 해서 복잡함 -> 경고 후 단일 실행 추천)
            print(
                "!!! Warning: Prediction-only mode with K-Fold is not fully supported in this script version."
            )
            print("!!! Please run without --kfold for single model inference.")

    else:
        # ==========================================
        # [MODE 2] Original Single Run Strategy
        # ==========================================
        print(
            f"--------------- {args.model} Train/Valid Split (Original) ---------------"
        )
        data = data_split_fn(args, original_data)
        data = data_loader_fn(args, data)

        ####################### Setting for Log
        setting = Setting()
        if args.predict == False:
            log_path = setting.get_log_path(args)
            logger = Logger(args, log_path)
            logger.save_args()

        ######################## Model
        print(f"--------------- INIT {args.model} ---------------")
        model = getattr(model_module, args.model)(args.model_args[args.model], data).to(
            args.device
        )

        if args.train.resume:
            model.load_state_dict(torch.load(args.train.resume_path, weights_only=True))

        ######################## TRAIN
        if not args.predict:
            print(f"--------------- {args.model} TRAINING ---------------")
            model = train(args, model, data, logger, setting)

        ######################## INFERENCE
        if not args.predict:
            print(f"--------------- {args.model} PREDICT ---------------")
            predicts = test(args, model, data, setting)
        else:
            print(f"--------------- {args.model} PREDICT ---------------")
            predicts = test(args, model, data, setting, args.checkpoint)

        ######################## SAVE PREDICT
        print(f"--------------- SAVE {args.model} PREDICT ---------------")
        submission = pd.read_csv(args.dataset.data_path + "sample_submission.csv")
        submission["rating"] = predicts

        filename = setting.get_submit_filename(args)
        print(f"Save Predict: {filename}")
        submission.to_csv(filename, index=False)


if __name__ == "__main__":

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description="parser")
    arg = parser.add_argument
    str2dict = lambda x: {k: int(v) for k, v in (i.split(":") for i in x.split(","))}

    # add basic arguments
    arg(
        "--config",
        "-c",
        "--c",
        type=str,
        help="Configuration 파일을 설정합니다.",
        required=True,
    )

    # [NEW] K-Fold Argument
    arg(
        "--kfold",
        "-k",
        type=int,
        default=None,
        help="K-Fold 학습을 수행하려면 fold 개수를 입력하세요 (예: 5). 입력하지 않으면 일반 학습을 진행합니다.",
    )

    arg(
        "--predict",
        "-p",
        "--p",
        "--pred",
        type=ast.literal_eval,
        help="학습을 생략할지 여부를 설정할 수 있습니다.",
    )
    arg(
        "--checkpoint",
        "-ckpt",
        "--ckpt",
        type=str,
        help="학습을 생략할 때 사용할 모델을 설정할 수 있습니다.",
    )
    arg(
        "--model",
        "-m",
        "--m",
        type=str,
        choices=[
            "FM",
            "FFM",
            "DeepFM",
            "NCF",
            "WDN",
            "DCN",
            "Image_FM",
            "Image_DeepFM",
            "Text_FM",
            "Text_DeepFM",
            "ResNet_DeepFM",
        ],
        help="학습 및 예측할 모델을 선택할 수 있습니다.",
    )
    arg(
        "--seed",
        "-s",
        "--s",
        type=int,
        help="데이터분할 및 모델 초기화 시 사용할 시드를 설정할 수 있습니다.",
    )
    arg(
        "--device",
        "-d",
        "--d",
        type=str,
        choices=["cuda", "cpu", "mps"],
        help="사용할 디바이스를 선택할 수 있습니다.",
    )
    arg(
        "--wandb",
        "--w",
        "-w",
        type=ast.literal_eval,
        help="wandb를 사용할지 여부를 설정할 수 있습니다.",
    )
    arg(
        "--wandb_project",
        "--wp",
        "-wp",
        type=str,
        help="wandb 프로젝트 이름을 설정할 수 있습니다.",
    )
    arg(
        "--run_name",
        "--rn",
        "-rn",
        "--r",
        "-r",
        type=str,
        help="wandb에서 사용할 run 이름을 설정할 수 있습니다.",
    )
    arg("--model_args", "--ma", "-ma", type=ast.literal_eval)
    arg("--dataloader", "--dl", "-dl", type=ast.literal_eval)
    arg("--dataset", "--dset", "-dset", type=ast.literal_eval)
    arg("--optimizer", "-opt", "--opt", type=ast.literal_eval)
    arg("--loss", "-l", "--l", type=str)
    arg("--lr_scheduler", "-lr", "--lr", type=ast.literal_eval)
    arg("--metrics", "-met", "--met", type=ast.literal_eval)
    arg("--train", "-t", "--t", type=ast.literal_eval)

    args = parser.parse_args()

    ######################## Config with yaml
    config_args = OmegaConf.create(vars(args))
    config_yaml = OmegaConf.load(args.config) if args.config else OmegaConf.create()

    # args 우선 적용
    for key in config_args.keys():
        if config_args[key] is not None:
            config_yaml[key] = config_args[key]

    # 사용되지 않는 정보 삭제
    if config_yaml.predict == False:
        del config_yaml.checkpoint
        if config_yaml.wandb == False:
            del config_yaml.wandb_project, config_yaml.run_name
        config_yaml.model_args = OmegaConf.create(
            {config_yaml.model: config_yaml.model_args[config_yaml.model]}
        )
        config_yaml.optimizer.args = {
            k: v
            for k, v in config_yaml.optimizer.args.items()
            if k
            in getattr(
                optimizer_module, config_yaml.optimizer.type
            ).__init__.__code__.co_varnames
        }
        if config_yaml.lr_scheduler.use == False:
            del config_yaml.lr_scheduler.type, config_yaml.lr_scheduler.args
        else:
            config_yaml.lr_scheduler.args = {
                k: v
                for k, v in config_yaml.lr_scheduler.args.items()
                if k
                in getattr(
                    scheduler_module, config_yaml.lr_scheduler.type
                ).__init__.__code__.co_varnames
            }
        if config_yaml.train.resume == False:
            del config_yaml.train.resume_path

    # Configuration 출력
    print(OmegaConf.to_yaml(config_yaml))

    ######################## W&B
    if args.wandb:
        import wandb

        wandb.init(
            project=config_yaml.wandb_project,
            config=OmegaConf.to_container(config_yaml, resolve=True),
            name=config_yaml.run_name if config_yaml.run_name else None,
            notes=config_yaml.memo if hasattr(config_yaml, "memo") else None,
            tags=[config_yaml.model],
            resume="allow",
        )
        config_yaml.run_href = wandb.run.get_url()
        wandb.run.log_code("./src")

    ######################## MAIN
    main(config_yaml)

    if args.wandb:
        wandb.finish()
