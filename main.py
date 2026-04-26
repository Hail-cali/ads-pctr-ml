"""CLI entry point for CTR prediction pipeline.

Usage:
    python main.py train --model deepfm
    python main.py export
    python main.py serve
    python main.py all --model deepfm
    python main.py compare
    python main.py hpo --trials 50
"""

import argparse
import logging

import yaml

from data.criteo import CriteoPreprocessor
from model import create_model
from serving.server import export_onnx
from train.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_train(config: dict, model_name: str):
    dc = config.get("data", {})
    preprocessor = CriteoPreprocessor(
        hash_bucket_size=dc.get("hash_bucket_size", 100_000),
        sample_size=dc.get("sample_size"),
    )
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        dc["filepath"],
        batch_size=dc.get("batch_size", 4096),
        num_workers=dc.get("num_workers", 4),
        seed=dc.get("seed", 42),
    )

    model = create_model(model_name, config)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_name}, Parameters: {param_count:,}")

    trainer = Trainer(model, config, model_name=model_name)
    best = trainer.train(train_loader, val_loader)
    test_metrics = trainer.test(test_loader)

    return model, trainer, {**best, **test_metrics}


def cmd_export(config: dict, model_name: str):
    model = create_model(model_name, config)
    trainer = Trainer(model, config, model_name=model_name)
    trainer.load_checkpoint()
    export_onnx(model, config)


def cmd_serve(config: dict):
    import uvicorn

    sc = config.get("serving", {})
    uvicorn.run("serving.server:app", host=sc.get("host", "0.0.0.0"), port=sc.get("port", 8000), reload=False)


def cmd_compare(config: dict):
    """Train all models and compare."""
    from experiments.visualize import plot_model_comparison, save_results_table

    results = {}
    for name in ["lr", "fm", "deepfm", "dcn_v2"]:
        logger.info(f"=== Training {name} ===")
        _, trainer, metrics = cmd_train(config, name)
        results[name] = metrics

    plot_model_comparison(results)
    save_results_table(results)
    logger.info("Comparison complete. See outputs/")


def cmd_hpo(config: dict, config_path: str, n_trials: int):
    from experiments.hpo import run_hpo

    run_hpo(config_path=config_path, n_trials=n_trials)


def main():
    parser = argparse.ArgumentParser(description="CTR Prediction Pipeline")
    parser.add_argument("command", choices=["train", "export", "serve", "all", "compare", "hpo"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="deepfm", choices=["deepfm"])
    parser.add_argument("--trials", type=int, default=50, help="HPO trials")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.command == "train":
        cmd_train(config, args.model)
    elif args.command == "export":
        cmd_export(config, args.model)
    elif args.command == "serve":
        cmd_serve(config)
    elif args.command == "all":
        cmd_train(config, args.model)
        cmd_export(config, args.model)
        cmd_serve(config)
    elif args.command == "compare":
        cmd_compare(config)
    elif args.command == "hpo":
        cmd_hpo(config, args.config, args.trials)


if __name__ == "__main__":
    main()
