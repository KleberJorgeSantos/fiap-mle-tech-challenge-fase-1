import logging
import sys
from pathlib import Path

from src.config import DATA_PATH, MODEL_DIR, SEED
from src.data.loader import get_features_target, load_raw, validate_schema
from src.data.preprocessing import build_pipeline, split_data, to_numpy
from src.evaluation.metrics import comparison_table, cost_analysis, evaluate_model
from src.models.baseline import BASELINES, train_baselines
from src.models.mlp import ChurnMLP
from src.models.trainer import predict_proba, save_model, train_mlp
from src.tracking.mlflow_utils import log_mlp_run, log_sklearn_run, setup_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Churn Prediction Pipeline ===")

    # 1. Load and prepare data
    df = load_raw(DATA_PATH)
    X, y = get_features_target(df)
    validate_schema(X)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    logger.info(
        "Split sizes → train=%d | val=%d | test=%d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # 2. Build preprocessing pipeline and transform
    pipeline = build_pipeline()
    X_train_t = to_numpy(pipeline.fit_transform(X_train))
    X_val_t = to_numpy(pipeline.transform(X_val))
    X_test_t = to_numpy(pipeline.transform(X_test))

    y_train_np = y_train.to_numpy()
    y_val_np = y_val.to_numpy()
    y_test_np = y_test.to_numpy()

    setup_experiment()
    all_results = {}

    # 3. Baselines
    logger.info("--- Training baselines ---")
    baseline_cv_results = train_baselines(X_train_t, y_train_np)

    for name, cv_metrics in baseline_cv_results.items():
        clf = BASELINES[name]
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_test_t)[:, 1]
        else:
            proba = clf.predict(X_test_t).astype(float)
        test_metrics = evaluate_model(y_test_np, proba)
        test_metrics["business_cost"] = cost_analysis(y_test_np, (proba >= 0.5).astype(int))
        all_results[name] = test_metrics
        log_sklearn_run(
            name=name,
            params={"model": name, "seed": SEED},
            metrics={**cv_metrics, **{f"test_{k}": v for k, v in test_metrics.items()}},
            model=clf,
        )

    # 4. MLP
    logger.info("--- Training MLP ---")
    input_dim = X_train_t.shape[1]
    mlp = ChurnMLP(input_dim=input_dim)
    history = train_mlp(
        model=mlp,
        X_train=X_train_t,
        y_train=y_train_np,
        X_val=X_val_t,
        y_val=y_val_np,
        epochs=100,
        lr=1e-3,
        batch_size=64,
        patience=10,
    )

    mlp_proba = predict_proba(mlp, X_test_t)
    mlp_metrics = evaluate_model(y_test_np, mlp_proba)
    mlp_metrics["business_cost"] = cost_analysis(y_test_np, (mlp_proba >= 0.5).astype(int))
    all_results["mlp"] = mlp_metrics

    log_mlp_run(
        name="mlp",
        params={
            "input_dim": input_dim,
            "hidden_dims": "[128,64,32]",
            "dropout": 0.3,
            "lr": 1e-3,
            "batch_size": 64,
            "seed": SEED,
        },
        metrics={f"test_{k}": v for k, v in mlp_metrics.items()},
        model=mlp,
        history=history,
    )

    # 5. Save model artifacts
    Path(MODEL_DIR).mkdir(exist_ok=True)
    save_model(mlp, pipeline, f"{MODEL_DIR}/churn_mlp")

    # 6. Comparison table
    table = comparison_table(all_results)
    logger.info("\n%s", table.to_string())
    table.to_csv(f"{MODEL_DIR}/comparison_table.csv")
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
