from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List
import warnings

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from skopt.space import Real, Integer, Categorical

from tqdm import tqdm


from src import data as data_utils
from src import eval as eval_utils
from src.model_training import train_model

warnings.filterwarnings("ignore", message="The objective has been evaluated at point")


DATA_DIR = Path("data/inputs")
OUTPUT_DIR = Path("data/outputs")
DATA_FILE = DATA_DIR / "ZSAC_ZVAC_Collaboration_data_v3_20241121.csv"
DATA_FILE_SEQUENCE = DATA_DIR / "ZSAC_ZVAC_Collaboration_sequence_data_20250124.csv"
FILTERED_OUTPUT = DATA_DIR / "filtered_data.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REF_DATE = pd.to_datetime("2022-01-01")
TARGET_COL = "final_outcome_amp"
ROUND_LABELS = {
    "agnostic": "Round 1.1",
    "background": "Round 2.0",
    "sequence": "Round 2.1",
}
DATA_MODES = ("agnostic", "background", "sequence")

N_SPLITS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Flags for agnostic filtering
APPLY_PCA = False
FILTER_MI = True
FILTER_CORRELATION = False
FILTER_RF = False
K_MI = 15
K_RF = 15
CORR_THRESHOLD = 0.9
PCA_COMPONENTS = 0.95

COLOR_MAP = {"background": "tab:blue", "agnostic": "tab:orange", "sequence": "tab:green"}

MODEL_CONFIGS = {
    "RandomForest": {
        "factory": lambda: RandomForestClassifier(random_state=RANDOM_STATE),
        "search_spaces": {
            "n_estimators": Integer(10, 1000),
            "max_depth": Integer(1, 60),
            "min_samples_split": Integer(2, 50),
            "min_samples_leaf": Integer(1, 20),
            "max_features": Categorical([None, "sqrt", "log2"]),
            "bootstrap": Categorical([True, False]),
            "criterion": Categorical(["gini", "entropy"]),
            "class_weight": Categorical(["balanced", None]),
        },
    },
    "LogisticRegression": {
        "factory": lambda: LogisticRegression(random_state=42, solver="liblinear", max_iter=10000),
        "search_spaces": [
            {
                "solver": Categorical(["liblinear"]),
                "penalty": Categorical(["l1", "l2"]),
                "C": Real(1e-3, 1e3, prior="log-uniform"),
                "class_weight": Categorical(["balanced", None]),
            },
            {
                "solver": Categorical(["saga"]),
                "penalty": Categorical(["l1", "l2"]),
                "C": Real(1e-3, 1e3, prior="log-uniform"),
                "class_weight": Categorical(["balanced", None]),
            },
        ],
    },
    "LinearDiscriminantAnalysis": {
        "factory": lambda: LinearDiscriminantAnalysis(),
        "search_spaces": {
            "solver": Categorical(["svd"]),
            "tol": Real(1e-6, 1e-2, prior="log-uniform"),
        },
    },
    "GradientBoosting": {
        "factory": lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
        "search_spaces": {
            "learning_rate": Real(0.01, 0.5, prior="log-uniform"),
            "n_estimators": Integer(50, 1000),
            "max_depth": Integer(2, 30),
            "min_samples_leaf": Integer(1, 60),
            "subsample": Real(0.5, 1.0),
            "max_features": Categorical([None, "sqrt", "log2"]),
        },
    },
    "XGBoost": {
        "factory": lambda: XGBClassifier(
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
            tree_method="hist",
            predictor="cpu_predictor",
            verbosity=0,
        ),
        "search_spaces": {
            "learning_rate": Real(0.01, 0.5, prior="log-uniform"),
            "n_estimators": Integer(50, 1000),
            "max_depth": Integer(3, 12),
            "reg_lambda": Real(0.0, 10.0),
            "min_child_weight": Integer(1, 10),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.3, 1.0),
        },
        "bayes_n_jobs": 1,
    },
}


#MODEL_CONFIGS = {"RandomForest": MODEL_CONFIGS["RandomForest"]}


def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.replace(r"[\\[\\]<>]", "", regex=True)
    return df


def load_datasets():
    base = pd.read_csv(DATA_FILE)
    base = data_utils.filter_data_date(base, REF_DATE).dropna(axis=1, how="all")
    base.to_csv(FILTERED_OUTPUT)
    if "visit_date" in base.columns:
        base = base.drop(columns=["visit_date"])
    base = sanitize_columns(base).set_index("record_id")

    seq = pd.read_csv(DATA_FILE_SEQUENCE)
    seq = sanitize_columns(seq.dropna(axis=1, how="all")).set_index("record_id")
    return base, seq


def _prefix_search_spaces(spaces, prefix: str):
    if isinstance(spaces, list):
        return [{f"{prefix}{k}": v for k, v in space.items()} for space in spaces]
    return {f"{prefix}{k}": v for k, v in spaces.items()}


def _build_pipeline_and_search_space(
    *,
    estimator_factory,
    search_spaces,
    data_mode: str,
    n_features: int,
    enable_selector: bool,
    k_default: int,
):
    steps = [
        ("drop_empty", data_utils.DropAllMissingColumns()),
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]

    spaces_prefixed = _prefix_search_spaces(search_spaces, "clf__")

    if data_mode == "agnostic" and enable_selector:
        k_max = max(1, min(k_default, n_features))
        steps.append(("selector", SelectKBest(mutual_info_classif, k=k_max)))
        if isinstance(spaces_prefixed, list):
            for space in spaces_prefixed:
                space["selector__k"] = Integer(1, k_max)
        else:
            spaces_prefixed["selector__k"] = Integer(1, k_max)

    steps.append(("clf", estimator_factory()))
    pipe = Pipeline(steps)
    return pipe, spaces_prefixed


def build_features(
    data_mode: str,
    raw_data: pd.DataFrame,
    raw_sequence: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    if data_mode == "background":
        data = data_utils.create_features_background(raw_data, REF_DATE)
    elif data_mode == "agnostic":
        data = data_utils.create_features_agnostic(raw_data, REF_DATE)
    elif data_mode == "sequence":
        data = data_utils.create_features_sequence(raw_data, raw_sequence, REF_DATE)
    else:
        raise ValueError(f"Unsupported data mode: {data_mode}")

    data = data.dropna(subset=[TARGET_COL])
    X = data.drop(TARGET_COL, axis=1)
    y = data[TARGET_COL]

    # Ensure downstream numeric-only preprocessing (imputer/scaler) does not fail
    # if any categorical strings slip through feature construction.
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        X[col] = X[col].astype("category").cat.codes.replace({-1: np.nan})

    # Drop columns with no observed values (all missing) to avoid imputer warnings.
    X = X.dropna(axis=1, how="all")

    X.columns = [str(col).replace("[", "").replace("]", "").replace("<", "").replace(">", "") for col in X.columns]
    return X, y


def summarize_parameters(params_per_split: List[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    all_keys = {key for params in params_per_split for key in params}
    for key in all_keys:
        values = [params[key] for params in params_per_split if key in params]
        if values:
            summary[key] = Counter(values).most_common(1)[0][0]
    return summary


def train_models_for_mode(
    data_mode: str,
    X: pd.DataFrame,
    y: pd.Series,
):
    split_metrics_by_model = defaultdict(list)
    aucs_per_model = defaultdict(list)
    split_params = defaultdict(list)
    fitted_models = defaultdict(list)
    feature_names_per_split = defaultdict(list)
    metric_summaries = {}
    optimized_summaries = {}
    rf_feature_importances = None

    #print("\n=== Mean Accuracy, AUC, F1, Sensitivity, Specificity, Precision, AUPR (All Splits) ===")
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n-- Training {model_name} --")
        split_metrics: List[Dict[str, object]] = []
        aucs: List[float] = []
        params_for_model: List[Dict[str, object]] = []
        last_importances = None

        n_features = X.shape[1]
        estimator, search_spaces = _build_pipeline_and_search_space(
            estimator_factory=config["factory"],
            search_spaces=config["search_spaces"],
            data_mode=data_mode,
            n_features=n_features,
            enable_selector=FILTER_MI,
            k_default=K_MI,
        )

        split_iter = range(N_SPLITS)
        if tqdm is not None:
            split_iter = tqdm(
                split_iter,
                desc=f"{data_mode}/{model_name}",
                unit="split",
                leave=False,
            )

        for split in split_iter:
            result = train_model(
                estimator=estimator,
                X=X,
                y=y,
                search_spaces=search_spaces,
                scoring="roc_auc",
                cv_splits=5,
                test_size=TEST_SIZE,
                n_iter=config.get("n_iter", 32),
                random_state=RANDOM_STATE + split,
                n_jobs=config.get("bayes_n_jobs", -1),
                verbose=0,
            )

            if result.y_scores is None:
                print(f"Warning: {model_name} split {split+1} did not return probability scores; skipping metrics.")
                continue

            params_for_model.append(result.best_params)
            fitted_models[model_name].append(result.best_estimator)

            feat_names = list(X.columns)
            be = result.best_estimator
            if hasattr(be, "named_steps") and "selector" in be.named_steps:
                selector = be.named_steps["selector"]
                if hasattr(selector, "get_support"):
                    mask = selector.get_support()
                    if len(mask) == len(feat_names):
                        feat_names = [n for n, keep in zip(feat_names, mask) if keep]
            feature_names_per_split[model_name].append(feat_names)

            metrics = eval_utils.evaluate_split(result.y_true, result.y_pred, result.y_scores)
            split_metrics.append(metrics)
            aucs.append(metrics["roc_auc"])

            if model_name == "RandomForest":
                rf_est = result.best_estimator
                if hasattr(rf_est, "named_steps") and "clf" in rf_est.named_steps:
                    rf_est = rf_est.named_steps["clf"]
                if hasattr(rf_est, "feature_importances_"):
                    last_importances = rf_est.feature_importances_

        if not split_metrics:
            continue

        summary = eval_utils.summarize_metrics(split_metrics, len(split_metrics))
        split_metrics_by_model[model_name] = split_metrics
        aucs_per_model[model_name] = aucs
        split_params[model_name] = params_for_model
        metric_summaries[model_name] = summary["standard"]
        optimized_summaries[model_name] = summary["optimized"]

        #eval_utils.print_metric_summary(model_name, summary["standard"])

        if model_name == "RandomForest" and last_importances is not None:
            last_feats = feature_names_per_split[model_name][-1] if feature_names_per_split[model_name] else list(X.columns)
            rf_feature_importances = dict(zip(last_feats, last_importances))

    print("\n=== Optimized Threshold Metrics (F1-Optimized Threshold) ===")
    for model_name, summary in optimized_summaries.items():
        eval_utils.print_optimized_summary(model_name, summary)

    param_summaries = {m: summarize_parameters(splits) for m, splits in split_params.items()}
    if False:
        print("\n=== Optimal Hyperparameters (Most Common Across Splits) ===")
        for model_name, params in param_summaries.items():
            print(f"\n** {model_name} **")
            if not params:
                print("(defaults)")
            for param, value in params.items():
                print(f"{param}: {value}")

    round_label = ROUND_LABELS.get(data_mode, data_mode.title())
    n_splits_plotted = max((len(metrics) for metrics in split_metrics_by_model.values()), default=0)
    eval_utils.plot_mean_roc_panel(
        split_metrics_by_model,
        data_mode=data_mode,
        round_label=round_label,
        n_splits=n_splits_plotted,
        output_path=str(OUTPUT_DIR / f"roc_curves_{data_mode}_{N_SPLITS}_splits.png"),
    )

    if rf_feature_importances:
        print(f"\n=== Feature Importances for RandomForest ({round_label}) ===")
        for feature, importance in sorted(rf_feature_importances.items(), key=lambda item: item[1], reverse=True):
            print(f"{feature}: {importance:.4f}")

    return {
        "split_metrics": dict(split_metrics_by_model),
        "aucs": dict(aucs_per_model),
        "split_params": dict(split_params),
        "param_summary": param_summaries,
        "fitted_models": dict(fitted_models),
        "feature_names": dict(feature_names_per_split),
    }


def report_auc_differences(
    results: Dict[str, Dict[str, object]],
    numerator_mode: str,
    denominator_mode: str,
    *,
    bar_color: str,
    mean_color: str,
):
    numerator = results.get(numerator_mode, {}).get("aucs", {})
    denominator = results.get(denominator_mode, {}).get("aucs", {})
    if not numerator or not denominator:
        return

    #print(f"\n=== AUROC Difference ({ROUND_LABELS.get(numerator_mode)} - {ROUND_LABELS.get(denominator_mode)}) across splits ===")
    for model_name in MODEL_CONFIGS.keys():
        diffs = eval_utils.compute_auc_differences(numerator.get(model_name, []), denominator.get(model_name, []))
        if diffs.size == 0:
            continue
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))
        #print(f"\n** {model_name} **")
        #print(f"  Mean AUROC Difference: {mean_diff:.4f}")
        #print(f"  Std Dev of Difference: {std_diff:.4f}")

        title = (
            f"AUROC Difference per Split for {model_name}\n"
            f"({ROUND_LABELS.get(numerator_mode)} - {ROUND_LABELS.get(denominator_mode)})"
        )
        output_path = OUTPUT_DIR / f"auroc_diff_{numerator_mode}_vs_{denominator_mode}_{model_name}.png"
        eval_utils.plot_auc_differences(
            diffs,
            title=title,
            ylabel="AUROC Difference",
            color=bar_color,
            mean_line_color=mean_color,
            output_path=str(output_path),
        )


def plot_cross_mode_rocs(results: Dict[str, Dict[str, object]]):
    labels = {mode: ROUND_LABELS.get(mode, mode.title()) for mode in DATA_MODES}
    for model_name in MODEL_CONFIGS.keys():
        split_metrics_per_mode = {}
        for mode in DATA_MODES:
            metrics = results.get(mode, {}).get("split_metrics", {}).get(model_name)
            if metrics:
                split_metrics_per_mode[mode] = metrics
        if len(split_metrics_per_mode) < 2:
            continue
        eval_utils.plot_roc_comparison_by_mode(
            model_name=model_name,
            split_metrics_per_mode=split_metrics_per_mode,
            labels=labels,
            colors=COLOR_MAP,
            n_points=100,
            output_path=str(OUTPUT_DIR / f"roc_compare_{model_name}.png"),
        )


def main():
    raw_data, raw_sequence = load_datasets()
    all_results: Dict[str, Dict[str, object]] = {}

    for data_mode in DATA_MODES:
        print(f"\n{'=' * 20} RUNNING IN MODE: {data_mode.upper()} {'=' * 20}\n")
        X, y = build_features(data_mode, raw_data, raw_sequence)

        print(f"Final shape of features X: {X.shape}")
        print(f"Final shape of target y: {y.shape}")
        print("\nTarget distribution:\n", y.value_counts(normalize=True))

        mode_results = train_models_for_mode(data_mode, X, y)
        all_results[data_mode] = mode_results

    report_auc_differences(all_results, "background", "agnostic", bar_color="skyblue", mean_color="darkblue")
    report_auc_differences(all_results, "sequence", "background", bar_color="lightcoral", mean_color="blue")
    plot_cross_mode_rocs(all_results)


if __name__ == "__main__":
    main()
