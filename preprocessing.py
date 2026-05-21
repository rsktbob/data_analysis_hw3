from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


TARGET = "ACTION"
ID_COL = "id"


BASE_FEATURES = [
    "RESOURCE",
    "MGR_ID",
    "ROLE_ROLLUP_1",
    "ROLE_ROLLUP_2",
    "ROLE_DEPTNAME",
    "ROLE_TITLE",
    "ROLE_FAMILY_DESC",
    "ROLE_FAMILY",
    "ROLE_CODE",
]


def load_data(train_path="train.csv", test_path="test.csv", sample_path="sampleSubmission.csv"):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)
    return train, test, sample_submission


def print_data_summary(train, test):
    print(f"Train shape: {train.shape}")
    print(f"Test shape : {test.shape}")
    print("")
    print("ACTION distribution:")
    print(train[TARGET].value_counts(normalize=True).sort_index())
    print("")
    print("Number of unique values:")
    print(train[BASE_FEATURES].nunique().sort_values(ascending=False))


def smooth_target_mean(values, target, global_mean, smoothing):
    stats = pd.DataFrame({"value": values, "target": target})
    grouped = stats.groupby("value")["target"].agg(["sum", "count"])
    return (grouped["sum"] + global_mean * smoothing) / (grouped["count"] + smoothing)


def make_oof_target_encoding(train, test, features, n_splits=5, smoothing=10, seed=42):
    y = train[TARGET].values
    global_mean = train[TARGET].mean()
    train_encoded = pd.DataFrame(index=train.index)
    test_encoded = pd.DataFrame(index=test.index)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for feature in features:
        encoded_col = f"{feature}_te"
        train_encoded[encoded_col] = np.nan

        for train_idx, valid_idx in skf.split(train, y):
            fold_train = train.iloc[train_idx]
            fold_valid = train.iloc[valid_idx]
            mapping = smooth_target_mean(
                fold_train[feature],
                fold_train[TARGET],
                global_mean=fold_train[TARGET].mean(),
                smoothing=smoothing,
            )
            train_encoded.loc[fold_valid.index, encoded_col] = (
                fold_valid[feature].map(mapping).fillna(global_mean)
            )

        full_mapping = smooth_target_mean(
            train[feature],
            train[TARGET],
            global_mean=global_mean,
            smoothing=smoothing,
        )
        test_encoded[encoded_col] = test[feature].map(full_mapping).fillna(global_mean)

    return train_encoded, test_encoded


def evaluate_model(model, x, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(x, y), start=1):
        x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(x_train, y_train)
        pred = model.predict_proba(x_valid)[:, 1]
        auc = roc_auc_score(y_valid, pred)
        scores.append(auc)
        print(f"Fold {fold}: AUC = {auc:.6f}")

    print(f"CV mean AUC = {np.mean(scores):.6f}")
    print(f"CV std  AUC = {np.std(scores):.6f}")
    return scores


def save_submission(sample_submission, test_ids, predictions, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    id_column = sample_submission.columns[0]
    target_column = sample_submission.columns[1]
    submission = pd.DataFrame({
        id_column: test_ids.values,
        target_column: predictions,
    })
    submission.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
