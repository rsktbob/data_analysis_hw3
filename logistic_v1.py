import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import (
    BASE_FEATURES,
    TARGET,
    evaluate_model,
    load_data,
    make_oof_target_encoding,
    print_data_summary,
    save_submission,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--smoothing", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, test, sample_submission = load_data()
    print_data_summary(train, test)

    x_train, x_test = make_oof_target_encoding(
        train,
        test,
        BASE_FEATURES,
        n_splits=args.folds,
        smoothing=args.smoothing,
        seed=args.seed,
    )
    y = train[TARGET]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed),
    )

    evaluate_model(model, x_train, y, n_splits=args.folds, seed=args.seed)
    model.fit(x_train, y)
    predictions = model.predict_proba(x_test)[:, 1]

    save_submission(
        sample_submission,
        test["id"],
        predictions,
        "submissions/submission_1_target_encoding_logistic.csv",
    )


if __name__ == "__main__":
    main()
