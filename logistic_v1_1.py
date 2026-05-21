import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from preprocessing import (
    BASE_FEATURES,
    TARGET,
    evaluate_model,
    load_data,
    print_data_summary,
    save_submission,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train, test, sample_submission = load_data()
    print_data_summary(train, test)

    x_train = train[BASE_FEATURES]
    x_test = test[BASE_FEATURES]
    y = train[TARGET]

    model = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed),
    )

    evaluate_model(model, x_train, y, n_splits=args.folds, seed=args.seed)
    model.fit(x_train, y)
    predictions = model.predict_proba(x_test)[:, 1]

    save_submission(
        sample_submission,
        test["id"],
        predictions,
        "submissions/submission_1_1_one_hot_logistic.csv",
    )


if __name__ == "__main__":
    main()
