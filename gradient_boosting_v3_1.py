import argparse

from sklearn.ensemble import GradientBoostingClassifier

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

    model = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=args.seed,
    )

    evaluate_model(model, x_train, y, n_splits=args.folds, seed=args.seed)
    model.fit(x_train, y)
    predictions = model.predict_proba(x_test)[:, 1]

    save_submission(
        sample_submission,
        test["id"],
        predictions,
        "submissions/submission_3_1_tuned_gradient_boosting.csv",
    )


if __name__ == "__main__":
    main()
