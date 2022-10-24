import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.neural_network import MLPRegressor

SimpleImputer.get_feature_names_out = lambda self, names=None: self.feature_names_in_

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


def main():
    seed = 123
    n_splits = 10
    n_max_iters = 100
    hidden_layer_structure = (25, 25)

    df = load_full_dataset("FILEPATH_HERE")
    
    # Define regressor
    regressor = MLPRegressor(
        hidden_layer_sizes=hidden_layer_structure,
        random_state=seed,
        max_iter=n_max_iters,
        # activation='logistic',
        solver='lbfgs',
        # batch_size=50,
        # verbose=True,
    )

    # Restrict by column values
    # df = df[df["Some_feature"].isin(["Some_feature_category1", "Some_feature_category2"])]

    # Restrict by row count per chosen column values
    # df = df.groupby("Some_feature").filter(lambda x: len(x) > 100)

    # Select target feature.
    target_feature = ""

    categorical_features = [
        # list categorical features here, selected features can be easily commented out if wanted.
    ]

    numerical_features = [
        # list numerical features here, selected features can be easily commented out if wanted.
    ]

    # Restrict features to the selected ones, assign feature matrix and target vector.
    df = df[categorical_features + numerical_features + [target_feature]]
    X, y = df[df.columns.difference([target_feature])], df[target_feature]

    numerical_transformer = Pipeline(steps=[("imputer", SimpleImputer()), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    data_preprocessor = ColumnTransformer(
        transformers=[
            ("cat_cols", categorical_transformer, categorical_features),
            ("num_cols", numerical_transformer, numerical_features),
        ]
    )

    pipeline = Pipeline([("data_preprocessing", data_preprocessor), ("regressor", regressor)])

    cv = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    results = cross_validate(
        pipeline, X, y.values.ravel(),
        cv=cv,
        # return_estimator=True,
        n_jobs=n_splits,
        # scoring="r2",
        verbose=0
    )

    print(results["test_score"])
    print(sum(results["test_score"]) / len(results["test_score"]))


def load_full_dataset(filename):
    data_types = {
        # Mapping from feature name to numpy data type.
    }

    return pd.read_csv(filename, delimiter=",", encoding="utf8", dtype=data_types, na_values="Unknown")


if __name__ == "__main__":
    main()
