import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

SimpleImputer.get_feature_names_out = lambda self, names=None: self.feature_names_in_

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


def main():
    seed = 123
    n_estimators = 25
    n_splits = 10

    df = load_full_dataset("FILEPATH_HERE")

    # Choose regressor
    # regressor = DecisionTreeRegressor(random_state=seed)
    regressor = RandomForestRegressor(n_estimators=n_estimators, n_jobs=2)

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
        pipeline, X, y.values.ravel(), cv=cv, return_estimator=True, n_jobs=-1, scoring="r2"
    )

    display_results(results)


def display_results(results):
    aggregate_feature_importances = calculate_aggregate_feature_importances(results)

    print("Feature importances:")
    print(aggregate_feature_importances.describe().loc[["mean", "std"]])
    print()

    accuracy = results["test_score"]

    print(f"R2 (mean): {accuracy.mean():.3f} ± {accuracy.std():.3f}")


def calculate_aggregate_feature_importances(results):
    # Concatenate feature importances from all estimators to a dataframe.
    feature_importances = None

    for estimator_pipeline in results["estimator"]:
        model = estimator_pipeline[1]
        feature_names = estimator_pipeline[:-1].get_feature_names_out()

        if feature_importances is None:
            feature_importances = pd.DataFrame(None, columns=feature_names)

        new_importances = pd.DataFrame([model.feature_importances_], columns=feature_names)
        feature_importances = pd.concat([feature_importances, new_importances])

    feature_importances = feature_importances.replace(np.nan, 0)

    # Aggregate encoded values (such as one hot encoded values) to a single categorical feature value.
    aggregate_feature_importances = pd.DataFrame()

    for feature in feature_importances:
        if feature.startswith("num_cols__"):
            trimmed_feature_name = feature[len("num_cols__") :]
        else:
            trimmed_feature_name = feature[len("cat_cols__") :].split("_")[0]

        if trimmed_feature_name in aggregate_feature_importances.columns:
            aggregate_feature_importances[trimmed_feature_name] += feature_importances[feature]
        else:
            aggregate_feature_importances[trimmed_feature_name] = feature_importances[feature]

    return aggregate_feature_importances


def load_full_dataset(filename):
    data_types = {
        # Mapping from feature name to numpy data type.
    }

    return pd.read_csv(filename, delimiter=",", encoding="utf8", dtype=data_types, na_values="Unknown")


if __name__ == "__main__":
    main()
