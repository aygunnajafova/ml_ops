from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data():
    data_path = Path("data/data_usage_production.parquet")
    df = pd.read_parquet(data_path)
    return df


def prepare_features(df):
    cols_to_consider = [
        "refill_total_m4",
        "istesen_data_extention_m2",
        "refill_total_m3",
        "lastrefillamount_m2",
        "tenure",
        "dpi_telegram_m2",
        "dpi_youtube_m2",
        "dpi_instagram_m2",
        "dpi_https_and_default_m2",
        "dpi_tik_tok_m2",
        "dpi_tik_tok_m3",
        "dpi_tik_tok_m4",
        "data_pack_usg_m2",
        "data_pack_usg_m3",
        "data_pack_usg_m6",
        "data_amount_lte_m2",
        "data_compl_usg_local_m2",
        "data_compl_usg_local_m3",
        "data_compl_usg_local_m4",
        "data_amount_lte_m6",
        "data_compl_usg_local_m6",
        "data_compl_usg_local_m5",
        "telephone_number",
    ]
    cat_cols = []
    num_cols = []
    for col in cols_to_consider:
        if col in df.columns:
            if df[col].dtype == "object":
                cat_cols.append(col)
            else:
                num_cols.append(col)
    if "data_compl_usg_local_m1" in num_cols:
        num_cols.remove("data_compl_usg_local_m1")
    print(f"Categorical features: {cat_cols}")
    print(f"Numerical features: {len(num_cols)}")
    return df, cat_cols, num_cols


def create_preprocessing_pipeline(cat_cols, num_cols):
    num_trans = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_trans = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_trans, num_cols),
            ("cat", cat_trans, cat_cols),
        ]
    )
    return preprocessor


def train_model(df, cat_cols, num_cols, preprocessor):
    sample_size = min(5000, len(df))
    df_sample = df.head(sample_size)
    X = df_sample.drop("data_compl_usg_local_m1", axis=1)
    y = df_sample["data_compl_usg_local_m1"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_leaf=5,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    print("Training model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")
    return model, X_test, y_test, y_pred


def hyperparameter_tuning(df, cat_cols, num_cols, preprocessor):
    sample_size = min(50000, len(df))
    X = df.drop("data_compl_usg_local_m1", axis=1).head(sample_size)
    y = df["data_compl_usg_local_m1"].head(sample_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(random_state=42)),
        ]
    )
    param_grid = {
        "regressor__max_depth": [10, 15, 20],
        "regressor__n_estimators": [50, 100],
        "regressor__min_samples_leaf": [3, 5, 10],
    }
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="r2",
        cv=3,
        verbose=1,
        n_jobs=-1,
    )
    print("Performing hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.3f}")
    return grid_search.best_estimator_


def save_model(model, model_name="telecom_data_usage_model.pkl"):
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / model_name
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    return model_path


def main():
    df = load_data()
    print(f"Loaded dataset with shape: {df.shape}")
    df, cat_cols, num_cols = prepare_features(df)
    preprocessor = create_preprocessing_pipeline(cat_cols, num_cols)
    model, X_test, y_test, y_pred = train_model(df, cat_cols, num_cols, preprocessor)
    model_path = save_model(model)
    print("Training completed successfully!")
    return model, model_path


if __name__ == "__main__":
    main()
