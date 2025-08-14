from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_processed_data():
    processed_dir = Path("data/processed")
    with open(processed_dir / "telecom_feature_info.pkl", "rb") as f:
        feature_info = joblib.load(f)
    X_sample = pd.read_csv(processed_dir / "telecom_features_sample.csv")
    y_sample = pd.read_csv(processed_dir / "telecom_target_sample.csv")
    return X_sample, y_sample, feature_info


def load_trained_model():
    models_dir = Path("models")
    model_path = models_dir / "telecom_data_usage_model.pkl"
    if model_path.exists():
        model = joblib.load(model_path)
        print(f"Loaded trained model from: {model_path}")
        return model
    else:
        return None


def create_preprocessing_pipeline(feature_info):
    cat_cols = feature_info["categorical_features"]
    num_cols = feature_info["numerical_features"]
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


def predict_data_usage(model, X_new, feature_info):
    if model is None:
        print("No model available for prediction.")
        return None
    expected_cols = (
        feature_info["categorical_features"] + feature_info["numerical_features"]
    )
    missing_cols = set(expected_cols) - set(X_new.columns)
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        for col in missing_cols:
            if col in feature_info["categorical_features"]:
                X_new[col] = "unknown"
            else:
                X_new[col] = 0
    X_new = X_new[expected_cols]
    try:
        predictions = model.predict(X_new)
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


def evaluate_model_performance(model, X_test, y_test, feature_info):
    if model is None:
        return None
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Model Performance on Test Data:")
    print(f"MSE: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")

    print(f"Prediction range: {float(y_pred.min()):.2f} to {float(y_pred.max()):.2f}")
    print(f"True value range: {float(y_test.min()):.2f} to {float(y_test.max()):.2f}")

    return {"mse": mse, "r2": r2, "predictions": y_pred}


def main():
    X_sample, y_sample, feature_info = load_processed_data()
    model = load_trained_model()
    if model is not None:
        performance = evaluate_model_performance(
            model, X_sample, y_sample, feature_info
        )
        print("\nPrediction analysis completed successfully!")
        return model, performance
    else:
        return None, None, None


if __name__ == "__main__":
    main()
