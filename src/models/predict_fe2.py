from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)


def load_trained_models():
    models_dir = Path("models")
    models = {}
    reg_model_path = models_dir / "ramen_regression_model.pkl"
    reg_scaler_path = models_dir / "ramen_regression_scaler.pkl"
    reg_pca_path = models_dir / "ramen_regression_pca.pkl"
    if reg_model_path.exists() and reg_scaler_path.exists() and reg_pca_path.exists():
        models["regression"] = {
            "model": joblib.load(reg_model_path),
            "scaler": joblib.load(reg_scaler_path),
            "pca": joblib.load(reg_pca_path),
        }
        print(f"Loaded regression model from: {reg_model_path}")
    else:
        print("Regression model not found. Please run train_fe2.py first.")
        models["regression"] = None
    clf_model_path = models_dir / "ramen_classification_model.pkl"
    clf_scaler_path = models_dir / "ramen_classification_scaler.pkl"
    clf_pca_path = models_dir / "ramen_classification_pca.pkl"
    if clf_model_path.exists() and clf_scaler_path.exists() and clf_pca_path.exists():
        models["classification"] = {
            "model": joblib.load(clf_model_path),
            "scaler": joblib.load(clf_scaler_path),
            "pca": joblib.load(clf_pca_path),
        }
    else:
        models["classification"] = None
    return models


def load_processed_data():
    processed_dir = Path("data/processed")
    df = pd.read_csv(processed_dir / "ramen_ratings_cleaned.csv", index_col=0)
    variety_embeddings = pd.read_csv(processed_dir / "ramen_variety_embeddings.csv")
    categorical_encoded = pd.read_csv(
        processed_dir / "ramen_categorical_encoded.csv", index_col=0
    )
    brand_encoded = pd.read_csv(processed_dir / "ramen_brand_encoded.csv", index_col=0)
    return df, variety_embeddings, categorical_encoded, brand_encoded


def prepare_features_for_prediction(
    df, variety_embeddings, categorical_encoded, brand_encoded
):
    features_list = []
    if variety_embeddings is not None:
        variety_embeddings.index = df.index
        features_list.append(variety_embeddings)
    if categorical_encoded is not None:
        features_list.append(categorical_encoded)
    if brand_encoded is not None:
        features_list.append(brand_encoded)
    if features_list:
        X = pd.concat(features_list, axis=1)
    else:
        X = df[["Country", "Style", "Brand"]]
        X = pd.get_dummies(X, drop_first=True)
    return X


def predict_star_rating(models, X, model_type="regression"):
    if models is None or models[model_type] is None:
        print(f"No {model_type} model available for prediction.")
        return None
    model_components = models[model_type]
    try:
        X_scaled = model_components["scaler"].transform(X)
        X_pca = model_components["pca"].transform(X_scaled)
        predictions = model_components["model"].predict(X_pca)
        return predictions
    except Exception:
        return None


def evaluate_predictions(y_true, y_pred_reg, y_pred_clf, models):
    results = {}
    if y_pred_reg is not None:
        mse = mean_squared_error(y_true, y_pred_reg)
        r2 = r2_score(y_true, y_pred_reg)
        print("\nRegression Model Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        results["regression"] = {"mse": mse, "r2": r2}
    if y_pred_clf is not None:
        y_true_class = y_true.round().astype(int)
        y_pred_class = y_pred_clf.round().astype(int)
        print("\nClassification Model Performance:")
        print(classification_report(y_true_class, y_pred_class))
        cm = confusion_matrix(y_true_class, y_pred_class)
        results["classification"] = {"confusion_matrix": cm}
    return results


def plot_prediction_results(y_true, y_pred_reg, y_pred_clf):
    plt.figure(figsize=(15, 5))
    if y_pred_reg is not None:
        plt.subplot(1, 3, 1)
        plt.scatter(y_true, y_pred_reg, alpha=0.6)
        plt.xlabel("True Star Ratings")
        plt.ylabel("Predicted Star Ratings")
        plt.title("Ramen Ratings: Regression Predictions")
        min_val = min(y_true.min(), y_pred_reg.min())
        max_val = max(y_true.max(), y_pred_reg.max())
        plt.plot([min_val, max_val], [min_val, max_val], "--r")
        plt.grid(True)
    if y_pred_clf is not None:
        y_true_class = y_true.round().astype(int)
        y_pred_class = y_pred_clf.round().astype(int)
        plt.subplot(1, 3, 2)
        cm = confusion_matrix(y_true_class, y_pred_class)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Classification Confusion Matrix")
        plt.subplot(1, 3, 3)
        plt.hist(y_pred_reg, bins=20, alpha=0.7, label="Predicted", density=True)
        plt.hist(y_true, bins=20, alpha=0.7, label="True", density=True)
        plt.xlabel("Star Ratings")
        plt.ylabel("Density")
        plt.title("Prediction vs True Distribution")
        plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    models = load_trained_models()
    if not any(models.values()):
        return None
    df, variety_embeddings, categorical_encoded, brand_encoded = load_processed_data()
    X = prepare_features_for_prediction(
        df, variety_embeddings, categorical_encoded, brand_encoded
    )
    y_true = df["Stars"]
    y_pred_reg = None
    y_pred_clf = None
    if models["regression"]:
        y_pred_reg = predict_star_rating(models, X, "regression")
    if models["classification"]:
        y_pred_clf = predict_star_rating(models, X, "classification")
    results = evaluate_predictions(y_true, y_pred_reg, y_pred_clf, models)
    return models, results


if __name__ == "__main__":
    main()
