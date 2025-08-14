from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def load_trained_models():
    models_dir = Path("models")
    models = {}
    rf_model_path = models_dir / "multisim_random_forest.pkl"
    if rf_model_path.exists():
        models["random_forest"] = joblib.load(rf_model_path)
    else:
        models["random_forest"] = None
    xgb_model_path = models_dir / "multisim_xgboost.pkl"
    if xgb_model_path.exists():
        models["xgboost"] = joblib.load(xgb_model_path)
    else:
        models["xgboost"] = None
    return models


def load_processed_data():
    processed_dir = Path("data/processed")
    X = pd.read_csv(processed_dir / "multisim_features.csv")
    y = pd.read_csv(processed_dir / "multisim_target.csv")
    y = y.iloc[:, 0]
    return X, y


def predict_multisim(models, X):
    predictions = {}
    for model_name, model in models.items():
        if model is not None:
            try:
                pred = model.predict(X)
                predictions[model_name] = pred
            except Exception:
                predictions[model_name] = None
        else:
            predictions[model_name] = None
    return predictions


def evaluate_predictions(y_true, predictions, models):
    results = {}
    for model_name, pred in predictions.items():
        if pred is not None and models[model_name] is not None:
            accuracy = accuracy_score(y_true, pred)
            print(f"\n{model_name.replace('_', ' ').title()} Performance:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, pred))
            results[model_name] = {"accuracy": accuracy, "predictions": pred}
        else:
            results[model_name] = None
    return results


def main():
    models = load_trained_models()
    if not any(models.values()):
        return None
    X, y_true = load_processed_data()
    predictions = predict_multisim(models, X)
    results = evaluate_predictions(y_true, predictions, models)
    print("\nPrediction analysis completed successfully!")
    return models, results


if __name__ == "__main__":
    main()
