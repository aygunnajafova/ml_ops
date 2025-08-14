from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier


def load_processed_data():
    processed_dir = Path("data/processed")
    X = pd.read_csv(processed_dir / "multisim_features.csv")
    y = pd.read_csv(processed_dir / "multisim_target.csv")
    y = y.iloc[:, 0]

    sample_size = min(5000, len(X))
    X = X.head(sample_size)
    y = y.head(sample_size)

    print(f"Loaded features with shape: {X.shape}")
    print(f"Loaded target with shape: {y.shape}")
    print(f"Using {sample_size} records for training to reduce computation time")
    return X, y


def train_random_forest(X, y):
    print("Training Random Forest Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    return rf_model, X_test, y_test, y_pred, accuracy


def train_xgboost(X, y):
    print("Training XGBoost Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    xgb_model = XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Base Accuracy: {accuracy:.4f}")
    print("Performing hyperparameter tuning...")
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "n_estimators": [50, 100, 200],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
    }
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=3,
        verbose=1,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    accuracy_best = accuracy_score(y_test, y_pred)
    print(f"XGBoost Tuned Accuracy: {accuracy_best:.4f}")
    return best_model, X_test, y_test, y_pred_best, accuracy_best


def save_models(rf_model, xgb_model, model_name_prefix="multisim"):
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    if rf_model is not None:
        joblib.dump(rf_model, models_dir / f"{model_name_prefix}_random_forest.pkl")
        print(
            f"Saved Random Forest model to: {models_dir / f'{model_name_prefix}_random_forest.pkl'}"
        )
    if xgb_model is not None:
        joblib.dump(xgb_model, models_dir / f"{model_name_prefix}_xgboost.pkl")
        print(
            f"Saved XGBoost model to: {models_dir / f'{model_name_prefix}_xgboost.pkl'}"
        )
    return models_dir


def plot_results(y_test_rf, y_pred_rf, y_test_xgb, y_pred_xgb):
    if y_test_rf is not None and y_pred_rf is not None:
        print("Random Forest Confusion Matrix:")
        cm_rf = confusion_matrix(y_test_rf, y_pred_rf)
        print(cm_rf)
        print("\nXGBoost Confusion Matrix:")
        cm_xgb = confusion_matrix(y_test_xgb, y_pred_xgb)
        print(cm_xgb)
        print(f"\nRandom Forest Accuracy: {accuracy_score(y_test_rf, y_pred_rf):.3f}")
        print(f"XGBoost Accuracy: {accuracy_score(y_test_xgb, y_pred_xgb):.3f}")
    print("Results printed (plots removed for headless execution)")


def main():
    X, y = load_processed_data()
    rf_model, X_test_rf, y_test_rf, y_pred_rf, rf_accuracy = train_random_forest(X, y)
    xgb_model, X_test_xgb, y_test_xgb, y_pred_xgb, xgb_accuracy = train_xgboost(X, y)
    plot_results(y_test_rf, y_pred_rf, y_test_xgb, y_pred_xgb)
    models_dir = save_models(rf_model, xgb_model)

    print("\nTraining completed successfully!")

    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
    print(f"Models saved to: {models_dir}")
    return rf_model, xgb_model, rf_accuracy, xgb_accuracy


if __name__ == "__main__":
    main()
