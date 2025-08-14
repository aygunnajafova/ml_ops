from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_processed_data():
    processed_dir = Path("data/processed")
    df = pd.read_csv(processed_dir / "ramen_ratings_cleaned.csv", index_col=0)
    variety_embeddings = pd.read_csv(processed_dir / "ramen_variety_embeddings.csv")
    categorical_encoded = pd.read_csv(
        processed_dir / "ramen_categorical_encoded.csv", index_col=0
    )
    brand_encoded = pd.read_csv(processed_dir / "ramen_brand_encoded.csv", index_col=0)
    with open(processed_dir / "ramen_processing_info.pkl", "rb") as f:
        processing_info = joblib.load(f)
    return df, variety_embeddings, categorical_encoded, brand_encoded, processing_info


def prepare_features(df, variety_embeddings, categorical_encoded, brand_encoded):
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
    y = df["Stars"]
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    return X, y


def train_regression_model(X, y):
    print("Training Regression Model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    n_components = min(200, X_train.shape[1])
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"Applied PCA with {n_components} components")
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Regression Model Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    return model, scaler, pca, X_test_pca, y_test, y_pred, {"mse": mse, "r2": r2}


def train_classification_model(X, y):
    print("Training Classification Model...")
    y_class = y.round().astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    n_components = min(200, X_train.shape[1])
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return clf, scaler, pca, X_test_pca, y_test, y_pred


def save_models(
    reg_model,
    clf_model,
    reg_scaler,
    clf_scaler,
    reg_pca,
    clf_pca,
    model_name_prefix="ramen",
):
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    if reg_model is not None:
        joblib.dump(reg_model, models_dir / f"{model_name_prefix}_regression_model.pkl")
        joblib.dump(
            reg_scaler, models_dir / f"{model_name_prefix}_regression_scaler.pkl"
        )
        joblib.dump(reg_pca, models_dir / f"{model_name_prefix}_regression_pca.pkl")
        print(f"Saved regression model components to {models_dir}")
    if clf_model is not None:
        joblib.dump(
            clf_model, models_dir / f"{model_name_prefix}_classification_model.pkl"
        )
        joblib.dump(
            clf_scaler, models_dir / f"{model_name_prefix}_classification_scaler.pkl"
        )
        joblib.dump(clf_pca, models_dir / f"{model_name_prefix}_classification_pca.pkl")
        print(f"Saved classification model components to {models_dir}")
    return models_dir


def plot_results(y_test_reg, y_pred_reg, y_test_clf, y_pred_clf):
    if y_test_reg is not None and y_pred_reg is not None:
        print("Regression Results:")
        print(f"Prediction range: {y_pred_reg.min():.2f} to {y_pred_reg.max():.2f}")
        print(f"True value range: {y_test_reg.min():.2f} to {y_test_reg.max():.2f}")
    if y_test_clf is not None and y_pred_clf is not None:
        print("\nClassification Results:")
        cm = confusion_matrix(y_test_clf, y_pred_clf)
        print("Confusion Matrix:")
        print(cm)
    print("Results printed (plots removed for headless execution)")


def main():
    df, variety_embeddings, categorical_encoded, brand_encoded, processing_info = (
        load_processed_data()
    )
    print(f"Loaded processed data with shape: {df.shape}")
    X, y = prepare_features(df, variety_embeddings, categorical_encoded, brand_encoded)
    reg_model, reg_scaler, reg_pca, X_test_reg, y_test_reg, y_pred_reg, reg_metrics = (
        train_regression_model(X, y)
    )
    clf_model, clf_scaler, clf_pca, X_test_clf, y_test_clf, y_pred_clf = (
        train_classification_model(X, y)
    )
    plot_results(y_test_reg, y_pred_reg, y_test_clf, y_pred_clf)
    models_dir = save_models(
        reg_model, clf_model, reg_scaler, clf_scaler, reg_pca, clf_pca
    )
    print("\nTraining completed successfully!")
    print(f"Models saved to: {models_dir}")
    return reg_model, clf_model, reg_metrics


if __name__ == "__main__":
    main()
