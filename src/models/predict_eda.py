from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_trained_model():
    models_dir = Path("models")
    model = joblib.load(models_dir / "housing_price_model.pkl")
    scaler = joblib.load(models_dir / "housing_price_scaler.pkl")
    return model, scaler


def load_test_data():
    data_path = Path("data/HousingPrices-Amsterdam-August-2021.csv")
    data = pd.read_csv(data_path)
    data_clean = data.dropna()
    X = data_clean[["Area", "Room"]]
    y = data_clean["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test, y_test


def predict_price(area, room, model, scaler):
    X_input = np.array([[area, room]])
    X_scaled = scaler.transform(X_input)
    predicted_price = model.predict(X_scaled)[0]
    return predicted_price


def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Performance on Test Data:")
    print(f"MSE: {mse:.2f}")
    print(f"R² Score: {r2:.3f}")

    return y_pred


def main():
    model, scaler = load_trained_model()

    X_test, y_test = load_test_data()

    evaluate_model(model, scaler, X_test, y_test)

    sample_areas = [50, 100, 150]
    sample_rooms = [2, 3, 4]

    for area, room in zip(sample_areas, sample_rooms):
        predict_price(area, room, model, scaler)

    print("\nPrediction completed successfully!")
    return model, scaler


if __name__ == "__main__":
    main()
