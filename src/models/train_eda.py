from pathlib import Path

import pandas as pd


def load_data():
    data_path = Path("data/HousingPrices-Amsterdam-August-2021.csv")
    data = pd.read_csv(data_path)
    return data


def perform_eda(data):
    print(f"The data has rows - {data.shape[0]} and columns - {data.shape[1]}")
    print("\nBelow are the data types")
    print(data.dtypes)
    print("\nBasic statistics:")
    print(data.describe())
    cols = data.columns
    for col in cols[1:]:
        if data[col].dtype != "object":
            print(f"\nFor key - {col}")
            print(
                f"Mean - {data[col].mean()}, Median - {data[col].median()}, Mode - {data[col].mode()[0]}"
            )
    categorical_cols = ["Address", "Zip"]
    for col_name in categorical_cols:
        print(f"\nDescription of col - {col_name}")
        print(data[col_name].describe())
    return data


def create_visualizations(data):
    plot_cols = data.columns.tolist()[1:]
    quant_cols = [col for col in plot_cols if data[col].dtype != "object"]
    for col in quant_cols:
        q25 = data[col].quantile(0.25)
        q50 = data[col].quantile(0.50)
        q75 = data[col].quantile(0.75)
        print(
            f"  25th percentile (Q1): {q25}, 50th percentile (Median): {q50}, 75th percentile (Q3): {q75}"
        )
    print("Statistical summaries printed (plots removed for headless execution)")


def generate_report(data):
    report = "EDA Report - Amsterdam Housing Prices"
    print(report)
    return report


def main():
    data = load_data()
    data = perform_eda(data)
    report = generate_report(data)
    print("EDA completed successfully!")
    return data, report


if __name__ == "__main__":
    main()
