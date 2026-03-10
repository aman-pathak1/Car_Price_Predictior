import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os


def train_and_save_model(data_path="car_data.csv", model_path="car_price_model.pkl"):
    """Train the car price prediction model and save it."""
    df = pd.read_csv(data_path)

    df["Car_Age"] = 2024 - df["Year"]

    le_fuel = LabelEncoder()
    le_seller = LabelEncoder()
    le_transmission = LabelEncoder()

    df["Fuel_Type_encoded"] = le_fuel.fit_transform(df["Fuel_Type"])
    df["Seller_Type_encoded"] = le_seller.fit_transform(df["Seller_Type"])
    df["Transmission_encoded"] = le_transmission.fit_transform(df["Transmission"])

    feature_cols = [
        "Car_Age",
        "Present_Price",
        "Kms_Driven",
        "Fuel_Type_encoded",
        "Seller_Type_encoded",
        "Transmission_encoded",
        "Owner",
    ]
    X = df[feature_cols]
    y = df["Selling_Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    encoders = {
        "fuel_type": le_fuel,
        "seller_type": le_seller,
        "transmission": le_transmission,
    }

    model_data = {"model": model, "encoders": encoders, "feature_cols": feature_cols}

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    score = model.score(X_test, y_test)
    return model_data, score


def load_model(model_path="car_price_model.pkl"):
    """Load the pre-trained model."""
    if not os.path.exists(model_path):
        model_data, _ = train_and_save_model()
        return model_data

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data


def predict_price(
    model_data,
    car_age,
    present_price,
    kms_driven,
    fuel_type,
    seller_type,
    transmission,
    owner,
):
    """Predict car price given features."""
    model = model_data["model"]
    encoders = model_data["encoders"]

    try:
        fuel_encoded = encoders["fuel_type"].transform([fuel_type])[0]
        seller_encoded = encoders["seller_type"].transform([seller_type])[0]
        trans_encoded = encoders["transmission"].transform([transmission])[0]
    except ValueError as e:
        raise ValueError(
            f"Invalid input value: {e}. "
            "Accepted fuel types: Petrol, Diesel, CNG. "
            "Accepted seller types: Dealer, Individual. "
            "Accepted transmissions: Manual, Automatic."
        ) from e

    features = pd.DataFrame(
        [
            {
                "Car_Age": car_age,
                "Present_Price": present_price,
                "Kms_Driven": kms_driven,
                "Fuel_Type_encoded": fuel_encoded,
                "Seller_Type_encoded": seller_encoded,
                "Transmission_encoded": trans_encoded,
                "Owner": owner,
            }
        ]
    )

    predicted_price = model.predict(features)[0]
    return max(0, predicted_price)


if __name__ == "__main__":
    model_data, score = train_and_save_model()
    print(f"Model trained successfully! R² Score: {score:.4f}")
