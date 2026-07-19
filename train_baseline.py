import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_prep import ucitaj_podatke, pripremi_za_model, podijeli_podatke

PUTANJA_MODELA = "models/baseline_model.pkl"


def treniraj_baseline_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluiraj_model(model, X_test, y_test, naziv="Model"):
    predikcije = model.predict(X_test)

    mae = mean_absolute_error(y_test, predikcije)
    rmse = np.sqrt(mean_squared_error(y_test, predikcije))
    r2 = r2_score(y_test, predikcije)

    print(f"\n--- {naziv} ---")
    print(f"MAE:  ${mae:,.0f}")
    print(f"RMSE: ${rmse:,.0f}")
    print(f"R2:   {r2:.3f}")

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


if __name__ == "__main__":
    df = ucitaj_podatke()
    X, y = pripremi_za_model(df)
    X_train, X_test, y_train, y_test = podijeli_podatke(X, y)

    model = treniraj_baseline_model(X_train, y_train)
    metrike = evaluiraj_model(model, X_test, y_test, naziv="Linear Regression (baseline)")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, PUTANJA_MODELA)
    print(f"\nModel sacuvan u: {PUTANJA_MODELA}")