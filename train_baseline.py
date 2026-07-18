# train_baseline.py
# -----------------------------------------------------------
# KORAK 2: Treniranje baseline modela - Linear Regression
#
# Linear Regression je najjednostavniji regresioni model - pokušava da
# nađe pravu liniju (u više dimenzija: ravan) koja najbolje opisuje vezu
# između karakteristika kuće i njene cijene. Koristimo ga kao "baseline"
# (početnu tačku) da bismo kasnije mogli da uporedimo koliko je bolji
# napredniji model (XGBoost, korak 3).
# -----------------------------------------------------------

import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_prep import ucitaj_podatke, pripremi_za_model, podijeli_podatke

PUTANJA_MODELA = "models/baseline_model.pkl"


def treniraj_baseline_model(X_train, y_train):
    """Trenira Linear Regression model na trening podacima."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluiraj_model(model, X_test, y_test, naziv="Model"):
    """
    Računa standardne metrike za regresiju i ispisuje ih:
      - MAE  (Mean Absolute Error)      - prosječna greška u dolarima, lako se razumije
      - RMSE (Root Mean Squared Error)  - kažnjava velike greške više nego MAE
      - R2   (R-squared)                - koliki dio varijacije cijene model objasni (1.0 = savršeno)

    Ova funkcija je napravljena tako da je koristimo i za baseline i za
    napredni model (korak 3), da bismo ih mogli fer uporediti istom metodom.
    """
    predikcije = model.predict(X_test)

    mae = mean_absolute_error(y_test, predikcije)
    rmse = np.sqrt(mean_squared_error(y_test, predikcije))
    r2 = r2_score(y_test, predikcije)

    print(f"\n--- {naziv} ---")
    print(f"MAE  (prosjecna greska):      ${mae:,.0f}")
    print(f"RMSE (greska sa kaznom):      ${rmse:,.0f}")
    print(f"R2   (objasnjena varijansa):  {r2:.3f}")

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