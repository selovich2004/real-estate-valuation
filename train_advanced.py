# train_advanced.py
# -----------------------------------------------------------
# KORAK 3: Treniranje naprednog modela (XGBoost) i poređenje sa baseline-om
#
# XGBoost (Extreme Gradient Boosting) je model sastavljen od MNOGO malih
# odlučujućih stabala (decision trees) koja se treniraju jedno za drugim -
# svako sledeće stablo pokušava da ispravi greške prethodnih. Zato obično
# bolje "hvata" složene, nelinearne veze u podacima nego Linear Regression
# (koja može da nacrta samo pravu liniju/ravan).
# -----------------------------------------------------------

import os
import joblib
import pandas as pd
from xgboost import XGBRegressor

from data_prep import ucitaj_podatke, pripremi_za_model, podijeli_podatke
from train_baseline import evaluiraj_model, PUTANJA_MODELA as PUTANJA_BASELINE

PUTANJA_MODELA = "models/advanced_model.pkl"


def treniraj_napredni_model(X_train, y_train):
    """
    Trenira XGBoost model.

    Parametri:
      - n_estimators=300   -> koliko stabala model pravi (jedno za drugim)
      - max_depth=4         -> koliko je svako stablo "duboko" (sprječava overfitting)
      - learning_rate=0.05  -> koliko brzo model uči (manje = sporije, ali stabilnije)
      - random_state=42     -> da rezultat uvijek bude isti kad ponovo pokrenemo kod
    """
    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    df = ucitaj_podatke()
    X, y = pripremi_za_model(df)
    X_train, X_test, y_train, y_test = podijeli_podatke(X, y)

    model = treniraj_napredni_model(X_train, y_train)
    metrike_napredni = evaluiraj_model(model, X_test, y_test, naziv="XGBoost (napredni model)")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, PUTANJA_MODELA)
    print(f"\nModel sacuvan u: {PUTANJA_MODELA}")

    # --- Poredjenje sa baseline modelom ---
    baseline_model = joblib.load(PUTANJA_BASELINE)
    metrike_baseline = evaluiraj_model(baseline_model, X_test, y_test, naziv="Linear Regression (baseline)")

    poredjenje = pd.DataFrame({
        "Linear Regression": metrike_baseline,
        "XGBoost": metrike_napredni,
    })

    print("\n=== POREDJENJE MODELA ===")
    print(poredjenje)

    if metrike_napredni["R2"] > metrike_baseline["R2"]:
        print("\nZakljucak: XGBoost bolje objasnjava varijaciju cijena (visi R2).")
    else:
        print("\nZakljucak: Linear Regression je uporediv ili bolji od XGBoost-a na ovom datasetu.")