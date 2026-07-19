import joblib
import pandas as pd

from data_prep import ucitaj_podatke, pripremi_za_model

PUTANJA_BASELINE = "models/baseline_model.pkl"
PUTANJA_NAPREDNI = "models/advanced_model.pkl"


def _grupisi_lokacije(serija):
    lokacije = serija[serija.index.str.startswith("lokacija_")]
    ostalo = serija[~serija.index.str.startswith("lokacija_")]
    return pd.concat([ostalo, pd.Series({"lokacija": lokacije.sum()})])


def xgboost_znacaj_atributa(model, feature_names):
    znacaj = pd.Series(model.feature_importances_, index=feature_names)
    return _grupisi_lokacije(znacaj).sort_values(ascending=False)


def linear_koeficijenti(model, feature_names):
    koeficijenti = pd.Series(model.coef_, index=feature_names)
    koeficijenti = koeficijenti[~koeficijenti.index.str.startswith("lokacija_")]

    tabela = koeficijenti.to_frame("uticaj_na_cijenu")
    tabela["smjer"] = tabela["uticaj_na_cijenu"].apply(
        lambda x: "povecava cijenu" if x > 0 else "smanjuje cijenu"
    )
    redoslijed = tabela["uticaj_na_cijenu"].abs().sort_values(ascending=False).index
    return tabela.reindex(redoslijed)


def najbolje_i_najgore_lokacije(model, feature_names, top_n=5):
    koeficijenti = pd.Series(model.coef_, index=feature_names)
    lokacije = koeficijenti[koeficijenti.index.str.startswith("lokacija_")]
    lokacije.index = lokacije.index.str.replace("lokacija_", "", regex=False)
    lokacije = lokacije.sort_values(ascending=False)

    return lokacije.head(top_n), lokacije.tail(top_n)


if __name__ == "__main__":
    df = ucitaj_podatke()
    X, y = pripremi_za_model(df)
    feature_names = X.columns

    xgb_model = joblib.load(PUTANJA_NAPREDNI)
    lr_model = joblib.load(PUTANJA_BASELINE)

    print("=== ZNACAJ ATRIBUTA (XGBoost) ===")
    print(xgboost_znacaj_atributa(xgb_model, feature_names))

    print("\n=== UTICAJ ATRIBUTA NA CIJENU (Linear Regression) ===")
    print(linear_koeficijenti(lr_model, feature_names))

    najbolje, najgore = najbolje_i_najgore_lokacije(lr_model, feature_names)
    print("\n=== LOKACIJE KOJE NAJVISE POVECAVAJU CIJENU ===")
    print(najbolje)
    print("\n=== LOKACIJE KOJE NAJVISE SMANJUJU CIJENU ===")
    print(najgore)