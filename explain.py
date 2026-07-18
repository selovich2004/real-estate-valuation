# explain.py
# -----------------------------------------------------------
# KORAK 5: Feature importance - koji atributi utiču na cijenu
#
# Ovaj fajl objašnjava DVIJE stvari:
#   1. Koliko je svaka karakteristika VAŽNA za XGBoost model (feature
#      importance) - ali XGBoost ne kaže da li je uticaj pozitivan ili
#      negativan, samo koliko je "bitan".
#   2. Da li svaka karakteristika cijenu POVEĆAVA ili SMANJUJE - to čitamo
#      iz koeficijenata Linear Regression modela (pozitivan koeficijent =
#      povećava cijenu, negativan = smanjuje).
# -----------------------------------------------------------

import joblib
import pandas as pd

from data_prep import ucitaj_podatke, pripremi_za_model

PUTANJA_BASELINE = "models/baseline_model.pkl"
PUTANJA_NAPREDNI = "models/advanced_model.pkl"


def _grupisi_lokacije(serija):
    """
    Pomoćna funkcija: sve pojedinačne lokacija_* kolone (jedna po kvartu,
    ima ih ~25) spaja u JEDAN red "lokacija" (zbir), da bi grafikon bio
    čitljiv umjesto da ima 25 sitnih stubića za svaku pojedinačnu lokaciju.
    """
    lokacije = serija[serija.index.str.startswith("lokacija_")]
    ostalo = serija[~serija.index.str.startswith("lokacija_")]
    return pd.concat([ostalo, pd.Series({"lokacija": lokacije.sum()})])


def xgboost_znacaj_atributa(model, feature_names):
    """
    Vraća Series sa značajem (importance) svake karakteristike za XGBoost
    model - veći broj znači da tu karakteristiku model "češće koristi"
    kada pravi predikciju.
    """
    znacaj = pd.Series(model.feature_importances_, index=feature_names)
    return _grupisi_lokacije(znacaj).sort_values(ascending=False)


def linear_koeficijenti(model, feature_names):
    """
    Vraća tabelu sa koeficijentima Linear Regression modela za sve
    NE-lokacijske karakteristike, sa kolonom "smjer" koja kaže da li ta
    karakteristika povećava ili smanjuje cijenu (i za koliko, po jedinici).
    """
    koeficijenti = pd.Series(model.coef_, index=feature_names)
    koeficijenti = koeficijenti[~koeficijenti.index.str.startswith("lokacija_")]

    tabela = koeficijenti.to_frame("uticaj_na_cijenu")
    tabela["smjer"] = tabela["uticaj_na_cijenu"].apply(
        lambda x: "povecava cijenu" if x > 0 else "smanjuje cijenu"
    )
    redoslijed = tabela["uticaj_na_cijenu"].abs().sort_values(ascending=False).index
    return tabela.reindex(redoslijed)


def najbolje_i_najgore_lokacije(model, feature_names, top_n=5):
    """
    Iz Linear Regression koeficijenata izdvaja koje lokacije (kvartovi)
    najviše povećavaju, a koje najviše smanjuju cijenu u odnosu na prosjek.
    """
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

    print("\n=== UTICAJ ATRIBUTA NA CIJENU (Linear Regression koeficijenti) ===")
    print(linear_koeficijenti(lr_model, feature_names))

    najbolje, najgore = najbolje_i_najgore_lokacije(lr_model, feature_names)
    print("\n=== LOKACIJE KOJE NAJVISE POVECAVAJU CIJENU (u odnosu na prosjek) ===")
    print(najbolje)
    print("\n=== LOKACIJE KOJE NAJVISE SMANJUJU CIJENU (u odnosu na prosjek) ===")
    print(najgore)