import os
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

PUTANJA_SIROVI_PODACI = "data/ames_housing_raw.csv"

MAPA_KOLONA = {
    "GrLivArea": "povrsina",
    "TotRmsAbvGrd": "broj_soba",
    "YearBuilt": "godina_izgradnje",
    "Neighborhood": "lokacija",
    "OverallQual": "kvalitet_stanje",
    "GarageCars": "garaza",
    "LotArea": "velicina_placa",
    "SalePrice": "cijena",
}


def preuzmi_sirove_podatke():
    if os.path.exists(PUTANJA_SIROVI_PODACI):
        return pd.read_csv(PUTANJA_SIROVI_PODACI)

    dataset = fetch_openml(data_id=42165, as_frame=True, parser="auto")
    df = dataset.frame

    os.makedirs("data", exist_ok=True)
    df.to_csv(PUTANJA_SIROVI_PODACI, index=False)
    return df


def ucitaj_podatke():
    df = preuzmi_sirove_podatke()
    df = df.dropna(subset=["SalePrice"]).copy()

    df["broj_kupatila"] = df["FullBath"].fillna(0) + 0.5 * df["HalfBath"].fillna(0)
    df["GarageCars"] = df["GarageCars"].fillna(0)

    kolone_original = list(MAPA_KOLONA.keys())
    df_cist = df[kolone_original + ["broj_kupatila"]].rename(columns=MAPA_KOLONA)
    df_cist = df_cist.dropna()

    df_cist["godina_izgradnje"] = df_cist["godina_izgradnje"].astype(int)
    df_cist["broj_soba"] = df_cist["broj_soba"].astype(int)
    df_cist["garaza"] = df_cist["garaza"].astype(int)

    return df_cist.reset_index(drop=True)


def sve_lokacije(df):
    return sorted(df["lokacija"].unique().tolist())


def pripremi_za_model(df):
    df = df.copy()
    kategorije = sve_lokacije(df)
    df["lokacija"] = pd.Categorical(df["lokacija"], categories=kategorije)

    X = df.drop(columns=["cijena"])
    X = pd.get_dummies(X, columns=["lokacija"], prefix="lokacija")
    y = df["cijena"]

    return X, y


def pripremi_jednu_nekretninu(df, nova_nekretnina):
    nova_kao_red = {**nova_nekretnina, "cijena": 0}
    df_sa_novom = pd.concat([df, pd.DataFrame([nova_kao_red])], ignore_index=True)

    X_sve, _ = pripremi_za_model(df_sa_novom)

    X_postojece = X_sve.iloc[:-1]
    X_nova = X_sve.iloc[[-1]]
    return X_postojece, X_nova


def pripremi_vise_nekretnina(df, nove_nekretnine):
    nove = nove_nekretnine.copy()
    nove["cijena"] = 0

    df_sa_novim = pd.concat([df, nove], ignore_index=True)
    X_sve, _ = pripremi_za_model(df_sa_novim)

    broj_novih = len(nove)
    X_postojece = X_sve.iloc[:-broj_novih]
    X_nove = X_sve.iloc[-broj_novih:]
    return X_postojece, X_nove


def podijeli_podatke(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    df = ucitaj_podatke()
    print("Broj nekretnina u datasetu:", len(df))
    print("\nKolone:", list(df.columns))
    print("\nPrvih 5 redova:")
    print(df.head())
    print("\nOsnovna statistika cijena:")
    print(df["cijena"].describe())

    X, y = pripremi_za_model(df)
    print("\nOblik X nakon one-hot encodinga:", X.shape)

    X_train, X_test, y_train, y_test = podijeli_podatke(X, y)
    print("Trening skup:", X_train.shape[0], "nekretnina")
    print("Test skup:", X_test.shape[0], "nekretnina")