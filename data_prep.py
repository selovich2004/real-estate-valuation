# data_prep.py
# -----------------------------------------------------------
# KORAK 1: Učitavanje i priprema podataka
#
# Ovaj fajl:
#   1. Preuzima Ames Housing dataset (podaci o prodatim kućama u gradu Ames, Iowa)
#      sa OpenML servisa (ako već nije preuzet ranije, koristi lokalnu kopiju)
#   2. Bira samo kolone koje su nam potrebne za projekat i preimenuje ih na naš jezik
#   3. Čisti nedostajuće vrijednosti
#   4. Priprema podatke u format pogodan za treniranje modela (one-hot encoding za lokaciju)
# -----------------------------------------------------------

import os
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Putanja gdje čuvamo lokalnu kopiju sirovog dataseta (da ga ne preuzimamo svaki put)
PUTANJA_SIROVI_PODACI = "data/ames_housing_raw.csv"

# Mapa: originalno ime kolone u datasetu -> naziv koji mi koristimo u projektu
# Ovo radimo da bi kod bio čitljiviji (npr. "GrLivArea" -> "povrsina")
MAPA_KOLONA = {
    "GrLivArea": "povrsina",          # stambena površina u kvadratnim stopama
    "TotRmsAbvGrd": "broj_soba",      # broj soba iznad prizemlja (bez kupatila)
    "YearBuilt": "godina_izgradnje",
    "Neighborhood": "lokacija",       # naziv gradske četvrti (kvarta)
    "OverallQual": "kvalitet_stanje", # ocjena kvaliteta/stanja objekta, skala 1-10
    "GarageCars": "garaza",           # koliko automobila staje u garažu
    "LotArea": "velicina_placa",      # veličina cijele parcele u kvadratnim stopama
    "SalePrice": "cijena",            # ciljna varijabla koju predviđamo
}

# Kolone iz kojih računamo broj kupatila (posebno se ne koriste dalje)
KOLONE_KUPATILA = ["FullBath", "HalfBath"]


def preuzmi_sirove_podatke():
    """
    Preuzima Ames Housing dataset sa OpenML-a (data_id=42165).
    Ako je dataset već jednom preuzet i sačuvan lokalno, koristi tu kopiju
    umjesto da ponovo skida sa interneta (brže i radi bez interneta na svakom pokretanju).
    """
    if os.path.exists(PUTANJA_SIROVI_PODACI):
        return pd.read_csv(PUTANJA_SIROVI_PODACI)

    print("Preuzimam Ames Housing dataset sa OpenML-a (samo prvi put)...")
    dataset = fetch_openml(data_id=42165, as_frame=True, parser="auto")
    df = dataset.frame

    os.makedirs("data", exist_ok=True)
    df.to_csv(PUTANJA_SIROVI_PODACI, index=False)
    return df


def ucitaj_podatke():
    """
    Glavna funkcija koja vraća OČIŠĆEN dataframe sa samo onim kolonama
    koje su nam potrebne, preimenovanim na srpski/crnogorski.
    """
    df = preuzmi_sirove_podatke()

    # Neki redovi mogu imati praznu ciljnu vrijednost (cijenu) - takve redove
    # ne možemo koristiti za treniranje pa ih izbacujemo
    df = df.dropna(subset=["SalePrice"]).copy()

    # Broj kupatila = broj punih kupatila + pola za svako pomoćno (WC bez tuša/kade)
    df["broj_kupatila"] = df["FullBath"].fillna(0) + 0.5 * df["HalfBath"].fillna(0)

    # Garaža - ako kuća nema garažu, vrijednost je prazna (NaN), što znači 0 mjesta
    df["GarageCars"] = df["GarageCars"].fillna(0)

    # Biramo samo kolone koje nas zanimaju i preimenujemo ih
    kolone_original = list(MAPA_KOLONA.keys())
    df_cist = df[kolone_original + ["broj_kupatila"]].rename(columns=MAPA_KOLONA)

    # Uklanjamo redove kojima nedostaje neka od preostalih vrijednosti (ima ih jako malo)
    df_cist = df_cist.dropna()

    # Tipovi podataka - godina i broj soba su cijeli brojevi radi ljepšeg prikaza
    df_cist["godina_izgradnje"] = df_cist["godina_izgradnje"].astype(int)
    df_cist["broj_soba"] = df_cist["broj_soba"].astype(int)
    df_cist["garaza"] = df_cist["garaza"].astype(int)

    df_cist = df_cist.reset_index(drop=True)
    return df_cist


def sve_lokacije(df):
    """Vraća listu svih mogućih lokacija (kvartova) u datasetu, abecednim redom."""
    return sorted(df["lokacija"].unique().tolist())


def pripremi_za_model(df):
    """
    Priprema podatke za mašinsko učenje:
      - X = ulazne karakteristike (sve kolone osim cijene)
      - y = cijena (ono što predviđamo)

    Lokacija je tekstualna kolona (npr. "NAmes", "CollgCr"...) pa je model
    ne može direktno koristiti. Zato pravimo "one-hot encoding": za svaku
    moguću lokaciju napravimo posebnu kolonu sa vrijednošću 0 ili 1
    (npr. kolona "lokacija_NAmes" = 1 ako je kuća u tom kvartu, inače 0).

    Kategorije lokacije fiksiramo unaprijed (cat_kategorije) da bi kolone
    uvijek bile iste, čak i kada kasnije budemo pravili predikciju za samo
    JEDNU novu kuću (bez toga bi pd.get_dummies napravio samo jednu kolonu).
    """
    df = df.copy()
    kategorije = sve_lokacije(df)
    df["lokacija"] = pd.Categorical(df["lokacija"], categories=kategorije)

    X = df.drop(columns=["cijena"])
    X = pd.get_dummies(X, columns=["lokacija"], prefix="lokacija")
    y = df["cijena"]

    return X, y


def pripremi_jednu_nekretninu(df, nova_nekretnina):
    """
    Pretvara JEDNU novu nekretninu (rječnik sa karakteristikama, npr. iz forme
    ili what-if simulatora) u isti numerički format (one-hot encoding) kao
    postojeće podatke, da bi model/KNN mogli da je koriste.

    Trik: privremeno dodamo novu nekretninu kao dodatni red postojećem
    datasetu i zajedno ih prevedemo u brojeve (pripremi_za_model) - tako smo
    sigurni da će kolone (npr. lokacija_NAmes, lokacija_CollgCr...) biti
    IDENTIČNE kao kod originalnih podataka, čak i za samo jednu novu kuću.

    Vraća:
      - X_postojece: karakteristike svih postojećih nekretnina (kao i prije)
      - X_nova: karakteristike nove nekretnine, u istom formatu (1 red)
    """
    nova_kao_red = {**nova_nekretnina, "cijena": 0}  # cijena je ovdje samo placeholder
    df_sa_novom = pd.concat([df, pd.DataFrame([nova_kao_red])], ignore_index=True)

    X_sve, _ = pripremi_za_model(df_sa_novom)

    X_postojece = X_sve.iloc[:-1]
    X_nova = X_sve.iloc[[-1]]
    return X_postojece, X_nova


def pripremi_vise_nekretnina(df, nove_nekretnine):
    """
    Isto kao pripremi_jednu_nekretninu, ali za VIŠE novih nekretnina
    odjednom (npr. učitanih iz CSV/Excel fajla pri uploadu).

    nove_nekretnine: DataFrame sa istim kolonama kao originalni podaci
                     (bez kolone "cijena")

    Vraća:
      - X_postojece: karakteristike svih postojećih nekretnina
      - X_nove: karakteristike svih novih nekretnina, u istom formatu
    """
    nove = nove_nekretnine.copy()
    nove["cijena"] = 0  # placeholder, nije bitno za pripremu karakteristika

    df_sa_novim = pd.concat([df, nove], ignore_index=True)
    X_sve, _ = pripremi_za_model(df_sa_novim)

    broj_novih = len(nove)
    X_postojece = X_sve.iloc[:-broj_novih]
    X_nove = X_sve.iloc[-broj_novih:]
    return X_postojece, X_nove


def podijeli_podatke(X, y, test_size=0.2, random_state=42):
    """
    Dijeli podatke na trening (za učenje modela) i test skup (za provjeru
    koliko je model dobar na podacima koje nije vidio).
    random_state=42 fiksiramo da bi podjela uvijek bila ista - to je bitno
    da bismo mogli fer da uporedimo baseline i napredni model na ISTOM test skupu.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Ovo se izvršava samo ako pokrenemo direktno ovaj fajl (python data_prep.py)
# Korisno za brzu provjeru da li sve radi kako treba
if __name__ == "__main__":
    df = ucitaj_podatke()
    print("Broj nekretnina u datasetu:", len(df))
    print("\nKolone:", list(df.columns))
    print("\nPrvih 5 redova:")
    print(df.head())
    print("\nOsnovna statistika cijena:")
    print(df["cijena"].describe())

    X, y = pripremi_za_model(df)
    print("\nOblik X (karakteristike) nakon one-hot encodinga:", X.shape)

    X_train, X_test, y_train, y_test = podijeli_podatke(X, y)
    print("Trening skup:", X_train.shape[0], "nekretnina")
    print("Test skup:", X_test.shape[0], "nekretnina")