from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from data_prep import ucitaj_podatke, pripremi_jednu_nekretninu

KOLONE_ZA_PRIKAZ = [
    "povrsina", "broj_soba", "broj_kupatila", "godina_izgradnje",
    "lokacija", "kvalitet_stanje", "garaza", "velicina_placa", "cijena",
]


def pronadji_slicne_nekretnine(df, nova_nekretnina, k=5):
    X_postojece, X_nova = pripremi_jednu_nekretninu(df, nova_nekretnina)

    scaler = StandardScaler()
    X_postojece_skalirano = scaler.fit_transform(X_postojece)
    X_nova_skalirano = scaler.transform(X_nova)

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_postojece_skalirano)

    udaljenosti, indeksi = knn.kneighbors(X_nova_skalirano)

    slicne_nekretnine = df.iloc[indeksi[0]].copy()
    slicne_nekretnine["udaljenost"] = udaljenosti[0]

    return slicne_nekretnine.sort_values("udaljenost")[KOLONE_ZA_PRIKAZ + ["udaljenost"]]


if __name__ == "__main__":
    df = ucitaj_podatke()

    primjer_nekretnine = {
        "povrsina": 1500,
        "broj_soba": 6,
        "broj_kupatila": 2.0,
        "godina_izgradnje": 2005,
        "lokacija": "NAmes",
        "kvalitet_stanje": 6,
        "garaza": 2,
        "velicina_placa": 9000,
    }

    slicne = pronadji_slicne_nekretnine(df, primjer_nekretnine, k=5)
    print("5 najslicnijih vec prodatih nekretnina:\n")
    print(slicne.to_string(index=False))