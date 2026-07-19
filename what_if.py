import joblib

from data_prep import ucitaj_podatke, pripremi_jednu_nekretninu

PUTANJA_NAPREDNI = "models/advanced_model.pkl"
RMSE_NAPREDNOG_MODELA = 29511


def predvidi_cijenu(model, df, nekretnina):
    _, X_nova = pripremi_jednu_nekretninu(df, nekretnina)
    return model.predict(X_nova)[0]


def predvidi_cijenu_sa_intervalom(model, df, nekretnina, nivo_pouzdanosti=1.96):
    predikcija = predvidi_cijenu(model, df, nekretnina)
    opseg = nivo_pouzdanosti * RMSE_NAPREDNOG_MODELA

    return {
        "predikcija": predikcija,
        "donja_granica": predikcija - opseg,
        "gornja_granica": predikcija + opseg,
    }


def simuliraj_izmjenu(model, df, polazna_nekretnina, izmjene):
    nova_nekretnina = {**polazna_nekretnina, **izmjene}

    cijena_prije = predvidi_cijenu(model, df, polazna_nekretnina)
    cijena_poslije = predvidi_cijenu(model, df, nova_nekretnina)

    razlika = cijena_poslije - cijena_prije
    razlika_procenat = (razlika / cijena_prije) * 100

    return {
        "cijena_prije": cijena_prije,
        "cijena_poslije": cijena_poslije,
        "razlika_dolari": razlika,
        "razlika_procenat": razlika_procenat,
    }


if __name__ == "__main__":
    df = ucitaj_podatke()
    model = joblib.load(PUTANJA_NAPREDNI)

    moja_nekretnina = {
        "povrsina": 1500,
        "broj_soba": 6,
        "broj_kupatila": 2.0,
        "godina_izgradnje": 2005,
        "lokacija": "NAmes",
        "kvalitet_stanje": 6,
        "garaza": 2,
        "velicina_placa": 9000,
    }

    print("=== PREDIKCIJA SA INTERVALOM ===")
    rezultat = predvidi_cijenu_sa_intervalom(model, df, moja_nekretnina)
    print(f"Procijenjena cijena: ${rezultat['predikcija']:,.0f}")
    print(f"Interval (95%): ${rezultat['donja_granica']:,.0f} - ${rezultat['gornja_granica']:,.0f}")

    print("\n=== WHAT-IF: dodavanje jos jednog kupatila ===")
    rez1 = simuliraj_izmjenu(model, df, moja_nekretnina, {"broj_kupatila": 3.0})
    print(f"Prije:   ${rez1['cijena_prije']:,.0f}")
    print(f"Poslije: ${rez1['cijena_poslije']:,.0f}")
    print(f"Razlika: ${rez1['razlika_dolari']:,.0f} ({rez1['razlika_procenat']:+.1f}%)")

    print("\n=== WHAT-IF: renoviranje (kvalitet/stanje 6 -> 9) ===")
    rez2 = simuliraj_izmjenu(model, df, moja_nekretnina, {"kvalitet_stanje": 9})
    print(f"Prije:   ${rez2['cijena_prije']:,.0f}")
    print(f"Poslije: ${rez2['cijena_poslije']:,.0f}")
    print(f"Razlika: ${rez2['razlika_dolari']:,.0f} ({rez2['razlika_procenat']:+.1f}%)")

    print("\n=== WHAT-IF: veca garaza (2 -> 3 mjesta) ===")
    rez3 = simuliraj_izmjenu(model, df, moja_nekretnina, {"garaza": 3})
    print(f"Prije:   ${rez3['cijena_prije']:,.0f}")
    print(f"Poslije: ${rez3['cijena_poslije']:,.0f}")
    print(f"Razlika: ${rez3['razlika_dolari']:,.0f} ({rez3['razlika_procenat']:+.1f}%)")

    print("\n=== WHAT-IF: povecanje povrsine (1500 -> 1800 sqft) ===")
    rez4 = simuliraj_izmjenu(model, df, moja_nekretnina, {"povrsina": 1800})
    print(f"Prije:   ${rez4['cijena_prije']:,.0f}")
    print(f"Poslije: ${rez4['cijena_poslije']:,.0f}")
    print(f"Razlika: ${rez4['razlika_dolari']:,.0f} ({rez4['razlika_procenat']:+.1f}%)")