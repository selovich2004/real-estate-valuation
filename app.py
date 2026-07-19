import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from data_prep import (
    ucitaj_podatke,
    sve_lokacije,
    pripremi_za_model,
    podijeli_podatke,
    pripremi_vise_nekretnina,
)
from what_if import predvidi_cijenu_sa_intervalom, simuliraj_izmjenu
from similar_homes import pronadji_slicne_nekretnine
from explain import xgboost_znacaj_atributa, linear_koeficijenti
from train_baseline import evaluiraj_model

st.set_page_config(page_title="Procjena vrijednosti nekretnina", page_icon="🏠", layout="wide")

PUTANJA_NAPREDNI = "models/advanced_model.pkl"
PUTANJA_BASELINE = "models/baseline_model.pkl"
SQFT_U_M2 = 0.092903


@st.cache_data
def ucitaj_sve_podatke():
    return ucitaj_podatke()


@st.cache_resource
def ucitaj_napredni_model():
    return joblib.load(PUTANJA_NAPREDNI)


@st.cache_resource
def ucitaj_baseline_model():
    return joblib.load(PUTANJA_BASELINE)


df = ucitaj_sve_podatke()
model = ucitaj_napredni_model()
baseline_model = ucitaj_baseline_model()
lokacije = sve_lokacije(df)

X, y = pripremi_za_model(df)
_, X_test, _, y_test = podijeli_podatke(X, y)

st.title("🏠 Procjena vrijednosti stambenih nekretnina")
st.caption("Ames Housing dataset - Linear Regression & XGBoost")

st.header("Unesi karakteristike nekretnine")

col1, col2 = st.columns(2)

with col1:
    povrsina = st.number_input(
        "Povrsina (kvadratne stope)", min_value=200, max_value=6000, value=1500, step=50
    )
    broj_soba = st.number_input("Broj soba (iznad prizemlja)", min_value=1, max_value=15, value=6)
    broj_kupatila = st.number_input("Broj kupatila", min_value=0.0, max_value=6.0, value=2.0, step=0.5)
    godina_izgradnje = st.number_input("Godina izgradnje", min_value=1870, max_value=2026, value=2005)

with col2:
    lokacija = st.selectbox("Lokacija (kvart)", lokacije)
    kvalitet_stanje = st.slider("Kvalitet/stanje objekta (1 = lose, 10 = odlicno)", 1, 10, 6)
    garaza = st.number_input("Garaza (broj mjesta)", min_value=0, max_value=5, value=2)
    velicina_placa = st.number_input(
        "Velicina placa (kvadratne stope)", min_value=500, max_value=50000, value=9000, step=100
    )

nekretnina = {
    "povrsina": povrsina,
    "broj_soba": broj_soba,
    "broj_kupatila": broj_kupatila,
    "godina_izgradnje": godina_izgradnje,
    "lokacija": lokacija,
    "kvalitet_stanje": kvalitet_stanje,
    "garaza": garaza,
    "velicina_placa": velicina_placa,
}

if st.button("Procijeni cijenu", type="primary"):
    rezultat = predvidi_cijenu_sa_intervalom(model, df, nekretnina)

    povrsina_m2 = povrsina * SQFT_U_M2
    cijena_po_m2 = rezultat["predikcija"] / povrsina_m2

    st.header("Rezultat procjene")

    kol1, kol2, kol3 = st.columns(3)
    kol1.metric("Procijenjena cijena", f"${rezultat['predikcija']:,.0f}")
    kol2.metric(
        "Interval (95%)",
        f"${rezultat['donja_granica']:,.0f} - ${rezultat['gornja_granica']:,.0f}",
    )
    kol3.metric("Cijena po m2", f"${cijena_po_m2:,.0f}")

    st.subheader("Najslicnije vec prodate nekretnine")
    slicne = pronadji_slicne_nekretnine(df, nekretnina, k=5)
    st.dataframe(slicne, use_container_width=True)


st.markdown("---")
st.header("Koji atributi uticu na cijenu")

kol_imp1, kol_imp2 = st.columns(2)

with kol_imp1:
    st.subheader("Znacaj atributa (XGBoost)")
    znacaj = xgboost_znacaj_atributa(model, X.columns).reset_index()
    znacaj.columns = ["atribut", "znacaj"]
    fig_znacaj = px.bar(
        znacaj.sort_values("znacaj"), x="znacaj", y="atribut", orientation="h",
        title="Koliko je svaki atribut bitan modelu",
    )
    st.plotly_chart(fig_znacaj, use_container_width=True)

with kol_imp2:
    st.subheader("Smjer uticaja (Linear Regression)")
    koeficijenti = linear_koeficijenti(baseline_model, X.columns)
    st.dataframe(koeficijenti, use_container_width=True)
    st.caption("Pozitivna vrijednost = povecava cijenu, negativna = smanjuje cijenu (po jedinici)")


st.markdown("---")
st.header("What-if simulator")
st.caption("Promijeni karakteristike nekretnine iz forme iznad i vidi kako se cijena mijenja uzivo")

kol_w1, kol_w2, kol_w3, kol_w4 = st.columns(4)

with kol_w1:
    whatif_povrsina = st.number_input(
        "Nova povrsina", min_value=200, max_value=6000, value=povrsina, step=50, key="whatif_povrsina"
    )
with kol_w2:
    whatif_broj_kupatila = st.number_input(
        "Novi broj kupatila", min_value=0.0, max_value=6.0, value=broj_kupatila, step=0.5,
        key="whatif_kupatila",
    )
with kol_w3:
    whatif_kvalitet = st.slider(
        "Novi kvalitet/stanje (renoviranje)", 1, 10, kvalitet_stanje, key="whatif_kvalitet"
    )
with kol_w4:
    whatif_garaza = st.number_input(
        "Nova garaza", min_value=0, max_value=5, value=garaza, key="whatif_garaza"
    )

izmjene = {
    "povrsina": whatif_povrsina,
    "broj_kupatila": whatif_broj_kupatila,
    "kvalitet_stanje": whatif_kvalitet,
    "garaza": whatif_garaza,
}

rezultat_whatif = simuliraj_izmjenu(model, df, nekretnina, izmjene)

kol_r1, kol_r2, kol_r3 = st.columns(3)
kol_r1.metric("Cijena prije", f"${rezultat_whatif['cijena_prije']:,.0f}")
kol_r2.metric(
    "Cijena poslije",
    f"${rezultat_whatif['cijena_poslije']:,.0f}",
    delta=f"${rezultat_whatif['razlika_dolari']:,.0f}",
)
kol_r3.metric("Promjena", f"{rezultat_whatif['razlika_procenat']:+.1f}%")


st.markdown("---")
st.header("Poredjenje modela")

metrike_baseline = evaluiraj_model(baseline_model, X_test, y_test, naziv="Linear Regression")
metrike_napredni = evaluiraj_model(model, X_test, y_test, naziv="XGBoost")

poredjenje = pd.DataFrame({"Linear Regression": metrike_baseline, "XGBoost": metrike_napredni})
st.dataframe(poredjenje, use_container_width=True)


st.markdown("---")
st.header("Ucitaj vise nekretnina odjednom (CSV ili Excel)")

POTREBNE_KOLONE = [
    "povrsina", "broj_soba", "broj_kupatila", "godina_izgradnje",
    "lokacija", "kvalitet_stanje", "garaza", "velicina_placa",
]

st.caption("Fajl treba da ima kolone: " + ", ".join(POTREBNE_KOLONE))

uploaded_file = st.file_uploader("Izaberi fajl", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        nove_nekretnine = pd.read_csv(uploaded_file)
    else:
        nove_nekretnine = pd.read_excel(uploaded_file)

    kolone_koje_fale = [k for k in POTREBNE_KOLONE if k not in nove_nekretnine.columns]

    if kolone_koje_fale:
        st.error("Fajlu nedostaju kolone: " + ", ".join(kolone_koje_fale))
    else:
        nepoznate_lokacije = set(nove_nekretnine["lokacija"]) - set(lokacije)
        if nepoznate_lokacije:
            st.warning(
                "Ove lokacije ne postoje u datasetu pa ce biti tretirane kao "
                "nepoznata/prosjecna lokacija: " + ", ".join(nepoznate_lokacije)
            )

        X_postojece, X_nove = pripremi_vise_nekretnina(df, nove_nekretnine[POTREBNE_KOLONE])
        predikcije = model.predict(X_nove)

        rezultati = nove_nekretnine.copy()
        rezultati["procijenjena_cijena"] = predikcije.round(0)

        st.subheader("Rezultati procjene")
        st.dataframe(rezultati, use_container_width=True)

        st.subheader("Poredjenje procijenjenih cijena (ucitane nekretnine)")
        rezultati_prikaz = rezultati.reset_index().rename(columns={"index": "redni_broj"})
        fig_poredjenje = px.bar(
            rezultati_prikaz, x="redni_broj", y="procijenjena_cijena",
            title="Poredjenje ucitanih nekretnina po procijenjenoj cijeni",
        )
        st.plotly_chart(fig_poredjenje, use_container_width=True)

st.markdown("---")
st.header("Dashboard - prosjecne cijene")

kol_a, kol_b = st.columns(2)

with kol_a:
    prosjek_po_lokaciji = (
        df.groupby("lokacija")["cijena"].mean().sort_values(ascending=False).reset_index()
    )
    fig_lokacija = px.bar(
        prosjek_po_lokaciji, x="lokacija", y="cijena",
        title="Prosjecna cijena po lokaciji",
    )
    st.plotly_chart(fig_lokacija, use_container_width=True)

with kol_b:
    prosjek_po_kvalitetu = df.groupby("kvalitet_stanje")["cijena"].mean().reset_index()
    fig_kvalitet = px.bar(
        prosjek_po_kvalitetu, x="kvalitet_stanje", y="cijena",
        title="Prosjecna cijena po kvalitetu/stanju objekta",
    )
    st.plotly_chart(fig_kvalitet, use_container_width=True)