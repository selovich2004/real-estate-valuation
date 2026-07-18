# 🏠 Procjena vrijednosti stambenih nekretnina

**Student:** Damir Selović  
**Broj indeksa:** 23/022  
**Fakultet:** Fakultet za informacione sisteme i tehnologije  
**Predmet:** Data Mining  

---

## O projektu

Sistem za automatsku procjenu tržišne vrijednosti stambenih nekretnina, na osnovu karakteristika kao što su površina, broj soba, broj kupatila, godina izgradnje, lokacija, stanje objekta, garaža i veličina placa.

Koristi Ames Housing dataset (OpenML, data_id=42165) i poredi dva regresiona modela:
- Linear Regression — baseline model
- XGBoost — napredni model

---

## Funkcionalnosti

- Predikcija cijene sa intervalom pouzdanosti (95%)
- Pronalaženje najsličnijih već prodatih nekretnina (KNN)
- Objašnjenje koji atributi povećavaju/smanjuju cijenu (feature importance)
- Cijena po kvadratnom metru
- What-if simulacija (promjena kupatila, renoviranje, garaža, površina)
- Upload CSV/Excel fajla sa više nekretnina odjednom
- Dashboard sa prosječnim cijenama po lokaciji i kvalitetu objekta
- Poređenje Linear Regression i XGBoost modela (MAE, RMSE, R²)

---

## Struktura projekta

real-estate-app/
├── app.py                  # Streamlit web aplikacija
├── data_prep.py            # Ucitavanje i priprema Ames Housing dataseta
├── train_baseline.py       # Treniranje Linear Regression modela
├── train_advanced.py       # Treniranje XGBoost modela
├── similar_homes.py        # KNN - pronalazenje najslicnijih nekretnina
├── explain.py              # Feature importance i smjer uticaja atributa
├── what_if.py              # Predikcija sa intervalom + what-if simulacija
├── requirements.txt        # Python paketi
├── data/                   # Lokalna kopija dataseta
└── models/                 # Sacuvani trenirani modeli

---

## Pokretanje

### 1. Instaliraj zavisnosti
pip install -r requirements.txt

### 2. Istreniraj modele (redoslijed je bitan)
python train_baseline.py
python train_advanced.py

### 3. Pokreni aplikaciju
streamlit run app.py

Aplikacija se otvara u browseru na http://localhost:8501

---

## Tehnički stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Streamlit · Plotly

---

## Dataset

Ames Housing dataset (Ames, Iowa) — karakteristike i prodajne cijene oko 1460 stambenih nekretnina, preuzet sa OpenML-a (https://www.openml.org/d/42165).

---

## Metode

- Linear Regression i XGBoost — regresioni modeli za predikciju cijene
- One-hot encoding — za pretvaranje lokacije u brojeve
- K-Nearest Neighbors (KNN) — za pronalazenje slicnih nekretnina
- Feature importance i koeficijenti — za objasnjenje uticaja atributa
- Interval predikcije — na osnovu RMSE modela (±1.96 × RMSE, ~95%)

---

## DISCLAIMER
Za razvoj ovog projekta korišćeni su AI alati: Claude AI i ChatGPT.