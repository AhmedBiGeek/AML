"""Professional Profile Explorer.

Run from this folder on any machine of the group:

    pip install -r requirements.txt
    streamlit run streamlit_app.py

The notebooks are not imported and are not modified. Each card calls the
model trained for that member.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS = Path(__file__).resolve().parent / "models"

EDUCATION = [
    ("Preschool", 1), ("1st-4th", 2), ("5th-6th", 3), ("7th-8th", 4),
    ("9th", 5), ("10th", 6), ("11th", 7), ("12th", 8), ("HS-grad", 9),
    ("Some-college", 10), ("Assoc-voc", 11), ("Assoc-acdm", 12),
    ("Bachelors", 13), ("Masters", 14), ("Prof-school", 15), ("Doctorate", 16),
]
WORKCLASS = [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked",
]
OCCUPATION = [
    "Adm-clerical", "Exec-managerial", "Prof-specialty", "Craft-repair",
    "Sales", "Tech-support", "Machine-op-inspct", "Transport-moving",
    "Handlers-cleaners", "Other-service", "Farming-fishing",
    "Protective-serv", "Priv-house-serv", "Armed-Forces",
]
RELATIONSHIP = ["Husband", "Wife", "Not-in-family", "Unmarried", "Own-child", "Other-relative"]
MARITAL = [
    "Married-civ-spouse", "Never-married", "Divorced", "Separated",
    "Widowed", "Married-spouse-absent", "Married-AF-spouse",
]
COUNTRIES = [
    "United-States", "Mexico", "Canada", "England", "Germany", "France",
    "Italy", "India", "China", "Japan", "Philippines", "Puerto-Rico",
    "Cuba", "El-Salvador", "Other",
]
SEX = ["Female", "Male"]
RACE = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]

PROFILES = {
    "Parcours courant": {
        "age": 35, "education": "HS-grad", "hours": 40, "workclass": "Private",
        "occupation": "Craft-repair", "relationship": "Husband",
        "marital": "Married-civ-spouse", "gain": 0, "loss": 0,
        "country": "United-States", "sex": "Male", "race": "White",
    },
    "Capital extrême": {
        "age": 22, "education": "Bachelors", "hours": 40, "workclass": "Private",
        "occupation": "Sales", "relationship": "Own-child",
        "marital": "Never-married", "gain": 99999, "loss": 0,
        "country": "United-States", "sex": "Female", "race": "White",
    },
}


def _cut(values, bins, labels):
    return pd.cut(values, bins=bins, labels=labels).astype(str)


def _classes(estimator):
    if hasattr(estimator, "classes_"):
        return list(estimator.classes_)
    return list(estimator.named_steps["clf"].classes_)


@st.cache_resource
def load_models():
    names = {
        "income": "ahmed_bejaoui.joblib",
        "occupation": "ahmed_guesmi.joblib",
        "rhythm": "nour_trabelsi.joblib",
        "education": "arij_yousfi.joblib",
        "cluster": "seifeddine_alaimi.joblib",
    }
    loaded = {}
    for key, filename in names.items():
        path = MODELS / filename
        loaded[key] = joblib.load(path) if path.exists() else None
    return loaded


def income_card(model, raw):
    featured = raw.copy()
    featured["workload"] = _cut(featured["hours-per-week"], [-np.inf, 34, 40, np.inf], ["part-time", "full-time", "overtime"])
    featured["workclass-occupation"] = featured["workclass"].astype(str) + "__" + featured["occupation"].astype(str)
    featured["occupation-group"] = featured["occupation"].map(model["occupation_mapping"]).fillna("unknown")
    probability = float(model["pipeline"].predict_proba(featured[model["features"]])[0, 1])
    threshold = float(model["threshold"])
    high = probability >= threshold
    return {
        "title": "Plus de 50 000 $" if high else "50 000 $ ou moins",
        "label": ">50K" if high else "<=50K",
        "score": probability,
        "bars": [("Probabilité > 50 000 $", probability)],
        "note": f"Seuil du modèle : {threshold:.3f}. Ce revenu est transmis aux autres modules.",
    }


def rhythm_card(model, raw, income_label):
    frame = raw.copy()
    frame["income"] = income_label
    frame["age-group"] = _cut(frame["age"], [-np.inf, 25, 40, 60, np.inf], ["young", "adult", "middle", "senior"])
    frame["education-level"] = _cut(
        frame["education-num"], [-np.inf, 4, 9, 12, 13, np.inf],
        ["school", "high-school", "postsecondary", "bachelor", "postgraduate"],
    )
    sector = {
        "Private": "private", "Federal-gov": "public", "State-gov": "public", "Local-gov": "public",
        "Self-emp-inc": "self-employed", "Self-emp-not-inc": "self-employed",
        "Without-pay": "other", "Never-worked": "other",
    }
    frame["employment-sector"] = frame["workclass"].map(sector).fillna("unknown")
    matrix = model["preprocessor"].transform(frame[model["features"]])
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    prepared = pd.DataFrame(matrix, columns=model["feature_names"])
    probabilities = model["model"].predict_proba(prepared)[0]
    order = np.argsort(-probabilities)
    bars = [(model["labels"][int(index)], float(probabilities[index])) for index in order]
    return {
        "title": bars[0][0],
        "bars": bars,
        "note": f"Revenu utilisé : {income_label}. Le rythme est estimé sans les heures saisies.",
    }


def occupation_card(model, raw, income_label):
    profile = raw.copy()
    profile["income"] = income_label
    encoded = pd.DataFrame(index=profile.index)
    encoded["education-num"] = profile["education-num"]
    encoded["age"] = profile["age"]
    encoded["hours-per-week"] = profile["hours-per-week"]
    encoded["income"] = (profile["income"] == ">50K").astype(int)
    encoded["workclass"] = profile["workclass"].map(model["workclass_codes"]).fillna(-1).astype(int)
    encoded["education-level"] = pd.cut(profile["education-num"], [0, 8, 9, 12, 13, 16], labels=False).astype(int)
    encoded["age-group"] = pd.cut(profile["age"], [0, 24, 34, 44, 54, 64, 120], labels=False).astype(int)
    encoded["workload"] = pd.cut(profile["hours-per-week"], [0, 34, 40, 49, 200], labels=False).astype(int)
    probabilities = model["family_model"].predict_proba(encoded)[0]
    classes = _classes(model["family_model"])
    family_names = list(model["family_classes"])
    winner = int(np.argmax(probabilities))
    confidence = float(probabilities[winner])
    bars = sorted(
        [(family_names[int(classes[index])], float(probabilities[index])) for index in range(len(probabilities))],
        key=lambda item: item[1],
        reverse=True,
    )
    if confidence < model["tau_mid"]:
        return {
            "title": "Profil trop incertain",
            "bars": bars,
            "note": f"Confiance {confidence:.0%}, sous le seuil modéré ({model['tau_mid']:.2f}).",
        }
    family_index = int(classes[winner])
    occupation_probabilities = model["occupation_model"].predict_proba(encoded)[0]
    job_names = []
    for index, encoded_class in enumerate(_classes(model["occupation_model"])):
        name = model["occupation_classes"][int(encoded_class)]
        family_of_job = model["family_classes"].index(model["families"][name])
        if family_of_job == family_index:
            job_names.append((name, float(occupation_probabilities[index])))
    job = max(job_names, key=lambda item: item[1])[0] if job_names else ""
    level = "élevée" if confidence >= model["tau"] else "modérée"
    return {
        "title": family_names[family_index],
        "bars": bars,
        "note": f"Fiabilité {level} ({confidence:.0%}). Métier indicatif : {job}.",
    }


def education_card(model, raw, income_label, sex, race):
    row = {
        "age": int(raw["age"].iloc[0]),
        "hours-per-week": int(raw["hours-per-week"].iloc[0]),
        "capital-gain": float(raw["capital-gain"].iloc[0]),
        "capital-loss": float(raw["capital-loss"].iloc[0]),
        "income": 1 if income_label == ">50K" else 0,
        "sex": 0 if sex == "Female" else 1,
        "has-capital": int(raw["capital-gain"].iloc[0] > 0 or raw["capital-loss"].iloc[0] > 0),
        "marital-status": raw["marital-status"].iloc[0],
    }
    age_labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    hour_labels = ["part-time", "standard", "heavy", "extreme"]
    row["age-group"] = int(pd.cut([row["age"]], [0, 24, 34, 44, 54, 64, 120], labels=age_labels).codes[0])
    row["workload"] = int(pd.cut([row["hours-per-week"]], [0, 34, 40, 50, 100], labels=hour_labels).codes[0])
    sector = {
        "Private": "private", "Self-emp-not-inc": "self-employed", "Self-emp-inc": "self-employed",
        "Federal-gov": "public", "State-gov": "public", "Local-gov": "public",
        "Without-pay": "other", "Never-worked": "other",
    }
    categories = {
        "occupation": raw["occupation"].iloc[0],
        "workclass": raw["workclass"].iloc[0],
        "marital-status": row.pop("marital-status"),
        "relationship": raw["relationship"].iloc[0],
        "race": race,
        "native-country": raw["native-country"].iloc[0],
        "employment-sector": sector.get(raw["workclass"].iloc[0], "other"),
    }
    encoded = {f"{name}_{value}": 1 for name, value in categories.items()}
    encoded.update(row)
    frame = pd.DataFrame([encoded]).reindex(columns=model["columns"], fill_value=0)
    probability = float(model["model"].predict_proba(frame)[0, 1])
    high = probability >= 0.5
    return {
        "title": "Licence ou plus" if high else "Sous la licence",
        "bars": [("Probabilité licence ou plus", probability)],
        "note": "Seuil 0,50. Sexe et origine sont utilisés seulement par ce module, comme dans le notebook d'Arij.",
    }


def cluster_card(model, raw, income_label):
    frame = raw.copy()
    frame["income"] = income_label
    capital = frame["capital-gain"] - frame["capital-loss"]
    frame["net_capital_log"] = np.sign(capital) * np.log1p(np.abs(capital))
    frame["age_group"] = pd.cut(frame["age"], model["age_bins"], labels=model["age_labels"], ordered=True)
    frame["workload"] = pd.cut(frame["hours-per-week"], model["hour_bins"], labels=model["hour_labels"], ordered=True)
    frame["work_life_ratio"] = frame["hours-per-week"] / frame["age"]
    transformed = model["preprocessor"].transform(frame[model["columns"]])
    group = int(model["mapping"][int(model["kmeans"].predict(transformed)[0])])
    return {
        "title": model["names"][group],
        "bars": [],
        "note": f"Groupe {group} sur 4. Sexe et origine ne servent pas à former les groupes.",
    }


def render_card(number, member, title, result):
    st.markdown(f"**{number} · {title}**")
    st.caption(member)
    if result is None:
        st.info("Modèle absent du dossier platform/models.")
        return
    st.markdown(f"<div class='result'>{result['title']}</div>", unsafe_allow_html=True)
    for label, value in result["bars"]:
        st.progress(min(max(value, 0.0), 1.0), text=f"{label} · {value:.0%}")
    st.caption(result["note"])


def apply_profile(profile):
    for key, value in profile.items():
        st.session_state[key] = value


def main():
    st.set_page_config(page_title="Professional Profile Explorer", page_icon="▣", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; max-width: 1180px;}
        .hero {background:#12343e; color:#f7fbfb; border-radius:18px; padding:26px 28px; margin-bottom:16px;}
        .hero h1 {color:#f7fbfb; font-size:1.8rem; margin:0 0 6px 0;}
        .hero p {margin:0; color:#d5e4e8;}
        .result {font-size:1.35rem; font-weight:650; color:#12343e; margin:4px 0 8px 0;}
        div[data-testid="stVerticalBlockBorderWrapper"] {border-color:#d7e2e6;}
        </style>
        <div class="hero">
          <h1>Professional Profile Explorer</h1>
          <p>Une carte, cinq modules. Prototype sur les données Adult de 1994 : ce n'est ni un outil de recrutement ni une garantie.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if "age" not in st.session_state:
        apply_profile(PROFILES["Parcours courant"])

    models = load_models()
    form, board = st.columns([0.92, 1.15], gap="large")
    with form:
        st.subheader("Profil")
        demo_a, demo_b = st.columns(2)
        if demo_a.button("Parcours courant", use_container_width=True):
            apply_profile(PROFILES["Parcours courant"])
            st.rerun()
        if demo_b.button("Capital extrême", use_container_width=True):
            apply_profile(PROFILES["Capital extrême"])
            st.rerun()
        age = st.slider("Âge", 17, 90, key="age")
        education_name = st.selectbox("Études", [name for name, _ in EDUCATION], key="education")
        hours = st.slider("Heures par semaine", 1, 99, key="hours")
        workclass = st.selectbox("Secteur", WORKCLASS, key="workclass")
        occupation = st.selectbox("Métier saisi", OCCUPATION, key="occupation")
        relationship = st.selectbox("Lien familial", RELATIONSHIP, key="relationship")
        marital = st.selectbox("Situation familiale", MARITAL, key="marital")
        gain = st.number_input("Gain en capital", 0, 99999, step=100, key="gain")
        loss = st.number_input("Perte en capital", 0, 99999, step=100, key="loss")
        country = st.selectbox("Pays", COUNTRIES, key="country")
        with st.expander("Champs utilisés seulement par le module d'Arij"):
            sex = st.selectbox("Sexe", SEX, key="sex")
            race = st.selectbox("Origine", RACE, key="race")
            st.caption("Les quatre autres modules ignorent ces deux champs.")

    education_num = dict(EDUCATION)[education_name]
    raw = pd.DataFrame([{
        "age": age,
        "education-num": education_num,
        "hours-per-week": hours,
        "workclass": workclass,
        "occupation": occupation,
        "relationship": relationship,
        "marital-status": marital,
        "capital-gain": gain,
        "capital-loss": loss,
        "native-country": country,
    }])
    income = income_card(models["income"], raw) if models["income"] else None
    income_label = income["label"] if income else "<=50K"
    cards = [
        ("01", "Ahmed Bejaoui", "Revenu estimé", income),
        ("02", "Ahmed Guesmi", "Famille de métiers", occupation_card(models["occupation"], raw, income_label) if models["occupation"] else None),
        ("03", "Nour Trabelsi", "Rythme de travail", rhythm_card(models["rhythm"], raw, income_label) if models["rhythm"] else None),
        ("04", "Arij Yousfi", "Niveau d'études associé", education_card(models["education"], raw, income_label, sex, race) if models["education"] else None),
        ("05", "Seifeddine Alaimi", "Groupe professionnel", cluster_card(models["cluster"], raw, income_label) if models["cluster"] else None),
    ]
    with board:
        st.subheader("Carte")
        for number, member, title, result in cards:
            with st.container(border=True):
                render_card(number, member, title, result)


if __name__ == "__main__":
    main()
