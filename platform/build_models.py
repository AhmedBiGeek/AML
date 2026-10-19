"""Fit the group models and store them next to the Streamlit app.

The notebooks in ``final notebooks`` are not modified. Ahmed Bejaoui and
Nour Trabelsi already saved their models; those files are copied. Ahmed
Guesmi, Arij Yousfi and Seifeddine Alaimi did not save a model file, so this
script refits the estimator their notebook kept, with the same data rules.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "adult" / "adult.data"
OUT = Path(__file__).resolve().parent / "models"
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]


def adult_train() -> pd.DataFrame:
    frame = pd.read_csv(DATA, names=COLUMNS, na_values="?", skipinitialspace=True)
    return frame


def copy_saved() -> None:
    pairs = {
        "member1_xgboost_best.joblib": "ahmed_bejaoui.joblib",
        "member3_rhythm_xgboost.joblib": "nour_trabelsi.joblib",
    }
    for source_name, target_name in pairs.items():
        source = ROOT / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, OUT / target_name)
        print("copied", target_name)


def build_guesmi() -> None:
    print("fitting Ahmed Guesmi...")
    frame = adult_train().drop_duplicates().dropna(subset=["occupation"]).reset_index(drop=True)
    families = {
        "Adm-clerical": "Bureau, vente & protection",
        "Sales": "Bureau, vente & protection",
        "Tech-support": "Bureau, vente & protection",
        "Protective-serv": "Bureau, vente & protection",
        "Armed-Forces": "Bureau, vente & protection",
        "Exec-managerial": "Cadres & prof. intellectuelles",
        "Prof-specialty": "Cadres & prof. intellectuelles",
        "Craft-repair": "Manuel & technique",
        "Farming-fishing": "Manuel & technique",
        "Machine-op-inspct": "Manuel & technique",
        "Transport-moving": "Manuel & technique",
        "Handlers-cleaners": "Services & manutention",
        "Other-service": "Services & manutention",
        "Priv-house-serv": "Services & manutention",
    }
    workclass_codes = {value: index for index, value in enumerate(sorted(frame["workclass"].dropna().unique()))}

    def engineer(raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=raw.index)
        out["education-num"] = raw["education-num"]
        out["age"] = raw["age"]
        out["hours-per-week"] = raw["hours-per-week"]
        out["income"] = (raw["income"] == ">50K").astype(int)
        out["workclass"] = raw["workclass"].map(workclass_codes).fillna(-1).astype(int)
        out["education-level"] = pd.cut(raw["education-num"], [0, 8, 9, 12, 13, 16], labels=False).astype(int)
        out["age-group"] = pd.cut(raw["age"], [0, 24, 34, 44, 54, 64, 120], labels=False).astype(int)
        out["workload"] = pd.cut(raw["hours-per-week"], [0, 34, 40, 49, 200], labels=False).astype(int)
        return out

    matrix = engineer(frame)
    categorical = [name in ["income", "workclass", "education-level", "age-group", "workload"] for name in matrix.columns]
    family_encoder = LabelEncoder().fit(frame["occupation"].map(families))
    occupation_encoder = LabelEncoder().fit(frame["occupation"])
    family_target = family_encoder.transform(frame["occupation"].map(families))
    occupation_target = occupation_encoder.transform(frame["occupation"])

    def target_counts(values):
        counts = Counter(values)
        largest = max(counts.values())
        return {key: int(max(value, 0.5 * largest)) for key, value in counts.items()}

    def model():
        return XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.08, subsample=0.8,
            colsample_bytree=0.8, objective="multi:softprob", random_state=42,
            n_jobs=1, tree_method="hist",
        )

    def pipe():
        return ImbPipeline([
            ("smote", SMOTENC(
                categorical_features=categorical, sampling_strategy=target_counts,
                k_neighbors=3, random_state=42,
            )),
            ("clf", model()),
        ])

    search = RandomizedSearchCV(
        pipe(),
        {
            "clf__n_estimators": [150, 250, 350],
            "clf__max_depth": [3, 4, 5, 6],
            "clf__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "clf__subsample": [0.7, 0.8, 1.0],
            "clf__colsample_bytree": [0.7, 0.8, 1.0],
            "clf__min_child_weight": [1, 3, 5],
            "clf__reg_lambda": [1, 5, 10],
        },
        n_iter=8,
        scoring="f1_macro",
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=1,
        refit=True,
    )
    search.fit(matrix, family_target)
    family_model = search.best_estimator_.named_steps["clf"]
    occupation_model = pipe().fit(matrix, occupation_target).named_steps["clf"]
    joblib.dump(
        {
            "family_model": family_model,
            "occupation_model": occupation_model,
            "families": families,
            "family_classes": list(family_encoder.classes_),
            "occupation_classes": list(occupation_encoder.classes_),
            "workclass_codes": workclass_codes,
            "tau": 0.68,
            "tau_mid": 0.53,
            "params": search.best_params_,
            "task": "multiclass occupation family",
        },
        OUT / "ahmed_guesmi.joblib",
    )
    print("saved ahmed_guesmi.joblib", search.best_params_)


def build_arij() -> None:
    print("fitting Arij Yousfi...")
    frame = adult_train().drop_duplicates().reset_index(drop=True)
    for column in ["workclass", "occupation", "native-country"]:
        frame[column] = frame[column].fillna("Unknown")
    frame["education-level"] = (frame["education-num"] >= 13).astype(int)
    frame = frame.drop(columns=["education", "education-num", "fnlwgt"])
    frame["age-group"] = pd.cut(
        frame["age"], bins=[0, 24, 34, 44, 54, 64, 120],
        labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
    )
    frame["workload"] = pd.cut(
        frame["hours-per-week"], bins=[0, 34, 40, 50, 100],
        labels=["part-time", "standard", "heavy", "extreme"],
    )
    sector = {
        "Private": "private", "Self-emp-not-inc": "self-employed", "Self-emp-inc": "self-employed",
        "Federal-gov": "public", "State-gov": "public", "Local-gov": "public",
        "Without-pay": "other", "Never-worked": "other", "Unknown": "other",
    }
    frame["employment-sector"] = frame["workclass"].map(sector)
    frame["has-capital"] = ((frame["capital-gain"] > 0) | (frame["capital-loss"] > 0)).astype(int)
    model_frame = frame.copy()
    model_frame["income"] = model_frame["income"].map({"<=50K": 0, ">50K": 1})
    model_frame["sex"] = model_frame["sex"].map({"Female": 0, "Male": 1})
    model_frame["age-group"] = model_frame["age-group"].cat.codes
    model_frame["workload"] = model_frame["workload"].cat.codes
    nominal = ["occupation", "workclass", "marital-status", "relationship", "race", "native-country", "employment-sector"]
    model_frame = pd.get_dummies(model_frame, columns=nominal, dtype=int)
    scaled = model_frame.copy()
    numeric = ["age", "hours-per-week", "capital-gain", "capital-loss"]
    scaled[numeric] = StandardScaler().fit_transform(scaled[numeric])
    correlation = scaled.corr(method="spearman")["education-level"].drop("education-level")
    selected = correlation[correlation.abs() > 0.05].index.tolist()
    features = model_frame[selected]
    target = model_frame["education-level"]
    train_x, _, train_y, _ = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)
    search = RandomizedSearchCV(
        XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=1),
        {
            "learning_rate": [0.1, 0.05, 0.01],
            "max_depth": [3, 5, 7],
            "n_estimators": [100, 200, 300],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "gamma": [0, 0.5, 1],
            "reg_alpha": [0, 1, 10],
            "reg_lambda": [1, 10],
        },
        n_iter=20,
        scoring="accuracy",
        cv=3,
        n_jobs=1,
        random_state=42,
    )
    search.fit(train_x, train_y)
    joblib.dump(
        {
            "model": search.best_estimator_,
            "columns": selected,
            "params": search.best_params_,
            "task": "binary education level (licence or more)",
        },
        OUT / "arij_yousfi.joblib",
    )
    print("saved arij_yousfi.joblib", len(selected), "columns", search.best_params_)


def build_seifeddine() -> None:
    print("fitting Seifeddine Alaimi...")
    frame = adult_train().dropna(subset=["workclass", "occupation"]).drop_duplicates()
    frame["net_capital_log"] = np.sign(frame["capital-gain"] - frame["capital-loss"]) * np.log1p(
        np.abs(frame["capital-gain"] - frame["capital-loss"])
    )
    age_bins = [0, 25, 35, 50, 65, 120]
    age_labels = [
        "17-25 (insertion)", "26-35 (installation)", "36-50 (plateau)",
        "51-65 (fin de carrière)", "66+ (retraite)",
    ]
    hour_bins = [0, 34, 40, 50, 100]
    hour_labels = ["temps partiel (<35h)", "temps plein (35-40h)", "soutenu (41-50h)", "intensif (>50h)"]
    frame["age_group"] = pd.cut(frame["age"], bins=age_bins, labels=age_labels, ordered=True)
    frame["workload"] = pd.cut(frame["hours-per-week"], bins=hour_bins, labels=hour_labels, ordered=True)
    frame["work_life_ratio"] = frame["hours-per-week"] / frame["age"]
    numeric = ["age", "education-num", "hours-per-week", "net_capital_log", "work_life_ratio"]
    ordinal = ["age_group", "workload"]
    categorical = ["occupation", "workclass", "income"]
    encoder = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)
    preprocessor = Pipeline([
        ("preprocessing", ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("ord", Pipeline([
                ("encode", OrdinalEncoder(categories=[age_labels, hour_labels])),
                ("scale", StandardScaler()),
            ]), ordinal),
            ("cat", encoder, categorical),
        ]))
    ])
    matrix = frame[numeric + ordinal + categorical]
    transformed = preprocessor.fit_transform(matrix)
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42).fit(transformed)
    order = pd.Series(kmeans.labels_).value_counts().index.tolist()
    mapping = {old: new for new, old in enumerate(order)}
    names = {
        0: "Actifs confirmés à temps plein",
        1: "Jeunes actifs en début de carrière",
        2: "Professionnels suractifs",
        3: "Hauts revenus à fort capital",
    }
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "kmeans": kmeans,
            "mapping": mapping,
            "names": names,
            "age_bins": age_bins,
            "age_labels": age_labels,
            "hour_bins": hour_bins,
            "hour_labels": hour_labels,
            "columns": numeric + ordinal + categorical,
            "task": "professional profile clustering",
        },
        OUT / "seifeddine_alaimi.joblib",
    )
    print("saved seifeddine_alaimi.joblib", {names[mapping[label]]: int(count) for label, count in zip(*np.unique(kmeans.labels_, return_counts=True))})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    copy_saved()
    build_seifeddine()
    build_arij()
    build_guesmi()
    print("models ready in", OUT)
