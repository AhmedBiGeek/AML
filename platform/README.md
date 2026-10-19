# Professional Profile Explorer

Plateforme Streamlit qui réunit les cinq modèles du groupe dans une seule
carte profil. Les notebooks du dossier `final notebooks` ne sont ni importés
ni modifiés pendant l'utilisation.

La documentation détaillée de l'architecture, des modèles, de l'installation,
des tests et du dépannage se trouve dans
[`DOCUMENTATION.md`](DOCUMENTATION.md). Sa version prête à imprimer est
[`DOCUMENTATION.pdf`](DOCUMENTATION.pdf).

## Installation (une seule fois)

Python 3.11 ou plus récent est recommandé.

```bash
python -m pip install -r requirements.txt
```

## Démarrage

Depuis le dossier `platform` :

```bash
python run_platform.py
```

Le navigateur s'ouvre sur `http://localhost:8501`.

Le dossier `models` fait partie de la plateforme. Il doit rester à côté de
`streamlit_app.py`; aucun entraînement et aucun accès à `adult.data` ne sont
nécessaires pour lancer l'application.

## Modules intégrés

1. Ahmed Bejaoui — revenu estimé
2. Ahmed Guesmi — famille de métiers
3. Nour Trabelsi — rythme de travail
4. Arij Yousfi — niveau d'études associé
5. Seifeddine Alaimi — groupe professionnel

Le revenu estimé par le premier module est transmis automatiquement aux
modules qui utilisent `income`.

`build_models.py` sert uniquement à reproduire les fichiers du dossier
`models` à partir des règles déjà présentes dans les notebooks. Il n'est pas
utilisé au démarrage normal de la plateforme.
