# Accidentologie - Étude des circonstances des accidents de la route en France de 2005 à 2022

Erika Méronville - Emmanuel Gautier

Première page streamlit :
https://fev24cdsaccidents-fnkswybzpjcy3a8zft6hy9.streamlit.app/

------------

Ce répertoire présente le processus de mise en oeuvre d'un projet de machine learning portant sur les accidents de la route en France au cours de la période de 2005 à 2022. Plus précisément, l’objectif principal consiste à explorer, préparer et modéliser un jeu de données complet dans le but de prédire la gravité des accidents routiers en fonction des circonstances qui les entourent.

Pour cela, on se basera sur les éléments recueillis (sur www.data.gouv.fr) qui se composent d'un ensemble de fichiers portant sur 4 rubriques complémentaires :
— caractéristiques : informations générales sur les circonstances de l’accident ;
— lieux : détails sur l’emplacement de l’accident ;
— véhicules : informations sur les véhicules impliqués ;
— usagers : données sur les personnes impliquées dans l’accident.

Notre démarche consistera à appliquer les étapes clés suivantes :
— exploration initiale des données : examen de la structure, des types de variables, et des valeurs pour chaque fichier ;
— analyse détaillée de chaque variable : étude de la distribution, des valeurs manquantes, des outliers, et de l’évolution temporelle ;
— identification des problèmes potentiels : repérage des incohérences, des changements de codification, et des valeurs aberrantes ;
— proposition de prétraitement : suggestions pour le nettoyage, la transformation et la création de nouvelles variables.
— modélisation : application de modèles de machine learning et évaluation de leurs performances à l’aide de métriques.


Organisation
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   ├── régression logistique
    |   ├── multiples modèles de classification
    |   └── deep learning
    |
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
