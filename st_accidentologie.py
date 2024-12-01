import streamlit as st
import json
import pandas as pd
import numpy as np
from sklearn.decomposition         import PCA
from sklearn.preprocessing         import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
# TODO  activer après compilation de tensorflow
import tensorflow as tf
import joblib
from st_demo import st_demo
import os
import warnings
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers


rep_raw = "data/raw/"
rep_processed = "data/processed/"
rep_models = "models/"
rep_figures = "reports/figures/"
rep_ref = "references/"


st.sidebar.title("Sommaire")

pages=["Présentation",      # 0
       "Jeu de données",    # 1
       "Visualisation",     # 2
       "Preprocessing",     # 3
       "Modélisation",      # 4
       "Démonstration"]     # 5
page=st.sidebar.radio("Aller vers", pages)                #
st.sidebar.write("Erika Méronville\n&\nEmmanuel Gautier")


##############################################################################
#
# Chargements en mémoire
#
# Ces chargements sont réalisés par des fonctions décorées par @st.cache_data
# pour que les lectures soient réalisées une seule fois.
#
##############################################################################

# Lecture des modèles entraînés

@st.cache_data
def read_models():

    SVC, pca, scaler_ml, scaler_DP = None, None, None, None

    try:
        SVC       = joblib.load(rep_models + "SVC.mdl")
        pca       = joblib.load(rep_models + "pca_ml.mdl")
        scaler_ml = joblib.load(rep_models + "acc_scaler.mdl")
        scaler_DP = joblib.load(rep_models + "scaler_DP.joblib")

    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")

    return SVC, pca, scaler_ml, scaler_DP

# Lecture du jeu de données avec une observation à zero
# ce jeu de données avec une seule observation est utilisé par la démo.

@st.cache_data
def read_df_zero():

    df_zero = None

    try:
        df_zero = pd.read_csv(rep_processed + "data_zero.csv", sep='\t')

    except Exception as e:
        st.error(f"Erreur lors du chargement du jeu de données vide : {str(e)}")

    return df_zero

# Lecture du jeu de données avec une observation à zero

@st.cache_data
def read_stats_ech():

    df_evol_grav, stats_expl_var = None, None

    try:
        df_evol_grav   = pd.read_csv(rep_processed + "evol_grav.csv", sep='\t')
        stats_expl_var = pd.read_csv(rep_processed + "stats_expl_var.csv", sep='\t')
        ech_dfc = pd.read_csv(rep_raw + "ech_caracteristiques.csv", sep = ',')
        ech_dfl = pd.read_csv(rep_raw + "ech_lieux.csv", sep = ',')
        ech_dfu = pd.read_csv(rep_raw + "ech_usagers.csv", sep = ',')
        ech_dfv = pd.read_csv(rep_raw + "ech_vehicules.csv", sep = ',')

    except Exception as e:
        st.error(f"Erreur lors du chargement du jeu de données : {str(e)}")

    return df_evol_grav, stats_expl_var, ech_dfc, ech_dfl, ech_dfu, ech_dfv

@st.cache_data
def read_info():

    desc_fic_raw, desc_vars = None, None

    try:
        with open(rep_ref + "desc_fic_raw.json", 'r', encoding='utf-8') as fichier:
            desc_fic_raw = json.load(fichier)

        with open(rep_ref + "desc_vars.json", 'r', encoding='utf-8') as fichier:
            desc_vars = json.load(fichier)

    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier d'information : {str(e)}")

    return desc_fic_raw, desc_vars

desc_fic_raw, desc_vars = read_info()

##############################################################################
#
# En-tête : Titre du projet
#
##############################################################################

st.title ("Accidentologie")

##############################################################################
#
# Page : Présentation du projet
#
##############################################################################

if page == pages[0] :
    st.header("Présentation du projet")
    st.subheader("Notre mission")
    st.write(
"""
Nous présentons notre projet de machine learning réalisé lors de notre formation dispensée par DataScientest.

Ce projet porte sur le thème des accidents de la route en France au cours de la période de 2005 à 2022.
Les données ainsi que leur description sont disponibles sur le site www.data.gouv.fr.

Ces données concernent 72 dataframes au total, soit 1 dataframe par année et par rubrique.

Notre mission consiste à explorer, préparer et modéliser le jeu de données dans le but de **prédire
la gravité des accidents routiers** en fonction des circonstances qui les entourent.

Avec ce projet, nous mettons en pratique les acquis de notre formation en réalisant un projet de data science complet.
"""
)


##############################################################################
#
#  Le jeu de données
#
##############################################################################

if page == pages[1]:
    st.header("Le jeu de données")
    st.markdown(
"""
Les données concernent 72 dataframes au total, soit 1 dataframe par année et par rubrique.

Un changement de codage de la gravité entre 2018 et 2019 nous contraint
à ne retenir que les données de 2019 à 2022, soit les quatre dernières années.

Les données sont réparties dans les 4 rubriques suivantes :

* caracteristiques : qui prend en compte les circonstances générales de l’accident.
* lieux : qui décrit l’endroit de l’accident.
* vehicules : qui énonce les véhicules impliqués dans l’accident.
* usagers : qui relate les usagers impliqués dans l’accident.
"""
    )
    st.image("reports/figures/diagramme.png", caption = "Diagramme données", width = 500)

    st.markdown(
"""
En raison de la grande variété de structuration des données, une uniformisation a été nécessaire pour faciliter leur exploration.
"""
)
    
    st.write("")
    st.image("reports/figures/exploration_11.png", caption = "Processus préalable à l'exploration des données", width = 700)

##############################################################################
#
# Page : Visualisation des données
#
##############################################################################

if page == pages[2]:

    df_evol_grav, stats_expl_var, ech_dfc, ech_dfl, ech_dfu, ech_dfv = read_stats_ech()

    st.header("Visualisation du jeu de données")

    st.subheader("Évolution sur 19 ans des nombres d'usagers impliqués")

    df_base100 = df_evol_grav
    df_base100["grav_1"] = 100. * df_base100["grav_1"] / df_base100.loc[0, "grav_1"]
    df_base100["grav_2"] = 100. * df_base100["grav_2"] / df_base100.loc[0, "grav_2"]
    df_base100["grav_3"] = 100. * df_base100["grav_3"] / df_base100.loc[0, "grav_3"]
    df_base100["grav_4"] = 100. * df_base100["grav_4"] / df_base100.loc[0, "grav_4"]
    # st.dataframe(df_base100)
    #st.write(f"Base grav_1 {df_base100.loc[0, 'grav_1']}")
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_base100 , x='annee', y='grav_1', label = "Indemmes", color = "green")
    sns.lineplot(data=df_base100 , x='annee', y='grav_2', label = "Tués", color = "red")
    sns.lineplot(data=df_base100 , x='annee', y='grav_3', label = "blessé hospitalisé", color = "orange")
    sns.lineplot(data=df_base100 , x='annee', y='grav_4', label = "blessé léger", color = "blue")
    plt.title("Évolution des nombres de personnes impliquées\ndans un accident de circulation selon la gravité")
    plt.xlabel("Année")
    plt.xticks (df_base100["annee"])
    plt.ylabel("Pourcentage")
    plt.legend(title='Modalités', loc = "upper center")
    plt.axvline(x=2019, color='black', linestyle=(0, (5, 5)), linewidth=2)
    plt.annotate("Changement\nde codification\nde notre var. cible", xy=(2019, 90),
                 xytext = (2019.5, 95),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5))

    plt.annotate("Effet COVID",
                 xy=(2020, 65),
                 xytext=(2019, 75),
                 arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5))

    plt.grid()
    st.pyplot(fig)
    annee = df_evol_grav["annee"].max()
    annee = st.selectbox("Année : ", df_evol_grav["annee"])

    mod_grav_nb = df_evol_grav[df_evol_grav["annee"] == annee][["grav_1", "grav_2", "grav_3", "grav_4"]]

    fig = plt.figure(figsize=(10, 6))
    plt.title("Répartition de la gravité")

    mod_grav_nb = mod_grav_nb.rename(columns = {"grav_1" : "Indemne", "grav_2" : "Tués",
                         "grav_3" : "Blessés hospitalisés", "grav_4" : "Blessés legers"})
    sns.barplot (mod_grav_nb)
    #plt.ylim(0, df_evol_grav[["grav_1", "grav_2", "grav_3", "grav_4"]].max())
    plt.ylim(0, df_evol_grav["grav_1"].max())

    st.pyplot(fig)

    #
    # Affichages pour chaque rubrique d'un échantillon et de quelques statistiques
    #

    lst_rub = {"caracteristiques": {"df" : ech_dfc, "label" : "caractéristiques"},
               "lieux": {"df" : ech_dfl, "label" : "lieux"},
               "usagers": {"df" : ech_dfu, "label" : "usagers"},
               "vehicules": {"df" : ech_dfv, "label" : "véhicules"}}

    for rub_clef, rub_desc in lst_rub.items():
        st.subheader(rub_clef)

        # Affichage de l'échantillon de données de la rubrique
        st.write("Échantillon des données de la rubrique")
        st.dataframe(rub_desc["df"])

        # Affichage d'un graphique de répartition d'une modalité
        variables = stats_expl_var[stats_expl_var["rubrique"] == rub_desc["label"]]["variable"].unique()
        variables_lib = [] # Liste des variables avec les libellés explicites
        for v in variables:
            if desc_vars[v].get("label") is not None:
                variables_lib.append(v + " : " + desc_vars[v].get("label"))
            else:
                variables_lib.append(v + " : ")
        chx_variable = st.selectbox("Variable : ", variables_lib)
        variable_df   = chx_variable.split(" : ")[0]
        variable_expl = chx_variable.split(" : ")[1]

        df1 = stats_expl_var[(stats_expl_var["rubrique"] == rub_desc["label"]) & (stats_expl_var["variable"] == variable_df)]
        df1["count"] = df1["count"].astype("int")
        df1 = df1[["modalite", "count"]]
        # st.dataframe (df1) # pour mise au point, à supprimer

        fig = plt.figure(figsize=(10, 6))
        plt.title(f"Répartition selon {chx_variable}")

        sns.barplot (df1, y = "modalite", x= "count", orient = "h")

        st.pyplot(fig)


##############################################################################
#
# Page : Pré processing
#
##############################################################################

if page == pages[3]:

    # Choix du type de preprocessing
    liste_preprocessing = ['Premier Preprocessing', 'Deuxième Preprocessing']
    choix_preprocessing = st.radio('Choix du Preprocessing', liste_preprocessing)

    if choix_preprocessing == liste_preprocessing[0]:
        st.write(liste_preprocessing[0])
        with st.expander("Premier preprocessing"):
            st.image (rep_figures + "preprocessing_10.png")

    if choix_preprocessing == liste_preprocessing[1]:
        st.write(liste_preprocessing[1])
        st.write("""
    La faible proportion de tués (2,64%) nous incite à regrouper les modalités de la cible
    en deux modalités "grave" et "non grave". La modalité "grave" correspond alors aux "Tués et hospitalisés plus de 24h"
    et représente 18% des observations. 
                 
    Aussi, nous décidons d'équilibrer notre jeu de données pour que chaque modalité de la cible soit également représentée,
    en échantillonnant (avec RandomUnderSampler()) les observations "non graves".
                 
    Cet équilibrage permet aussi de réduire les temps d'entraînement.
        """)
        with st.expander("Preprocessing") :
            st.image (rep_figures + "preprocessing_21.png")

        with st.expander("Preprocessing") :
            st.image (rep_figures + "preprocessing_22.png")


##############################################################################
#
# Modélisations : LR, SVC et DP
#
##############################################################################

if page == pages[4] :

    st.subheader("Un problème de classification")
    st.markdown ("""
    Notre objectif est de **prédire la gravité des accidents** de la circulation 
    pour les usagers en fonction des circonstances qui les entourent.

    Plus précisément, il s'agit de dire si un accident est grave ou non. Nous sommes donc en face 
    d'un problème de classification, pour lequel nous pouvons entrainer plusieurs modèles.
                 
    Pour cela, nous retiendrons la variable 'grav' comme variable cible.
        
    Nous examinerons la performance des modèles entraînés à travers plusieurs métriques de classification :
    - Accuracy (Taux de Précision globale) : proportion totale de prédictions correctes.
    - Precision (Taux d’Exactitude) : proportion des accidents qui sont réellement graves.
    - Recall (Taux de Rappel) : proportion d’accidents graves correctement identifiés parmi tous les
    accidents réellement graves.
    - F1-Score : moyenne harmonique entre la ’precision’ et le ’recall’.
    - AUC-ROC : capacité du modèle à distinguer les différentes classes entre elles.
    
    """)

    st.subheader("Régression logistique : un premier essai")
    st.write("""
    Nous réalisons un premier essai à partir du jeu de données issu du premier preprocessing, 
    dont la variable cible se divise en 4 classes :
    - "Indemne",
    - "Hospitalisé moins de 24h"
    - "Hospitalisé plus de 24h"
    - "Tué".
            """)

    with st.expander("Afficher le processus de logistique régression"):
        
        st.image(rep_figures + "model_lr.png", width=350)

    if st.checkbox("Afficher les meilleurs paramètres et score", key="lr_params"):
        st.write("Meilleurs paramètres: {'C': 1, 'l1_ratio': 0.4, 'max_iter': 1000, 'penalty': 'elasticnet'}")
        st.write("Meilleur score: 0.5895009263286214")
        st.write("Accuracy du meilleur modèle: 0.5923140628339388")
          
    st.markdown("""
            Cette modélisation obtient des **scores insuffisants** : 59% seulement.
            """)

    st.subheader("Machine learning : la recherche de performance")
    st.markdown("""
    Nous utilisons le jeu de données issu du deuxième preprocessing avec une cible binaire
    et équilibrée :
    - accident non grave (grav_grave de classe 0) = "Indemne" + "Hospitalisé moins de 24h" 
    - accident grave (grav_grave de classe 1) = "Hospitalisé plus de 24h" + "Tué".
    
    Un premier essai avec **lazypredict** nous donne quelques indications sur les performances.
    Nous essayons 11 modèles : des modèles découverts dans les cours, LGBM et même dummy proposés par lazypredict 
    (dummy est essayé pour comparaison).
    
    Le jeu de données contient environ 260 variables explicatives. Ce nombre est très important, alors
    nous décidons d'appliquer une PCA.
             
    Après avoir appliqué la PCA, nous normalisons le jeu de données de dimension réduite.
             
    Deux graphiques nous aident à choisir le nombre de composantes
    en affichant la variance expliquée par nombre de composantes.
             
    Nous choisissons 100 composantes afin d'avoir au moins 90% de variance expliquée
    pour ne pas diminuer les performances des modèles. 
    """)
    st.image(rep_figures + "choix_PCA.png")

    st.write("""
            Nous entraînons alors les modèles. Le code utilise un dictionnaire des modèles avec les paramètres à essayer,
            une boucle d'essais avec GridSearchCV et des affichages de nombreuses métriques. 
            
            Le dictionnaire a permis très simplement d'ajouter des modèles et de faire des essais de paramètres.
            Nous affichons de nombreuses métriques pour les étudier, finalement le graphe des vrais/faux positifs/négatifs
            nous paraît le plus clair.
            """)

    st.write("""
    Le modèle SVC semble le plus performant pour trouver "le plus sûrement" les causes d'accident graves pour les usagers.
    Il a le meilleur recall (82%), la plus faible proportion de faux négatifs; il présente toutefois un surapprentissage 
    trop important.
            
    Le modèle LGBM a des scores proches et un faible surapprentissage; il a l'avantage d'être très rapide à entraîner
    et d'occuper peu de place sur le disque.
    """)
    st.image(rep_figures + "comparaison_scores.png")
    st.image(rep_figures + "evaluation_performances.png")

    st.write ("""
    Enfin, pour évaluer les modèles et vérifier la cohérence des prédictions avec les variables explicatives,
    nous utilisons SHAP pour afficher leurs liens sur un graphique.
              
    Il apparaît très clairement que le port du casque ou de la ceinture de sécurité sont les éléments les plus
    importants pour éviter des conséquences graves. 
    """)
    st.image(rep_figures + "shap_summary_LGBM.png")

    st.subheader("Deep learning : un modèle plus efficace")
    st.write("")
    st.write("""
        Nous utilisons à nouveau le jeu de données issu du deuxième preprocessing avec une cible binaire (grave ou non grave).
        """)
    
    with st.expander("Afficher les détails pour le deep learning"):

        st.image(rep_figures + "model_dl.png", width=350)

    st.write("")
    st.write("Performances du modèle :")
    if st.checkbox("Afficher l'évaluation sur l'ensemble de test :", key="dl_eval"):
        st.write("Test loss: 0.4792")
        st.write("Test accuracy: 0.7781")

    if st.checkbox("Afficher le rapport de classification", key="dl_class_report"):
        dl_class_rep = pd.read_csv(rep_processed + "dl_class_report.csv", sep = '\t', index_col=0)
        st.dataframe(dl_class_rep)

    if st.checkbox("Afficher la matrice de confusion", key="dl_conf_matrix"):
        st.write("[[13240  4525]")
        st.write("[ 3358 14406]]")

    if st.checkbox(f"Afficher le score AUC-ROC") :
        st.write("0.8570")

    st.write("")
    st.write("Visualisation graphique :")    
    st.write("")
    st.image(rep_figures + "training_history.png", caption="Historique d'entraînement", width=680)
    st.write("=" * 84)

    st.write("")
    st.write("Voici les caractéristiques qui ont le plus d’impact (global) sur les prédictions du modèle :")
    st.image(rep_figures + "feature_importance.png", caption="Features importances", width=680)
    st.write("Par exemple, 'obsm_1' est la caractéristique la plus importante, liée à un type d’obstacle mobile impliqué dans l’accident.")

    st.write("")
    st.write("Analyse SHAP :")   
    st.write("")
    st.write("Voici les caractéristiques qui ont le plus d’impact (marginal) sur les prédictions du modèle :")
    st.image(rep_figures + "shap_summary_plot.png", caption="Labels explicatifs", width=680)
    st.markdown("""Les points sur chaque ligne nous disent deux choses :
            
    * Leur position :
    - À gauche, ils réduisent le risque d’accident grave.
    - À droite, ils augmentent ce risque.

    * Leur couleur :
    - Rouge signifie une valeur élevée pour ce facteur.
    - Bleu signifie une valeur basse.
    
    Par exemple, pour la ceinture de sécurité (deuxième ligne) :
    - Les points rouges à gauche signifient que le port de la ceinture (valeur élevée) 
    tend à réduire la gravité de l’accident.
    - Les points bleus à droite indiquent que l’absence de ceinture (valeur basse) 
    tend à augmenter la gravité de l’accident
        """)
    st.write("")



##############################################################################
#
# Page : Démonstration avec deep learning
#
# Cette page présente :
#  - des choix de circonstances;
#  - ? ? ? le choix du modèle  ? ? ?;
#  - Les valeurs données au modèle ;
#  - la prédiction de la gravité avec le ou les modèles;
#  - la probabilité de la gravité
#
##############################################################################


if page == pages[5]:

    def load_and_preprocess_data(data_path):
        df = pd.read_csv(data_path, sep='\t')
        df = df.astype(int)
        X = df.drop('grav_grave', axis=1)
        y = df['grav_grave']
        return df, X, y

    # Charger les données
    data_path = os.path.join(rep_processed, "EchData.csv")
    df, X, y = load_and_preprocess_data(data_path)
    
    # Obtenir le nombre de features pour input_shape
    input_shape = X.shape[1]

    def create_and_compile_model():
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', 
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])
        return model  

    # Vérifier si le modèle et le scaler existent
    model_architecture_path = rep_models + 'model_architecture.json'
    model_weights_path = rep_models + 'model.weights.h5'
    scaler_path = rep_models + 'scaler_DP.joblib'


    if os.path.exists(model_architecture_path) and os.path.exists(model_weights_path) and os.path.exists(scaler_path):  # Changement ici
        try:
            # Charger l'architecture du modèle
            with open(model_architecture_path, 'r') as f:
                model_json = f.read()
            loaded_model = model_from_json(model_json)
            
            # Charger les poids
            loaded_model.load_weights(model_weights_path)
            
            # Compiler le modèle
            loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            # Charger le scaler
            scaler = joblib.load(scaler_path)
            
            st.success("Modèle chargé avec succès!")
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle: {str(e)}")
            loaded_model = None
            scaler = None
    else:
        st.warning("Modèle non trouvé. Entraînement d'un nouveau modèle nécessaire.")
        loaded_model = None
        scaler = None

    if loaded_model is None:
        if st.button("Entraîner un nouveau modèle"):
            with st.spinner("Entraînement en cours..."):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Créer et entraîner le modèle
                loaded_model = create_and_compile_model()
                loaded_model.fit(X_train_scaled, y_train, 
                                epochs=50, 
                                batch_size=32, 
                                verbose=0)
                
                # Sauvegarder l'architecture du modèle
                model_json = loaded_model.to_json()
                with open(rep_models + 'model_architecture.json', 'w') as f:
                    f.write(model_json)
                
                # Sauvegarder les poids avec le nouveau format de nom
                loaded_model.save_weights(rep_models + 'model.weights.h5')
                
                # Sauvegarder le scaler
                joblib.dump(scaler, rep_models + 'scaler_DP.joblib')
                
                st.success("Modèle entraîné et sauvegardé avec succès!")
    
    
    if loaded_model is not None and scaler is not None:
        # Interface de prédiction
        st.header("Prédiction de la gravité d'un accident")
        features = pd.DataFrame(0, index=[0], columns=X.columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Informations sur l'usager")
            
            # Sexe
            sexe_dict = {"Homme": "sexe_m", "Femme": "sexe_f"}
            sexe = st.radio("Sexe", options=list(sexe_dict.keys()))
            for col in features.columns:
                if col.startswith('sexe_'):
                    features[col] = 0
            features[sexe_dict[sexe]] = 1
            
            st.write("")
            
            # Catégorie d'usager
            catu_dict = {"Conducteur": "catu_1", "Passager": "catu_2", "Piéton": "catu_3"}
            catu = st.radio("Catégorie d'usager", options=list(catu_dict.keys()))
            for col in features.columns:
                if col.startswith('catu_'):
                    features[col] = 0
            features[catu_dict[catu]] = 1

        with col2:
            st.subheader("Conditions de l'accident")
            
            # Mois
            mois_dict = {
                "Janvier": "mois_1", "Février": "mois_2", "Mars": "mois_3", 
                "Avril": "mois_4", "Mai": "mois_5", "Juin": "mois_6",
                "Juillet": "mois_7", "Août": "mois_8", "Septembre": "mois_9",
                "Octobre": "mois_10", "Novembre": "mois_11", "Décembre": "mois_12"
            }
            mois = st.selectbox("Mois", options=list(mois_dict.keys()))
            for col in features.columns:
                if col.startswith('mois_'):
                    features[col] = 0
            features[mois_dict[mois]] = 1

            # Jour de semaine
            jsem_dict = {
                "lundi": "jsem_1", "mardi": "jsem_2", "mercredi": "jsem_3", 
                "jeudi": "jsem_4", "vendredi": "jsem_5", "samedi": "jsem_6",
                "dimanche": "jsem_7"}
            jsem = st.selectbox("Jour de semaine", options=list(jsem_dict.keys()))
            for col in features.columns:
                if col.startswith('jsem_'):
                    features[col] = 0
            features[jsem_dict[jsem]] = 1

            # Dispositif de sécurité
            secu_dict = {
                "Ceinture": "secu_ceinture", "Casque": 'secu_casque', "Dispositif enfant": 'secu_dispenfant', 
                "Gilet de sécurité": 'secu_gilet', "Airbag": 'secu_airbag23RM', "Gants": 'secu_gants'}
            secu = st.selectbox("Dispositif de sécurité", options=list(secu_dict.keys()))
            for col in features.columns:
                if col.startswith('secu_'):
                    features[col] = 0
            features[secu_dict[secu]] = 1

            # Vitesse maximale autorisée
            vma_dict = {
                "30 km/h": "vma_30m", "40 km/h": 'vma_40', "50 km/h": 'vma_50', "60 km/h": 'vma_60', "70 km/h": 'vma_70',
                "80 km/h": 'vma_80', "90 km/h": 'vma_90', "110 km/h": 'vma_100', "130 km/h": 'vma_130' }
            vma = st.select_slider("Vitesse maximale autorisée (km/h)", options=list(vma_dict.keys()))
            for col in features.columns:
                if col.startswith('vma_'):
                    features[col] = 0
            features[vma_dict[vma]] = 1

            # Catégorie route
            catr_dict = {
                "Autoroute": "catr_1", "Route nationale": "catr_2", "Route départementale": "catr_3", 
                "Voie communale": "catr_4", "Hors réseau public": "catr_5", "Parc de stationnement ouvert à la circulation publique": "catr_6",
                "Route de métropole urbaine": "catr_7"}
            catr = st.selectbox("Catégorie de route", options=list(catr_dict.keys()))
            for col in features.columns:
                if col.startswith('catr_'):
                    features[col] = 0
            features[catr_dict[catr]] = 1

            # Motorisation du véhicule
            motor_dict = {
                "Hydrocarbure": "motor_1", "Hybride électrique": "motor_2", "Electrique": "motor_3", "Hydrogène": "motor_4"}
            motor = st.selectbox("Motorisation du véhicule", options=list(motor_dict.keys()))
            for col in features.columns:
                if col.startswith('motor_'):
                    features[col] = 0
            features[motor_dict[motor]] = 1

            # Situation de l'accident
            situ_dict = {
                "Sur chaussée": "situ_1", "Sur bande d'arrêt d'urgence": "situ_2", "Sur accotement": "situ_3", 
                "Sur trottoir": "situ_4", "Sur piste cyclable": "situ_5", "Sur autre voie spéciale": "situ_6"}
            situ = st.selectbox("Situation de l'accident", options=list(situ_dict.keys()))
            for col in features.columns:
                if col.startswith('situ_'):
                    features[col] = 0
            features[situ_dict[situ]] = 1

            # Obstacle mobile heurté
            obsm_dict = {
                "Piéton": "obsm_1", "Véhicule": "obsm_2", "Véhicule sur rail": "obsm_4", 
                "Animal domestique": "obsm_5", "Animal sauvage": "obsm_6"}
            obsm = st.selectbox("Obstacle mobile heurté", options=list(obsm_dict.keys()))
            for col in features.columns:
                if col.startswith('obsm_'):
                    features[col] = 0
            features[obsm_dict[obsm]] = 1

            # Type d'intersection
            infra_dict = {
                "Souterrain - tunnel": "int_1", "Pont - autopont": "int_2", "Bretelle de raccordement": "int_3", 
                "Voie ferrée": "int_4", "Carrefour aménagé": "int_5", "Zone piétonne": "int_6", "Zone de péage": "int_7", 
                "Chantier": "int_8"}
            infra = st.selectbox("Type d'intersection", options=list(infra_dict.keys()))
            for col in features.columns:
                if col.startswith('infra_'):
                    features[col] = 0
            features[infra_dict[infra]] = 1
            
            # Conditions atmosphériques
            atm_dict = {
                "Normale": "atm_1", "Pluie légère": "atm_2", "Pluie forte": "atm_3", 
                "Neige - grêle": "atm_4", "Brouillard - fumée": "atm_5", "Vent fort - tempête": "atm_6",
                "Temps éblouissant": "atm_7", "Temps couvert": "atm_8"
            }
            atm = st.selectbox("Conditions atmosphériques", options=list(atm_dict.keys()))
            for col in features.columns:
                if col.startswith('atm_'):
                    features[col] = 0
            features[atm_dict[atm]] = 1
            
            # Luminosité
            lum_dict = {
                "Plein jour": "lum_1", "Crépuscule - aube": "lum_2", 
                "Nuit sans éclairage public": "lum_3",
                "Nuit avec éclairage public non allumé": "lum_4", 
                "Nuit avec éclairage public allumé": "lum_5"
            }
            lum = st.selectbox("Luminosité", options=list(lum_dict.keys()))
            for col in features.columns:
                if col.startswith('lum_'):
                    features[col] = 0
            features[lum_dict[lum]] = 1
        
        # Bouton de prédiction
        if st.button("Prédire la gravité"):
                try:
                    # Normalisation des données
                    features_scaled = scaler.transform(features)
                    
                    # Prédiction
                    with st.spinner("Calcul de la prédiction en cours..."):
                        prediction_proba = loaded_model.predict(features_scaled)
                        prediction = (prediction_proba > 0.5).astype(int)
                    
                    # Affichage du résultat
                    st.subheader("Résultat de la prédiction")
                    
                    proba_grave = float(prediction_proba[0])
                    proba_non_grave = 1 - proba_grave
                    
                    st.write(f"Probabilité accident non grave: {proba_non_grave:.1%}")
                    st.write(f"Probabilité accident grave: {proba_grave:.1%}")
                    
                    if prediction[0] == 1:
                        st.error(f"⚠️ Risque élevé d'accident grave (Probabilité : {proba_grave:.1%})")
                    else:
                        st.success(f"✅ Risque faible d'accident grave (Probabilité : {proba_grave:.1%})")

                
                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {str(e)}")
                    st.write("Debug - Features utilisées:", features.columns[features.iloc[0] == 1].tolist())
