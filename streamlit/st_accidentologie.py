import streamlit as st

import pandas as pd
import numpy as np

from sklearn.decomposition         import PCA
from sklearn.preprocessing         import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

# TODO  activer après compilation de tensorflow
# import tensorflow as tf

import joblib

st.sidebar.title("Sommaire")

pages=["Présentation", "Data Vizualization", "Modélisation pas SVC", "Modélisation par Deep Learning"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.write("Erika Méronville\n&\nEmmanuel Gautier")

###############################################################################
#
#  Lecture des données.
#
#  Cette lecture est réalisée par la fonction read_data().
#  Pour éviter la relecture à chaque commande elle est décorée par @st.cache_data
#
###############################################################################

@st.cache_data
def read_data():

    SVC, pca, X_zero, scaler, DP, scaler_DP = None, None, None, None, None, None

    try : 
        SVC = joblib.load("models/SVC.mdl")
        pca = joblib.load("models/pca_ml.mdl")
    except Exception as inst:
        st.write (inst.args)
        st.write (inst)
        SVC, pca = None, None

    try:
        scaler_ml = joblib.load("models/acc_scaler.mdl")

        scaler_DP = joblib.load("models/scaler_DP.joblib")

        # TODO activer après compilation de tensorflow
        # DP = tf.saved_model.load("models/acc_DeepLearning.h5")

    except Exception as inst:
        st.write (inst.args)
        st.write (inst)
        DP = None

    # Lectures des synthèses pour la visualisation de données
    # Données synthétiques car plus rapide et faible espace de stockage
    try:
        df_zero        = pd.read_csv("data/processed/data_zero.csv",      sep = '\t')
        df_evol_grav   = pd.read_csv("data/processed/evol_grav.csv",      sep = '\t')
        stats_expl_var = pd.read_csv("data/processed/stats_expl_var.csv", sep = '\t')
        # st.write (f" --> df_zero 1 : {df_zero.shape[0]} x {df_zero.shape[1]}")
        X_zero = df_zero.drop(["grav_grave", "agg"], axis = 1)
        # st.write (f" --> df_zero 2 : {df_zero.shape[0]} x {df_zero.shape[1]}")
        #X_zero = df_zero.drop("agg",        axis = 1)

    except Exception as inst:
        st.write (inst.args)
        st.write (inst)
        SVC, pca = None, None
        pass

    # La fonction est "décorée" avec @st.cache_data pour éviter les relectures
    # à chaque affichage d'une page. Les résultats renvoyés avec return sont
    # alors mis en cache. les resultats sont donc mis dans un dictionnaire.
    return {"SVC" : SVC, "PCA" : pca, "X_zero" : X_zero,
            "scaler_ml" : scaler_ml, "DP" : DP,
            "evol_grav" : df_evol_grav,
            "scaler_DP" : scaler_DP,
            "stats_expl_var" : stats_expl_var}

llll = read_data()
SVC = llll["SVC"]
DP = llll["DP"]
pca = llll["PCA"]
X_zero = llll["X_zero"]
scaler_ml = llll["scaler_ml"]
scaler_DP = llll["scaler_DP"]
df_evol_grav = llll["evol_grav"]
stats_expl_var = llll["stats_expl_var"]


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

ce projet porte sur le thème des accidents de la route en France au cours de la période de 2005 à 2022.
Les données ainsi que leur description sont disponibles sur le site www.data.gouv.fr.

Ces données concernent 72 dataframes au total, soit 1 dataframe par année et par rubrique.
Un changement de codage de la gravité entre 2018 et 2019 nous contraint de ne retenir que les données de 2019 à 2022, soit les quatre dernières années.

Notre mission consiste à explorer, préparer et modéliser le jeu de données dans le but de prédire
la gravité des accidents routiers en fonction des circonstances qui les entourent.
"""
)
    st.subheader("Notre progression")
    st.write(
"""
    Nous avons déjà réalisé l'exploration, la préparation et la modélisation des données.
    Ce travail doit être accessible à tous, c'est l'objet de cette présentation avec streamlit en cours de réalisation.

    C'est un travail en cours de réalisation; nous développons en ce moment à l'affichage
    de prédictions avec  des valeurs saisies et ramanions le code.

    Avec ce projet nous mettons en pratique les acquis de notre formation en réalisant un projet de data science complet.
"""
)

##############################################################################
#
# Page : Visualisation des données
#
##############################################################################

if page == pages[1] :
    st.header("Visualisation du jeu de données")

    st.subheader ("Évolution sur 19 ans des nombres d'usagers impliqués")

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

    st.subheader ("Répartitions par modalités")

    # TODO : trier la liste des rubriques par ordre alphabétique
    rubriques = stats_expl_var["rubrique"].unique()
    rubrique = st.selectbox("Rubrique : ", rubriques)
    variables = stats_expl_var[stats_expl_var["rubrique"] == rubrique]["variable"].unique()
    variable = st.selectbox("Variable : ", variables)

    df1 = stats_expl_var[(stats_expl_var["rubrique"] == rubrique) & (stats_expl_var["variable"] == variable)]
    df1["count"] = df1["count"].astype("int")
    df1 = df1[["modalite", "count"]]
    # st.dataframe (df1) # pour mise au point, à supprimer

    fig = plt.figure(figsize=(10, 6))
    plt.title(f"Répartition selon {variable}")

    sns.barplot (df1, y = "modalite", x= "count", orient = "h")

    st.pyplot(fig)

##############################################################################
#
# Modélisation : SVC
#
##############################################################################

if page == pages[2] :
    st.header("Modélisation par SVC")
    st.subheader("Performances du modèle")

    Xp = X_zero.copy()


    choix = ["Conducteur", "Passager", "Piéton"]
    catu = st.selectbox("Catégorie d'usager", choix)
    st.write(f"Catégorie d'usager {catu}")
    if catu == choix[0] : 
        Xp["catu_1"]  = 1
    elif catu== choix[1] : 
        Xp["catu_2"]  = 1
    else:
        Xp["catu_3"]  = 1

    #choix = ["Normale", "Mouillée", "glissante", "autre"]
    #chps = ["surf_norm", "surf_mouil", "surf_gliss", "surf_autre"]
    #surf = st.selectbox("Catégorie d'usager", choix)
    #chp = chps[choix.index(surf)]
    #Xp[chp] = 1
    #st.write (f"choix : {surf} champ : {chp}")
    # st.dataframe(df_0)
  
    st.write ("Développement en cours.")
    st.write ("Affichage du DataFrame avec une seule 'observation' destiné à la prédiction")

    st.dataframe(Xp)
 
    #st.write ("Avec la prédiction réalisée nous afficherons le résultat")

    #Xp = pca.transform(df_0)
    #Xp = scaler.transform(Xp)
    #ypred = SVC.predict(Xp)

    #st.write(f"Prédiction : {ypred}")


##############################################################################
#
# Modélisation : Deep Learning
#
##############################################################################

if page == pages[3] :
    st.write("### Modélisation par Deep Learning")

    Xp = X_zero.copy()

    #----- Choix catu : Catégorie d'usager : conducteur, passager ou piéton -----

    choix = ["Conducteur", "Passager", "Piéton"]
    catu = st.selectbox("Catégorie d'usager", choix)
    st.write(f"Catégorie d'usager {catu}")
    if catu == choix[0] : 
        Xp["catu_1"]  = 1
    elif catu== choix[1] : 
        Xp["catu_2"]  = 1
    else:
        Xp["catu_3"]  = 1

    #----- Choix surf : État de la surface -----
    choix = ["Normale", "Mouillée", "glissante", "autre"]
    chps = ["surf_norm", "surf_mouil", "surf_gliss", "surf_autre"]
    surf = st.selectbox("Catégorie d'usager", choix)
    chp = chps[choix.index(surf)]
    Xp[chp] = 1
    st.write (f"choix : {surf} champ : {chp}")


    st.dataframe(Xp)

    Xp = scaler_DP.transform(Xp)

    st.write ("Développement en cours")

    # TODO Activer après compilation de tensorflow
    # ypred = DP.predict(Xp)

    #st.write(f"Prédiction : {ypred}")

