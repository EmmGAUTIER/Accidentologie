import streamlit as st

import pandas as pd
import numpy as np
from sklearn.decomposition         import PCA
from sklearn.preprocessing         import StandardScaler

import joblib

st.title ("Accidentologie")
st.sidebar.title("Sommaire")
pages=["Présentation", "Data Vizualization", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

###############################################################################
#
#  Lecture des données.
#
#  Cette lecture est réalisée par la fonction read_data().
#  Pour éviter la relecture à chaque commande elle est décorée par @st.cache
#
###############################################################################

@st.cache_data
def read_data():
    dfd = pd.read_csv("data/processed/data.csv", sep = '\t', index_col = None)
    X = dfd.drop(['grav_grave'], axis = 1)
    y= dfd.grav_grave
    pca = PCA(n_components=100)
    pca.fit(X)
    X = pca.transform(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    SVC = joblib.load("models/SVC.mdl")
    #pca = joblib.load()
    return {"data" : dfd, "SVC" : SVC, "PCA" : pca, "X" : X, "scaler" : scaler}

llll = read_data()
SVC = llll["SVC"]
df = llll["data"]
pca = llll["PCA"]
X = llll["X"]
scaler = llll["scaler"]

##############################################################################
#
# Page : Présentation du projet
#
##############################################################################

if page == pages[0] :
    st.write("### Présentation")
    st.write("C'est notre travail.")

##############################################################################
#
# Page : Visualisation des données
#
##############################################################################

if page == pages[1] :
    st.write("### Visualisation")
    # L'affichage du DataFrame n'est pas interressant à cause du trop grand nombre de variables.
    # st.dataframe(df.head(10))
    st.write (f"Nombre d'observation : {df.shape[0]}")
    st.write (f"Nombre de variables  : {df.shape[1]}")

    choix = ["Conducteur", "Passager", "Piéton"]
    catu = st.selectbox("Catégorie d'usager", choix)

    nom_var = "catu_"+str(choix.index(catu) + 1)
    valc = df[df[nom_var] == True].grav_grave.value_counts()

    st.write (valc)

##############################################################################
#
# Modélisation
#
##############################################################################

if page == pages[2] :
    st.write("### Modélisation")

    df_0 = pd.DataFrame([[0] * len(df.columns)], columns=df.columns)
    df_0 = df_0.drop(['grav_grave'], axis = 1)

    choix = ["Conducteur", "Passager", "Piéton"]
    catu = st.selectbox("Catégorie d'usager", choix)
    st.write(f"Catégorie d'usager {catu}")
    if catu == choix[0] : 
        df_0["catu_1"]  = 1
    elif catu== choix[1] : 
        df_0["catu_2"]  = 1
    else:
        df_0["catu_3"]  = 1

    choix = ["Normale", "Mouillée", "glissante", "autre"]
    chps = ["surf_norm", "surf_mouil", "surf_gliss", "surf_autre"]
    surf = st.selectbox("Catégorie d'usager", choix)
    chp = chps[choix.index(surf)]
    df_0[chp] = 1
    st.write (f"choix : {surf} champ : {chp}")


    st.dataframe(df_0)
  
    Xp = pca.transform(df_0)
    Xp = scaler.transform(Xp)
    ypred = SVC.predict(Xp)

    st.write(f"Prédiction : {ypred}")

