import streamlit as st

import pandas as pd
import numpy as np

from sklearn.decomposition         import PCA
from sklearn.preprocessing         import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

import joblib

st.title ("Accidentologie")
st.sidebar.title("Sommaire")
pages=["Présentation", "Data Vizualization", "Modélisation pas SVC", "Modélisation par Deep Learning"]
page=st.sidebar.radio("Aller vers", pages)

columns =["ferie","secu_ceinture","secu_casque","secu_dispenfant","secu_gilet","secu_airbag23RM","secu_gants","age_enfant","age_jeune","age_adulte","age_3age","hr_matin","hr_midi","hr_am","hr_soir","hr_nuit","sexe_m","sexe_f","nbv_1","nbv_2","nbv_3",
"nbv_4","nbv_plus","surf_norm","surf_mouil","surf_gliss","surf_autre","vma_30m","vma_40","vma_50","vma_60","vma_70","vma_80","vma_90","vma_110","vma_130","actp_1","actp_2","actp_3","actp_4","actp_5",
"actp_6","actp_7","actp_8","actp_9","actp_A","actp_B","atm_1","atm_2","atm_3","atm_4","atm_5","atm_6","atm_7","atm_8","catr_1","catr_2","catr_3","catr_4","catr_5","catr_6","catr_7","catu_1","catu_2",
"catu_3","catv_7","catv_17","catv_33","catv_42","catv_30","catv_37","catv_32","catv_50","catv_38","catv_10","catv_1","catv_40","catv_15","catv_14","catv_99","catv_2","catv_80","catv_34","catv_60",
"catv_31","catv_21","catv_3","catv_13","catv_20","catv_43","catv_36","catv_39","catv_16","catv_35","catv_41","choc_5","choc_3","choc_1","choc_4","choc_2","choc_8","choc_6","choc_7","choc_9",
"circ_3","circ_1","circ_2","circ_4","col_2","col_6","col_4","col_3","col_5","col_7","col_1","etatp_1","etatp_2","etatp_3","infra_2","infra_9","infra_1","infra_5","infra_4","infra_6","infra_3","infra_8",
"infra_7","int_1","int_3","int_9","int_4","int_2","int_6","int_5","int_7","int_8","jsem_6","jsem_4","jsem_5","jsem_3","jsem_1","jsem_2","jsem_7","locp_2","locp_3","locp_1","locp_5","locp_4","locp_8",
"locp_9","locp_6","locp_7","lum_4","lum_3","lum_1","lum_5","lum_2","manv_23","manv_11","manv_2","manv_21","manv_1","manv_9","manv_26","manv_15","manv_17","manv_4","manv_12","manv_16","manv_19","manv_13",
"manv_14","manv_3","manv_10","manv_5","manv_24","manv_18","manv_20","manv_7","manv_22","manv_25","manv_6","manv_8","mois_11","mois_9","mois_7","mois_2","mois_1","mois_5","mois_4","mois_8","mois_6",
"mois_10","mois_3","mois_12","motor_1","motor_6","motor_3","motor_5","motor_2","motor_4","obs_1","obs_4","obs_14","obs_9","obs_6","obs_15","obs_13","obs_8","obs_2","obs_16","obs_12","obs_3","obs_7",
"obs_17","obs_11","obs_5","obs_10","obsm_2","obsm_1","obsm_9","obsm_6","obsm_4","obsm_5","place_2","place_1","place_10","place_3","place_4","place_7","place_9","place_6","place_8","place_5","plan_2",
"plan_3","plan_1","plan_4","prof_1","prof_4","prof_2","prof_3","senc_2","senc_1","senc_3","senc_0","situ_1","situ_2","situ_4","situ_6","situ_8","situ_3","situ_5","trajet_0","trajet_5","trajet_9","trajet_1","trajet_4","trajet_2","trajet_3","vosp_0","vosp_1","vosp_3","vosp_2"]

df_0 = pd.DataFrame([[0] * len(columns)], columns=columns)

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

    dfd, SVC, pca, X, scaler, DP = None, None, None, None, None, None

    try:
        # dfd = pd.read_csv("data/processed/data.csv", sep = '\t', index_col = None)
        # X = dfd.drop(['grav_grave'], axis = 1)
        # y= dfd.grav_grave
        # pca = PCA(n_components=100)
        # pca.fit(X)
        # X = pca.transform(X)
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)
        pass
    except Exception :
        dfd, X = None, None

    try : 
        SVC = joblib.load("models/SVC.mdl")
        #pca = joblib.load()
    except Exception:
        SVC, pca = None, None

    try:
        scaler = joblib.load("../models/acc_scaler.mdl")
        DP = tf.saved_model.load("../models/acc_DeepLearning.h5")
    except Exception:
        DP = None

    ##########################################################################
    # Lectures des synthèses pour la visualisation de données
    #
    # Données synthétiques car plus rapide et faible espace de stockage
    #
    try:
        df_evol_grav = pd.read_csv("data/processed/evol_grav.csv", sep = '\t')
    except Exception:
        df_evol_grav = None

    # La fonction est "décorée" avec @st.cache_data pour éviter les relectures
    # à chaque affichage d'une page. Les résultats renvoyés avec return sont
    # alors mis en cache. les resultats sont donc mis dans un dictionnaire.
    return {"data" : dfd, "SVC" : SVC, "PCA" : pca, "X" : X,
            "scaler" : scaler, "DP" : DP, "evol_grav" : df_evol_grav }

llll = read_data()
SVC = llll["SVC"]
df = llll["data"]
pca = llll["PCA"]
X = llll["X"]
scaler = llll["scaler"]
df_evol_grav = llll["evol_grav"]

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

    st.write ("## Évolution sur 18 ans des nombres d'usagers impliqués")

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
    plt.title("Évolution des nombres de personnes impliquées dans un accident de circulation selon gravité")
    plt.xlabel("Année")
    plt.xticks (df_base100["annee"])
    plt.ylabel("Pourcentage")
    plt.legend(title='Position')
    plt.grid()
    #plt.show()
    st.pyplot(fig)


    #choix = ["Conducteur", "Passager", "Piéton"]
    #catu = st.selectbox("Catégorie d'usager", choix)

    #nom_var = "catu_"+str(choix.index(catu) + 1)
    #valc = df[df[nom_var] == True].grav_grave.value_counts()

    #st.write (valc)

##############################################################################
#
# Modélisation : SVC
#
##############################################################################

if page == pages[2] :
    st.write("### Modélisation par SVC")

    Xp = df_0

    choix = ["Conducteur", "Passager", "Piéton"]
    catu = st.selectbox("Catégorie d'usager", choix)
    st.write(f"Catégorie d'usager {catu}")
    if catu == choix[0] : 
        Xp["catu_1"]  = 1
    elif catu== choix[1] : 
        Xp["catu_2"]  = 1
    else:
        Xp["catu_3"]  = 1

    choix = ["Normale", "Mouillée", "glissante", "autre"]
    chps = ["surf_norm", "surf_mouil", "surf_gliss", "surf_autre"]
    surf = st.selectbox("Catégorie d'usager", choix)
    chp = chps[choix.index(surf)]
    Xp[chp] = 1
    st.write (f"choix : {surf} champ : {chp}")


    # st.dataframe(df_0)
  
    Xp = pca.transform(df_0)
    Xp = scaler.transform(Xp)
    ypred = SVC.predict(Xp)

    st.write(f"Prédiction : {ypred}")


##############################################################################
#
# Modélisation : Deep Learning
#
##############################################################################

if page == pages[3] :
    st.write("### Modélisation par Deep Learning")

    Xp = df_0

    choix = ["Conducteur", "Passager", "Piéton"]
    catu = st.selectbox("Catégorie d'usager", choix)
    st.write(f"Catégorie d'usager {catu}")
    if catu == choix[0] : 
        Xp["catu_1"]  = 1
    elif catu== choix[1] : 
        Xp["catu_2"]  = 1
    else:
        Xp["catu_3"]  = 1

    choix = ["Normale", "Mouillée", "glissante", "autre"]
    chps = ["surf_norm", "surf_mouil", "surf_gliss", "surf_autre"]
    surf = st.selectbox("Catégorie d'usager", choix)
    chp = chps[choix.index(surf)]
    Xp[chp] = 1
    st.write (f"choix : {surf} champ : {chp}")

    Xp = scaler.transform(Xp)
    ypred = SVC.predict(Xp)

    st.write(f"Prédiction : {ypred}")

