import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import logging
import joblib
import math
import json
import csv
import os
import re

from PIL import Image
from tabulate import tabulate
from IPython.display import display, HTML
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


# Fonction pour charger les données avec mise en cache
@st.cache_data
def load_data():
    try:
        #caracteristiques = pd.read_csv("caracteristiques_raw_4.csv", sep="\t", index_col=0)
        #lieux = pd.read_csv("lieux_raw_4.csv", sep="\t", index_col=0)
        #usagers = pd.read_csv("usagers_raw_4.csv", sep="\t", index_col=0)
        #vehicules = pd.read_csv("vehicules_raw_4.csv", sep="\t", index_col=0)
        caracteristiques = pd.read_csv("ech_caracteristiques.csv", sep=",", index_col=False)
        lieux = pd.read_csv("ech_lieux.csv", sep=",", index_col=False)
        usagers = pd.read_csv("ech_usagers.csv", sep=",", index_col=False)
        vehicules = pd.read_csv("ech_vehicules.csv", sep=",", index_col=False)
        df = pd.read_csv("data.csv", sep="\t")
        return caracteristiques, lieux, usagers, vehicules, df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return None, None, None, None, None

# Chargement des données
caracteristiques, lieux, usagers, vehicules, df = load_data()
    
st.title("Accidentologie")

st.sidebar.title("Sommaire")
pages=["Exploration", "Visualisation", "Preprocessing", "Modélisation", "Prédiction (démo)"]
page=st.sidebar.radio("Aller vers", pages)



##############################################################################
#
# Page : Exploration
#
##############################################################################

if page == pages[0]:
    st.header("Exploration")
    
    st.write("72 dataframes chargés* :")
    st.write("- 4 rubriques concernées : 'caracteristiques', 'lieux', 'usagers', 'vehicules'")
    st.write("- 18 datasets par rubrique : une pour chaque année de 2005 à 2022")
    st.write("(*) source : [data.gouv.fr](https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2022/)")
    st.write("")

    # Liste des DataFrames et leurs noms
    dataframes = {'caracteristiques': caracteristiques, 'lieux': lieux, 'usagers': usagers, 'vehicules': vehicules}

    # Sélection du DataFrame à afficher
    nom_dataframe = st.selectbox("Sélectionnez une rubrique de DataFrame pour afficher un extrait :", list(dataframes.keys()))
    if nom_dataframe == "caracteristiques":
        st.write("Cette rubrique fournit les informations générales sur les circonstances de l’accident.")
    if nom_dataframe == "lieux":
        st.write("Cette rubrique fournit les détails sur l’emplacement de l’accident.")
    if nom_dataframe == "usagers":
        st.write("Cette rubrique fournit des données sur les personnes impliquées dans l’accident.")
    if nom_dataframe == "vehicules":
        st.write("Cette rubrique fournit des informations sur les véhicules impliqués.")

    # Fonction pour afficher le DataFrame sélectionné
    def afficher_dataframe(dataframe, nom):
        st.dataframe(dataframe.head())

        if st.checkbox(f"Afficher le résumé des données brutes avant concaténation") :
            if nom == "caracteristiques":
                stats1_caracteristiques = pd.read_csv("stats1_caracteristiques.csv", sep = '\t', index_col=0)
                st.dataframe(stats1_caracteristiques.T)
            if nom == "lieux":
                stats1_lieux = pd.read_csv("stats1_lieux.csv", sep = '\t', index_col=0)
                st.dataframe(stats1_lieux.T)

            if nom == "usagers":
                stats1_usagers = pd.read_csv("stats1_usagers.csv", sep = '\t', index_col=0)
                st.dataframe(stats1_usagers.T)

            if nom == "vehicules":
                stats1_vehicules = pd.read_csv("stats1_vehicules.csv", sep = '\t', index_col=0)
                st.dataframe(stats1_vehicules.T)

        if st.checkbox(f"Afficher le résumé des données brutes après concaténation") :
            st.session_state.caracteristiques_shape = (1176873, 17)
            if nom == "caracteristiques":
                st.write(f"Shape :", {st.session_state.caracteristiques_shape})
                stats2_caracteristiques = pd.read_csv("stats2_caracteristiques.csv", sep = '\t', index_col=0)
                st.dataframe(stats2_caracteristiques.T)
            st.session_state.lieux_shape = (1176873, 19)
            if nom == "lieux":
                st.write(f"Shape :", {st.session_state.lieux_shape})
                stats2_lieux = pd.read_csv("stats2_lieux.csv", sep = '\t', index_col=0)
                st.dataframe(stats2_lieux.T)
            st.session_state.usagers_shape = (2636377, 17)
            if nom == "usagers":
                st.write(f"Shape :", {st.session_state.usagers_shape})
                stats2_usagers = pd.read_csv("stats2_usagers.csv", sep = '\t', index_col=0)
                st.dataframe(stats2_usagers.T)
            st.session_state.vehicules_shape = (2009395, 11)
            if nom == "vehicules":
                st.write(f"Shape :", {st.session_state.vehicules_shape})
                stats2_vehicules = pd.read_csv("stats2_vehicules.csv", sep = '\t', index_col=0)
                st.dataframe(stats2_vehicules.T)

    # Appeler la fonction d'affichage d'un DataFrame sélectionné
    afficher_dataframe(dataframes[nom_dataframe], nom_dataframe)



##############################################################################
#
# Page : Visualisation
#
##############################################################################

elif page == pages[1]:
    st.header("Visualisation")
   
    merged_df = pd.merge(caracteristiques, lieux, on='Num_Acc', how='inner')
    merged_df = pd.merge(merged_df, usagers, on='Num_Acc', how='inner')
    merged_df = pd.merge(merged_df, vehicules, on=['Num_Acc', 'id_vehicule', 'num_veh'], how='inner')

    # Définition des variables à traiter
    variables = ['grav', 'agg', 'an', 'catu', 'lum', 'mois', 'sexe']

    # Dictionnaire de labels pour les variables
    variable_labels = {
        'grav': 'Gravité',
        'agg': 'Agglomération',
        'an': 'Année',
        'catu': "Catégorie d'usager",
        'lum': "Lumière",
        "mois": "Mois",
        "sexe": "Sexe"
    }

    # Dictionnaires pour les labels
    mois_int_order = range(1, 13)
    mois_labels = {str(i): label for i, label in zip(mois_int_order, ["Janv.", "Févr.", "Mars", "Avr.", "Mai", "Juin", 
                                                                      "Juil.", "Août", "Sept.", "Oct.", "Nov.", "Déc."])}
    mois_order = ["Janv.", "Févr.", "Mars", "Avr.", "Mai", "Juin", "Juil.", "Août", "Sept.", "Oct.", "Nov.", "Déc."]
    sexe_labels = {'1': 'Homme', '2': 'Femme'}
    agg_int_order = range(1, 3)
    agg_labels = {str(i): label for i, label in zip(agg_int_order, ['Hors agglomération', 'En agglomération'])}
    agg_order = ['Hors agglomération', 'En agglomération']
    catu_labels = {'1': 'Conducteur', '2': 'Passager', '3': 'Piéton'}
    lum_labels = {'1': 'Plein jour', '2': 'Crépuscule - aube', 
                  '3': 'Nuit sans éclairage', 
                  '4': 'Nuit avec éclairage public non allumé', 
                  '5': 'Nuit avec éclairage public allumé'}

    # Conversion des colonnes en chaînes et exclusion des valeurs -1
    merged_df[variables] = merged_df[variables].astype(str)
    for var in variables:
        merged_df = merged_df[merged_df[var] != "-1"]

    # Renommer les valeurs de gravité
    gravite_labels = {"1": "Indemne", "4": "Blessé léger", "3": "Blessé hospitalisé", "2": "Tué"}
    merged_df["grav"] = merged_df["grav"].map(gravite_labels)

    # Remplacer les valeurs numériques par leurs noms correspondants
    merged_df["mois"] = merged_df["mois"].astype(int)  # Convertir en entiers
    merged_df = merged_df.sort_values("mois")  # Trier par mois
    merged_df["mois"] = merged_df["mois"].astype(str).map(mois_labels)  # Convertir en labels
    merged_df["sexe"] = merged_df["sexe"].map(sexe_labels)
    merged_df["agg"] = merged_df["agg"].astype(int)  # Convertir en entiers
    merged_df = merged_df.sort_values("agg")  # Trier par agglomération
    merged_df["agg"] = merged_df["agg"].astype(str).map(agg_labels)  # Convertir en labels
    merged_df["catu"] = merged_df["catu"].map(catu_labels)
    merged_df["lum"] = merged_df["lum"].map(lum_labels)

    # Interface Streamlit
    st.write("Sélectionnez une variable à visualiser en fonction de la gravité :")

    # Sélection de la variable pour le countplot (exclure ‘grav’ des options)
    variable = st.selectbox("Choisissez une variable :", [var for var in variables if var != "grav"])

    # Obtenir les valeurs uniques de grav dans l'ordre souhaité
    unique_grav = ["Indemne", "Blessé léger", "Blessé hospitalisé", "Tué"]

    # Visualisation avec tri numérique pour la variable sélectionnée
    fig = plt.figure(figsize=(10, 6))

    # Déterminer l'ordre basé sur la variable sélectionnée
    if variable == "mois":
       order = mois_order  # Utiliser l'ordre défini pour les mois
    if variable == "agg":
        order = agg_order  # Utiliser l'ordre défini pour l'agglomération'
    else:
        variable_values = merged_df[variable].unique()
        order = sorted(variable_values)  # Tri par défaut pour d'autres variables
    
    sns.countplot(x=variable, hue="grav", data=merged_df, order=order, hue_order=unique_grav)

    plt.title(f"Distribution de la gravité en fonction de {variable_labels[variable]}")
    
    # Incliner les labels de l'axe x si la variable est lumière
    if variable == 'lum':
        plt.xticks(rotation=90)  # Incliner les labels à 90 degrés

    plt.xlabel(variable_labels[variable])  # Utiliser le label ici
    plt.ylabel("Nombre d'accidents")
    plt.legend(title="Gravité")

    st.pyplot(fig)

    # Histogramme basé sur la même variable sélectionnée
    st.subheader(f"Histogramme des accidents par {variable_labels[variable]}")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    if variable == "an":
        sns.histplot(data=merged_df, x="an", hue="grav", multiple="stack", bins=30)
        plt.title("Histogramme des accidents par année")
        plt.xlabel("Année")
    elif variable == "catu":
        sns.histplot(data=merged_df, x="catu", hue="grav", multiple="stack", discrete=True)
        plt.title("Histogramme des accidents par catégorie d'usager")
        plt.xlabel("Catégorie d'usager")
    elif variable == "lum":
        sns.histplot(data=merged_df, x="lum", hue="grav", multiple="stack", discrete=True)
        plt.title("Histogramme des accidents par luminosité")
        plt.xlabel("Luminosité")
    elif variable == "sexe":
        sns.histplot(data=merged_df, x="sexe", hue="grav", multiple="stack", discrete=True)
        plt.title("Histogramme des accidents par sexe")
        plt.xlabel("Sexe")
    elif variable == "agg":
        sns.histplot(data=merged_df, x="agg", hue="grav", multiple="stack", discrete=True)
        plt.title("Histogramme des accidents par agglomération")
        plt.xlabel("Agglomération")

    plt.ylabel("Nombre d'accidents")
    plt.legend(title="Gravité")
    st.pyplot(fig2)



##############################################################################
#
# Page : Preprocessing
#
##############################################################################
@st.cache_data
def process_uploaded_file(_file):
    if _file is not None:
        try:
            return pd.read_csv(_file, index_col=0, sep='\t')
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {str(e)}")
    return None

if page == pages[2]:
    st.header("Étapes de preprocessing des données")

    # Définition des fonctions de chargement et preprocessing
    def load_and_merge_dataframes(uploaded_files):
        dataframes = {}
        expected_files = ['caracteristiques', 'lieux', 'usagers', 'vehicules']
        
        # Chargement
        for file, expected in zip(uploaded_files.values(), expected_files):
            if file is not None:
                df = pd.read_csv(file, sep=',', encoding='utf-8', index_col=False)
                if 'Unnamed: 0' in df.columns:
                    df = df.drop('Unnamed: 0', axis=1)
                dataframes[expected] = df
        
        # Fusion
        df = pd.merge(dataframes['usagers'], dataframes['caracteristiques'], on='Num_Acc', how='left')        
        df = pd.merge(df, dataframes['lieux'], on='Num_Acc', how='left')
        df = pd.merge(df, dataframes['vehicules'], on=['Num_Acc', 'num_veh', 'id_vehicule'], how='left')
        
        return df

    # Configuration du chargement des fichiers (commun aux deux options de preprocessing)
    st.sidebar.header('Chargement des données')
    uploaded_files = {
        'caracteristiques': st.sidebar.file_uploader("Fichier 'caracteristiques'", type=['csv'], key="carac_upload"),
        'lieux': st.sidebar.file_uploader("Fichier 'lieux'", type=['csv'], key="lieux_upload"),
        'usagers': st.sidebar.file_uploader("Fichier 'usagers'", type=['csv'], key="usagers_upload"),
        'vehicules': st.sidebar.file_uploader("Fichier 'vehicules'", type=['csv'], key="vehicules_upload")}

    if not all(uploaded_files.values()):
        st.info('Veuillez charger tous les fichiers pour commencer le preprocessing')
    else:
        try:
            # Chargement initial des données
            df = load_and_merge_dataframes(uploaded_files)
            
            # Choix du type de preprocessing
            choix = ['preprocessing basique', 'preprocessing avancé']
            option = st.radio('Choix du preprocessing', choix)
            
            if option == 'preprocessing basique':
                # Prétraitement des données
                with st.expander(f"Prétraitement des données"):     
                    
                    # Renommer 'Accident_Id' en 'Num_Acc' si nécessaire
                    if 'Accident_Id' in df.columns:
                        df = df.rename(columns={'Accident_Id': 'Num_Acc'})
                        
                    # Ajouter 2000 à 'an' si < 2000
                    if 'an' in df.columns:
                        df.loc[:, 'an'] = df['an'].apply(lambda x: x + 2000 if x < 2000 else x)
                    
                    # Convertir 'hrmn' de 'HHMM' à 'HH:MM'
                    if 'hrmn' in df.columns:
                        df.loc[:, 'hrmn'] = df['hrmn'].apply(lambda x: f"{str(x).zfill(4)[:2]}:{str(x).zfill(4)[2:]}")
                    
                    # Remplacer les valeurs NaN par -1
                    for col in ['lum', 'int', 'atm', 'col']:
                        if col in df.columns:
                            df[col] = df[col].fillna(-1)
                    
                    # Supprimer les colonnes non nécessaires
                    df = df.drop(columns=['adr', 'lat', 'long'], errors='ignore')

                    # Remplacer les valeurs NaN par -1
                    for col in ['circ', 'vosp', 'prof', 'pr', 'pr1', 'plan', 'surf', 'infra', 'situ']:
                        if col in df.columns:
                            df[col] = df[col].fillna(-1)
                    
                    # Remplacer les valeurs NaN et 0 par -1 pour 'lartpc'
                    if 'lartpc' in df.columns:
                        df['lartpc'] = df['lartpc'].replace(0, -1).fillna(-1)
                    
                    # Remplacer les valeurs NaN et > 130 par -1 pour 'vma'
                    if 'vma' in df.columns:
                        df.loc[df['vma'] > 130, 'vma'] = -1
                        df['vma'] = df['vma'].fillna(-1)
                        
                    # Supprimer les colonnes 'voie', 'v1', 'v2', 'larrout'
                    df = df.drop(columns=['voie', 'v1', 'v2', 'larrout'], errors='ignore')

                    # Remplacer les valeurs NaN par -1
                    for col in ['place', 'catu', 'grav', 'sexe', 'trajet', 'secu1', 'secu2', 'secu3', 'locp', 'actp', 'etatp']:
                        if col in df.columns:
                            df[col] = df[col].fillna(-1)
                    
                    # Remplacer les valeurs 4 par -1 pour 'catu'
                    if 'catu' in df.columns:
                        df['catu'] = df['catu'].replace(4, -1)
                    
                    # Gérer les outliers pour 'an_nais'
                    if 'an_nais' in df.columns:
                        df.loc[:, 'an_nais'] = df['an_nais'].apply(lambda x: pd.NA if x < 1900 else x)

                    for col in ['senc', 'obs', 'obsm', 'choc', 'manv', 'motor']:
                        if col in df.columns:
                            df[col] = df[col].fillna(-1)
                    
                    # Remplacer les valeurs NaN par 0 pour 'catv'
                    if 'catv' in df.columns:
                        df['catv'] = df['catv'].fillna(0)
                    
                    # Supprimer la colonne 'occutc'
                    df = df.drop(columns=['occutc'], errors='ignore')
                
                    for col in df.columns:
                        if df[col].dtype == np.float64 or df[col].dtype == np.int64:
                            df.loc[:, col] = df[col].fillna(-1)

                    # Gestion des cas particuliers
                    if 'lartpc' in df.columns:
                        df.loc[:, 'lartpc'] = df['lartpc'].replace(0, -1).fillna(-1)
                    if 'catu' in df.columns:
                        df.loc[:, 'catu'] = df['catu'].replace(4, -1)
                    if 'an_nais' in df.columns:
                        df.loc[:, 'an_nais'] = df['an_nais'].apply(lambda x: pd.NA if x < 1900 else x)

                    
                    st.write("- Renommage de la variable 'Accident_Id' en 'Num_Acc'")
                    st.write("- Conversion de 'HHMM' à 'HH:MM'")
                    st.write("- Remplacement des valeurs NaN")
                    st.write("- Suppression de colonnes non nécessaires")
                    st.write("- Gestion des outliers pour 'an_nais'")
                    st.write("- ...")
                    st.write("Extrait du DataFrame :")   
                    st.write(df.head())
                    st.write(df.shape)

                # Vérification des NaN dans le DataFrame final
                def check_nan_presence(df):
                    with st.expander("Vérification des valeurs manquantes"):  

                        nan_columns = df.columns[df.isna().any()].tolist()
                        if nan_columns:
                            nan_proportions = df[nan_columns].isna().mean() * 100
                            for col, prop in nan_proportions.items():
                                st.write(f"Colonne '{col}' contient {prop:.2f}% de NaN.")
                        else:
                            st.write("Aucune colonne ne contient de NaN.")
                
                check_nan_presence(df)

                # Vérification du nombre de valeurs uniques dans chaque colonne après fusion
                def check_nunique(df):
                    with st.expander("Vérification du nombre de valeurs uniques"):

                        unique_proportions = df.nunique()
                        for col, prop in unique_proportions.items():
                            st.write(f"Colonne '{col}' a {prop} valeurs uniques.")

                check_nunique(df)

                # Prétraitement final après fusion
                def preprocessing_final_dataframe(df):
                    with st.expander("Vérification des types de données finaux"):

                        # Supprimer les colonnes
                        df = df.drop(columns=['id_usager', 'Num_Acc', 'com', 'id_vehicule', 'num_veh','lartpc'])
                        
                        # Modifier la colonne 'hrmn' en 'hour' et la passer du format HH:MM au format HH
                        if 'hrmn' in df.columns:
                            df['hour'] = df['hrmn'].str[:2].astype(int)
                            df = df.drop(columns=['hrmn'])
                        
                        # Remplacer les valeurs manquantes de 'an_nais' par le mode de la colonne
                        if 'an_nais' in df.columns:
                            mode_an_nais = df['an_nais'].mode()[0]
                            df['an_nais'] = df['an_nais'].fillna(mode_an_nais).astype(int)

                        st.write("- Supression de colonnes : 'id_usager', 'Num_Acc', 'com', 'id_vehicule', 'num_veh','lartpc'")
                        st.write("- Modification du format de la colonne 'hrmn'")
                        st.write("- Remplacement des valeurs manquantes de 'an_nais' par le mode de la colonne")
                        st.write(df.shape)

                    return df
                
                # Encodage du dataframe
                def encode_dataframe(df):
                    dummy_columns = ['lum', 'agg', 'int', 'atm', 'col', 'catr', 'circ', 'prof', 'place', 'catu', 'sexe', 
                                    'trajet', 'secu1', 'secu2', 'secu3', 'locp', 'actp', 'etatp', 'senc', 'catv', 'obs', 
                                    'obsm', 'choc', 'manv', 'motor', 'plan', 'surf','an','infra','dep','situ','vosp']
                    df = pd.get_dummies(df, columns=dummy_columns, drop_first= True)

                    return df
                              
                # Lancement du processus final
                df = preprocessing_final_dataframe(df)
                
                final_merged_df = encode_dataframe(df)

                with st.expander("Dataframe prétraité"):
                    st.write(f"Nombre de valeurs nulles : {final_merged_df.isnull().sum().sum()}")
                    st.write("Extrait du DataFrame :")
                    st.write(final_merged_df.head())
                    st.write(final_merged_df.shape)

                
                with st.expander("Colonnes 'objet' restantes"):
                    # Repérage des colonnes 'objet' restantes
                    object_columns = final_merged_df.select_dtypes(include=['object']).columns
                    st.write("Nombre de colonnes objet restantes:", len(object_columns))
                    st.write("Liste de colonnes objet restantes:", list(object_columns))

                    # Boucle de modification des colonnes 'objet' restantes
                    st.write("Transformation des colonnes 'objet' en 'int'")
                    for column in object_columns:
                        try:
                            final_merged_df[column] = final_merged_df[column].astype('int64')
                        except ValueError:
                            try:
                                final_merged_df = final_merged_df[pd.to_numeric(final_merged_df[column], errors='coerce').notnull()]
                                final_merged_df[column] = final_merged_df[column].astype('int64')
                            except ValueError:
                                st.warning(f"Impossible de convertir la colonne '{column}' en entier.")
                
                    # Affichage de la nouvelle dimension du dataframe
                    st.write("Nouveau shape du dataframe:", final_merged_df.shape)

                # Sauvegarde du DataFrame prétraité
                st.download_button("💾 Télécharger le DataFrame final", final_merged_df.to_csv(index=False).encode('utf-8'), "basic_processing.csv", "text/csv", key='download-csv')            


            elif option == 'preprocessing avancé':
                def preprocess_data(df):
                    st.header('1. Fusion des DataFrames')
                    
                    with st.expander("1.1 Fusion des DataFrames"):
                        # La fusion des DataFrames se fait dans la fonction load_and_merge_dataframes
                        st.write("Extrait du DataFrame fusionné :")
                        st.write(df.head())
                        st.write(df.shape)

                    st.header('2. Prétraitement des données')
                
                    with st.expander("2.1 Suppression d'observations avec gravité inconnue"):
                        # Suppression des observations avec gravité inconnue
                        n_initial = len(df)
                        df = df[df['grav'] != -1]
                        n_final = len(df)
                        st.write(f"{n_initial - n_final} lignes supprimées")
                        st.write(df.shape)

                    with st.expander("2.2 Création de variables"):
                        c_initial = df.shape[1]

                        # Création de la variable 'jsem' (jour de la semaine)
                        df_date = df[["an", "mois", "jour"]]
                        df_date = df_date.rename({"an": "year", "mois": "month", "jour": "day"}, axis=1)
                        df_date["ts"] = pd.to_datetime(df_date)
                        df_date["jsem"] = df_date.ts.apply(lambda x: x.weekday()+1)
                        df["jsem"] = df_date.jsem
                        df_date = None
                                    
                        # Création de la variable 'ferie'
                        df['ferie'] = False
                        jours_feries = [((1, 1), "Jour de l'an"), ((1, 5), "Fête du travail"), ((8, 5), "Victoire 1945"),
                                    ((14, 7), "Fête nationale"), ((15, 8), "Assomption"), ((1, 11), "Toussaint"),
                                    ((11, 11), "Armistice"), ((25, 12), "Noël")]
                        for (mois, jour), nom in jours_feries:
                            df.loc[(df['mois'] == mois) & (df['jour'] == jour), 'ferie'] = True
                        
                        # Calcul de la variable 'age'
                        df['age'] = df['an'].astype(float) - df['an_nais'].astype(float)
                        df.loc[df['age'] < 0, 'age'] = -1

                        c_final = df.shape[1]

                        st.write("- 'jsem' (jour de la semaine)")
                        st.write("- 'ferie' (jour férié)")
                        st.write("- 'age' (âge)")

                        st.write(f"{c_final - c_initial} colonnes créées")
                        st.write(df.shape)

                    # Nettoyage préalable des variables catégorielles
                    variables_a_nettoyer = ['secu1', 'secu2', 'secu3', 'sexe', 'nbv', 'surf',
                                            'agg', 'grav', 'catv', 'choc', 'circ', 'col', 'etatp',
                                            'infra', 'int', 'locp', 'lum', 'manv', 'motor', 
                                            'obs', 'obsm', 'place', 'plan', 'prof', 'senc', 
                                            'situ', 'trajet', 'vosp']

                    for var in variables_a_nettoyer:
                        if var in df.columns:  # Vérifier si la variable existe
                            df[var] = (df[var].astype(str).str.strip().str.replace(r'\s+', ''))
                            
                    with st.expander("2.3 Dichotomisation de variables sans fonction"):
                        c_initial = df.shape[1]

                        # Dichotomisation des équipements de sécurité
                        for equip in ['ceinture', 'casque', 'dispenfant', 'gilet', 'airbag23RM', 'gants']:
                            df[f'secu_{equip}'] = False
                        for i in range(1, 4):
                            col = f'secu{i}'
                            if col in df.columns:
                                df.loc[df[col] == 1, 'secu_ceinture'] = True
                                df.loc[df[col] == 2, 'secu_casque'] = True
                                df.loc[df[col] == 3, 'secu_dispenfant'] = True
                                df.loc[df[col] == 4, 'secu_gilet'] = True
                                df.loc[df[col].isin([5, 7]), 'secu_airbag23RM'] = True
                                df.loc[df[col].isin([6, 7]), 'secu_gants'] = True
                        
                        # Catégorisation de l'âge
                        df['age_enfant'] = (df['age'] >= 0) & (df['age'] <= 15)
                        df['age_jeune'] = (df['age'] > 15) & (df['age'] <= 25)
                        df['age_adulte'] = (df['age'] > 25) & (df['age'] <= 64)
                        df['age_3age'] = (df['age'] > 64)
                        
                        # Dichotomisation de la période de la journée
                        df['hrmn'] = df['hrmn'].astype(str).str.zfill(4)
                        df['hr_matin'] = (df['hrmn'] >= "0600") & (df['hrmn'] < "1200")
                        df['hr_midi'] = (df['hrmn'] >= "1200") & (df['hrmn'] < "1400")
                        df['hr_am'] = (df['hrmn'] >= "1400") & (df['hrmn'] < "1800")
                        df['hr_soir'] = (df['hrmn'] >= "1800") & (df['hrmn'] < "2100")
                        df['hr_nuit'] = (df['hrmn'] >= "2100") | (df['hrmn'] < "0600")

                        # Dichotomisation du sexe
                        df["sexe_m"] = df.sexe == '1'
                        df["sexe_f"] = df.sexe == '2'

                        # Dichotomisation de la gravité
                        df["grav_grave"]       = df.grav.isin(["2", "3"])

                        # Dichotomisation du nombre de voies de circulation avec regroupement
                        df["nbv_1"]    = df.nbv == '1'
                        df["nbv_2"]    = df.nbv == '2'
                        df["nbv_3"]    = df.nbv == '3'
                        df["nbv_4"]    = df.nbv == '4'
                        df["nbv_plus"] = df.nbv.isin(['5', '6', '7', '8', '9', '10','11', '12'])

                        # Dichotomisation de l'état de la surface
                        df["surf_norm"]  = df.surf == '1'
                        df["surf_mouil"] = df.surf == '2'
                        df["surf_gliss"] = df.surf.isin(['3', '4', '5', '6', '7', '8', '9'])
                        df["surf_autre"] = df.surf == '9'

                        # Dichotomisation de la vitesse maximale autorisée
                        vma_int = df.vma.astype(int)
                        df["vma_30m"] = vma_int.isin([10, 20, 30])
                        df["vma_40"]  = vma_int == 40
                        df["vma_50"]  = vma_int == 50
                        df["vma_60"]  = vma_int == 60
                        df["vma_70"]  = vma_int == 70
                        df["vma_80"]  = vma_int == 80
                        df["vma_90"]  = vma_int == 90
                        df["vma_110"] = vma_int == 110
                        df["vma_130"] = vma_int == 130

                        # agg : En ou hors agglomération
                        df['agg_agg'] = df.agg == '1'
                    
                        c_final = df.shape[1]

                        st.write("- Equipements de sécurité ('ceinture', 'casque', 'dispenfant', 'gilet', 'airbag23RM', 'gants'')")
                        st.write("- Âge ('age_enfant', 'age_jeune', 'age_adulte', 'age_3age')")
                        st.write("- Créneau horaire ('hr_matin', 'hr_midi', 'hr_am', 'hr_soir', 'hr_nuit')")
                        st.write("- Sexe ('sexe_m', 'sexe_f')")
                        st.write("- Gravité ('grav_grave')")
                        st.write("- Nombre de voies de circulation ('nbv_1', 'nbv_2', 'nbv_3', 'nbv_4', 'nbv_plus')")
                        st.write("- Etat de la surface ('surf_norm', 'surf_mouil', 'surf_gliss', 'surf_autre')")
                        st.write("- En ou hors agglomération ('agg')")    

                        st.write(f"{c_final - c_initial} colonnes créées")
                        st.write(df.shape)

                
                    with st.expander('2.4 Dichotomisation de variables avec fonction'):
                        c_initial = df.shape[1]

                        # Initialisation de desc_vars
                        desc_vars = {"columns": []}

                        # Initialisation de la liste des variables écartées
                        var_ecartees = []
                
                        # Fonction de dichotomisation
                        def dichotomisation (df, column, var_ecartees, desc_vars=None, dummies=None, mod_ecartees = None):
                            """
                            Cette fonction fait la dichotomisation des seules modalités de la
                            liste fournie par dummies. Elle permet de ne pas dichotomiser les
                            modalités : "non renseigné", "Autre", "Non applicable", ...
                            Elle utilise si possible les infos de desc_vars pour nommer les colonnes.
                            et elle complète desc_vars.
                            """
                            try :
                                desc_vars = desc_vars.get("columns")
                                col_desc = {}
                                for c in desc_vars :
                                    if c.get("name") == column:
                                        col_desc = c
                                        break
                            except :
                                col_desc = {}

                            if dummies is None:
                                dum = list(df[column].unique())
                            else:
                                dum = dummies
                            if mod_ecartees is not None:
                                dum = [x for x in dum if x not in mod_ecartees]
                            print()
                            
                            # Pour chaque variable correspondant à une modalité
                            for c in dum: # c : modalité
                                
                                # Le nom de la nvle variable est le nom de la var. et la modalité
                                new_col_name = column + "_" + str(c)
                                
                                # Création de la nouvelle variable (colonne)
                                df[new_col_name] = df[column] == c
                                
                                # Recherche d'une description existante dans la liste des descriptions
                                # les descriptions sont dans un tableau qu'il faut adresser avec un entier
                                desc_new_col = None
                                for ic in range(len(desc_vars)):
                                    if desc_vars[ic].get("name") == new_col_name:
                                        desc_new_col = desc_vars[ic]
                                        break
                                    
                                if desc_new_col is None:
                                    desc_new_col = {}
                                    desc_new_col["name"] = new_col_name
                                    desc_new_col["dtype"] = "bool"
                                    values = col_desc.get("values")
                                    if values is not None and values.get(c) is not None:
                                        desc_new_col["label"] = col_desc.get("label") + " : " + values.get(c)
                                    else :
                                        desc_new_col["label"] = col_desc.get("label")
                                    desc_vars.append(desc_new_col)

                            var_ecartees.append ((column, "Dichotomisation"))           
                            return
                        
                        # actp : action du piéton
                        dummies = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B']
                        dichotomisation(df, "actp", var_ecartees, desc_vars, dummies = dummies)

                        # atm : Conditions atmosphériques
                        dummies = ['1', '2', '3', '4', '5', '6', '7', '8']
                        dichotomisation(df, "atm", var_ecartees, desc_vars, dummies = dummies)

                        # catr : Catégorie de route
                        dummies = ['1', '2', '3', '4', '5', '6', '7']
                        dichotomisation(df, "catr", var_ecartees, desc_vars, dummies = dummies)

                        # catu : Catégorie d'usager
                        dummies = ['1', '2', '3']
                        dichotomisation(df, "catu", var_ecartees, desc_vars, dummies = dummies)

                        # catv : Catégorie de véhicule
                        dichotomisation(df, "catv", var_ecartees, desc_vars, dummies = None, mod_ecartees=[0, -1, '-1', ' -1'])

                        # choc : Point de choc initial
                        dichotomisation(df, "choc", var_ecartees, desc_vars, dummies = None, mod_ecartees=[0, '0', -1, '-1', ' -1'])

                        # circ : Circulation
                        dichotomisation(df, "circ", var_ecartees, desc_vars, dummies = None, mod_ecartees=[-1, '-1', ' -1'])
            
                        # col : Type de collision
                        dichotomisation(df, "col", var_ecartees, desc_vars, dummies = None, mod_ecartees=[-1, '-1', ' -1'])

                        # etatp: Piéton seul
                        dichotomisation(df, "etatp", var_ecartees, desc_vars, dummies = ['1', '2', '3'])

                        # infra : Aménagement - infrastructure
                        dichotomisation(df, "infra", var_ecartees, desc_vars, dummies = None, mod_ecartees=[0, '0', -1, '-1', ' -1'])

                        # int : type d'intersection
                        dichotomisation(df, "int", var_ecartees, desc_vars, dummies = None, mod_ecartees=[-1, '-1', ' -1'])

                        # jsem : Jour de la semaine
                        dichotomisation(df, "jsem", var_ecartees, desc_vars, dummies = None, mod_ecartees=None)

                        # locp : Localisation du piéton
                        dichotomisation(df, "locp", var_ecartees, desc_vars, dummies = None, mod_ecartees = [0, '0', -1, '-1', ' -1'])

                        # lum : Lumière modalité
                        dichotomisation(df, "lum", var_ecartees, desc_vars, dummies = None, mod_ecartees = [-1, '-1', ' -1'])

                        # manv : Manœuvre
                        dichotomisation(df, "manv", var_ecartees, desc_vars, dummies = None, mod_ecartees = [0, '0', -1, '-1', ' -1'])

                        # mois : Mois
                        dichotomisation(df, "mois", var_ecartees, desc_vars, dummies = None, mod_ecartees = None)

                        # motor : Motorisation
                        dichotomisation(df, "motor", var_ecartees, desc_vars, dummies = None, mod_ecartees = [0, '0', -1, '-1', ' -1'])

                        # obs : Obstacle fixe heurté
                        dichotomisation(df, "obs", var_ecartees, desc_vars, dummies = None, mod_ecartees = [0, '0', -1, '-1', ' -1'])

                        # obsm : Obstacle mobile heurté
                        dichotomisation(df, "obsm", var_ecartees, desc_vars, dummies = None, mod_ecartees = [0, '0', -1, '-1', ' -1'])

                        # place : Place de l'usager dans le véhicule
                        dichotomisation(df, "place", var_ecartees, desc_vars, dummies = None, mod_ecartees = [-1, '-1', ' -1'])

                        # plan : Tracé en plan
                        dichotomisation(df, "plan", var_ecartees, desc_vars, dummies = None, mod_ecartees = [-1, '-1', ' -1'])

                        # prof : Déclivité
                        dichotomisation(df, "prof", var_ecartees, desc_vars, dummies = None, mod_ecartees = [-1, '-1', ' -1'])

                        # senc : sens de circulation
                        dichotomisation(df, "senc", var_ecartees, desc_vars, dummies = None, mod_ecartees = [-1, '-1', ' -1'])

                        # situ : Situation de l'accident
                        dichotomisation(df, "situ", var_ecartees, desc_vars, dummies = None, mod_ecartees = [-1, '-1', ' -1'])

                        # trajet : Motif du trajet
                        dichotomisation(df, "trajet", var_ecartees, desc_vars, dummies = None, mod_ecartees = [-1, '-1', ' -1'])

                        # vosp : Présence d'une voie réservée
                        dichotomisation(df, "vosp", var_ecartees, desc_vars, dummies = None, mod_ecartees = [-1, '-1', ' -1'])

                        c_final = df.shape[1]

                        st.write("- Action du piéton : 'actp'")
                        st.write("- Conditions atmosphériques : 'atm'")
                        st.write("- Catégorie de route : 'catr'")
                        st.write("- Catégorie d'usager : 'catu'")
                        st.write("- Catégorie de véhicule : 'catv'")
                        st.write("- Point de choc initial : 'choc'")
                        st.write("- Circulation : 'circ'")
                        st.write("- Type de collision : 'col'")
                        st.write("- Piéton seul : 'etatp'")
                        st.write("- Aménagement - infrastructure : 'infra'")
                        st.write("- Type d'intersection : 'int'")
                        st.write("- Jour de la semaine : 'jsem'")
                        st.write("- Localisation du piéton : 'locp'")
                        st.write("- Lumière modalité : 'lum'")
                        st.write("- Manoeuvre : 'manv'")
                        st.write("- Mois : 'mois'")
                        st.write("- Motorisation : 'motor'")
                        st.write("- Obstable fixe heurté : 'obs'")
                        st.write("- Ostable mobile heurté : 'obsm'")
                        st.write("- Place de l'usager dans le véhicule : 'place'")
                        st.write("- Tracé en plan : 'plan'")
                        st.write("- Déclivité : 'prof'")
                        st.write("- Sens de la circulation : 'senc'")
                        st.write("- Situation de l'accident : 'situ'")
                        st.write("- Motif du trajet : 'trajet'")
                        st.write("- Présence d'une voie réservée : 'vosp'")
                                
                        st.write(f"{c_final - c_initial} colonnes créées")
                        st.write(df.shape)
                

                    st.header('3. Nettoyage final')
                    with st.expander("3.1 Suppression des variables"):
                        # Suppression des variables
                        var_to_drop = [
                            'adr', 'an', 'an_nais', 'atm', 'catr', 'com', 'dep', 'grav', 'id_usager', 'id_vehicule',
                            'jour', 'larrout', 'lartpc', 'lat', 'long', 'lum', 'Num_Acc', 'num_veh', 'occutc', 'pr', 
                            'pr1', 'secu1', 'secu2', 'secu3', 'surf', 'voie', 'v1', 'v2',
                            
                            'actp', 'age', 'agg', 'catu', 'catv', 'choc', 'circ', 'col', 'etatp', 'hrmn',
                            'infra', 'int', 'jsem', 'locp', 'manv', 'mois', 'motor', 'nbv', 'obs', 'obsm', 
                            'place', 'plan', 'place', 'plan', 'prof', 'senc', 'sexe', 'situ', 'trajet', 'vma', 
                            'vosp']
                        
                        st.session_state.df_shape = len(var_to_drop)
                        st.write(f"Nombre de variables supprimées :", st.session_state.df_shape)
                        df = df.drop(columns=var_to_drop)
                        st.write("Nombre de colonnes restantes :", df.shape[1])

                    with st.expander("3.2 Suppression des doublons"):
                        # Suppression des doublons
                        n_before = len(df)
                        df = df.drop_duplicates()
                        n_after = len(df)
                        st.write(f"Nombre de doublons supprimés : {n_before - n_after}")
                        st.write("Nombre de lignes restantes :", df.shape[0])

                    st.write(df.shape)
                    st.write(df.head())

                    st.header('4. Équilibrage et finalisation')
                    with st.expander("4.1 Répartition des modalités 'grav'"):
                        st.write("Répartition des modalités avant réduction :")
                        st.write(df.value_counts("grav_grave"))
                        st.write(f"total : {df.shape[0]:6d}")
                        X = df.drop("grav_grave", axis = 1)
                        y = df.grav_grave
                        rus = RandomUnderSampler(random_state = 8421)
                        X, y = rus.fit_resample(X, y)
                        df = pd.concat([X, y], axis = 1)
                        st.write("Répartition des modalités après réduction :")
                        st.write(df.value_counts("grav_grave"))
                        st.write(f"total : {df.shape[0]:6d}")

                    st.header('5. Résultat final')
                    
                    if st.checkbox("Afficher le DataFrame final"):
                        st.dataframe(df)
                    
                    st.download_button("💾 Télécharger le DataFrame final", df.to_csv(index=False).encode('utf-8'), "advanced_processing.csv", "text/csv", key='download-csv')
                
                preprocess_data(df)

        except Exception as e:
            st.error(f"Une erreur s'est produite : {str(e)}")
            st.exception(e)

##############################################################################
#
# Page : Modélisation
#
##############################################################################

if page == pages[3]:
    st.header("Modélisations")

    # Choix du type de preprocessing
    choix_model = ['Modélisation basique (régression logistique)', 'Multiples modèles de classification', 'Modélisation avancée (deep learning)']
    option_model = st.radio('Choix de la modélisation', choix_model)

    if option_model == 'Modélisation basique (régression logistique)':
        st.write("")
        st.session_state.df_shape = (374319, 387)
        st.write("1. Chargement du DataFrame prétraité:")
        with st.expander("Afficher les détails"):
            st.write(f"Dimension du dataframe 'df' :", st.session_state.df_shape)

        st.write("")
        st.write("2. Séparation des features et de la variable cible :")
        with st.expander("Afficher les détails"):
            st.code('''
                    X = df.drop(columns=['grav'])
                    y = df['grav']
                    ''')

        st.write("")
        st.write("3. Standardisation des features :")
        with st.expander("Afficher les détails"):
            st.code('''
                    scaler = StandardScaler()
                    X_scaled = X.copy()
                    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
                    ''')
            
        st.write("")
        st.write("4. Division en ensembles d'entraînement et de test :")
        with st.expander("Afficher les détails"):
            st.code('''
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=12)
                    ''')
            
        st.write("")
        st.write("5. Implémentation d'un modèle de régression logistique :")
        with st.expander("Afficher les détails"):
            st.code('''
                    # Définition du classificateur
                    LR = LogisticRegression(random_state=123, tol=1e-3, solver='saga')
                    ''')

            st.code('''
                    # Validation croisée
                    cv3 = KFold(n_splits=3, random_state=111, shuffle=True)
                    ''')

            st.code('''
                    # Définition de l'espace de recherche des hyperparamètres
                    param_grid = [
                        {'C': [0.1, 1, 10], 'penalty': ['l1'], 'max_iter': [1000]},
                        {'C': [0.1, 1, 10], 'penalty': ['l2'], 'max_iter': [1000]},
                        {'C': [0.1, 1, 10], 'penalty': ['elasticnet'], 'max_iter': [1000], 'l1_ratio': np.linspace(0, 1, 6)}
                    ]
                    ''')
            
        st.write("")
        st.write("6. Création d'un GridSearchCV personnalisé pour sauvegarder les résultats partiels :")
        with st.expander("Afficher les détails"):
            st.code('''
                    class GridSearchWithProgress(GridSearchCV):
                        def fit(self, X, y=None, **fit_params):
                            n_candidates = sum(len(ParameterGrid(grid)) for grid in self.param_grid)
                            n_completed = 0
                            
                            results = []
                            
                            for params in ParameterGrid(self.param_grid):
                                start_time = time.time()
                                self.estimator.set_params(**params)
                                cv_results = cross_validate(self.estimator, X, y, cv=self.cv, scoring=self.scoring, return_train_score=True)
                                end_time = time.time()
                                
                                result = {
                                    'params': params,
                                    'mean_test_score': cv_results['test_score'].mean()
                                }
                                
                                results.append(result)
                                
                                n_completed += 1
                                
                                # Enregistrer les résultats partiels
                                joblib.dump(results, 'gridcv_results_LR.joblib')
                                
                                # Affichage des résultats partiels
                                print(f"Résultat partiel {n_completed}/{n_candidates}:")
                                print(f"Paramètres: {params}")
                                print(f"Score moyen: {cv_results['test_score'].mean():.4f} (+/- {cv_results['test_score'].std():.4f})")
                                print(f"Temps de fit: {end_time - start_time:.2f} secondes")
                                print()
                                
                            self.cv_results_ = results
                            best_index = max(range(len(results)), key=lambda i: results[i]['mean_test_score'])
                            self.best_params_ = results[best_index]['params']
                            self.best_score_ = results[best_index]['mean_test_score']
                            
                            # Définition du meilleur estimateur
                            self.best_estimator_ = self.estimator.set_params(**self.best_params_)
                            self.best_estimator_.fit(X, y)
                            
                            return self
                    ''')
            
        st.write("")
        st.write("7. Exécution du GridSearchCV avec contrôle de l'état d'avancement :")
        with st.expander("Afficher les détails"):
            st.code('''
                grid_search = GridSearchWithProgress(LR, param_grid, cv=cv3, n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)
                ''')
        
        st.write("")
        if st.checkbox(f"Afficher les meilleurs paramètres et score") :
            st.write("Meilleurs paramètres: {'C': 1, 'l1_ratio': 0.4, 'max_iter': 1000, 'penalty': 'elasticnet'}")
            st.write("Meilleur score: 0.5895009263286214")
            st.write("Accuracy du meilleur modèle: 0.5923140628339388")

        st.write("")
        if st.checkbox(f"Afficher le rapport de classification") :
            lr_class_report = pd.read_csv('lr_class_report.csv', sep='\t', index_col=0)
            st.dataframe(lr_class_report)


    elif option_model == 'Multiples modèles de classification':
        st.write("")
        st.session_state.df_shape = (177642, 265)
        st.write("1. Chargement du DataFrame prétraité:")
        with st.expander("Afficher les détails"):
            st.write(f"Dimension du dataframe 'df' :", st.session_state.df_shape)

        st.write("")
        st.write("2. Séparation des features et de la variable cible :")
        with st.expander("Afficher les détails"):
            st.code('''
                    X = df.drop('grav_grave', axis=1)
                    y = df['grav_grave']
                    ''')

        st.write("")
        st.write("3. Réduction de dimension :")
        with st.expander("Afficher les détails"):
            st.code('''
                pca = PCA(n_components = 100)
                pca.fit(X)
                X_pca = pca.transform(X)
                    ''')

        st.write("")
        st.write("4. Standardisation des features :")
        with st.expander("Afficher les détails"):
            st.code('''
                    scaler = StandardScaler()
                    X_pca_scale = scaler.fit_transform(X_pca)
                    ''')

        st.write("")
        st.write("3. Division en ensembles d'entraînement et de test :")
        with st.expander("Afficher les détails"):
            st.code('''
                    X_train, X_test, y_train, y_test = train_test_split(X_pca_scale, y, test_size = .20, random_state = 1234)
                    ''')            
        
        st.write("")
        st.write("4. Choix des modèles de classification :")
        with st.expander("Afficher les détails"):
            st.code('''
                    modeles = {

                    "SVC" : {
                        "prmsgscv" : {},
                        "prmfixes" : {},
                        "pred"     : None,
                        "perf"     : [],
                        "nom"      : "SVC",
                        "libelle"  : "Classification à support de vecteurs",
                        "classe"   : "sklearn.svm.SVC",
                        "instance" : SVC(),
                        "grid"     : None
                    },
                        
                    "LR" : {
                        "prmsgscv" : {'solver' : ['liblinear', 'lbfgs'],
                                    'C' : [0.003, 0.005, 0.01, 0.02, 0.04]},
                        "prmfixes" : {},
                        "perf"     : {},
                        "nom"      : "LogisticRegression",
                        "libelle"  : "Régression logistique",
                        "classe"   : "sklearn.xxx",
                        "instance" : LogisticRegression(max_iter = 10000),
                        "grid"     : None
                    },

                    "HGB" : {
                        "prmsgscv" : {"max_leaf_nodes" : [None, 15, 31, 63 ],
                                    "l2_regularization" : [0.001, 0.01, 0.1, 1, 10],},
                        "prmfixes" : {},
                        "perf"     : {},
                        "nom"      : "HistGradientBoostingClassifier",
                        "libelle"  : "Hist Gradient Boosting Classifier",
                        "classe"   : "sklearn.neighbors.HistGradientBoostingClassifier",
                        "instance" : HistGradientBoostingClassifier(random_state = 421),
                        "grid"     : None
                    },

                    "LGBM" : {
                        "prmsgscv" : {"boosting_type" : ["gbdt", "dart", "goss"],},
                        "prmfixes" : {},
                        "perf"     : {},
                        "nom"      : "LGBMClassifier",
                        "libelle"  : "LGBMClassifier",
                        "classe"   : "sklearn.",
                        "instance" : LGBMClassifier(verbosity=-1, random_state = 421),
                        "grid"     : None
                    },

                    "DT" : {
                        "prmsgscv" : {"criterion" : ["gini", "entropy", "log_loss"],
                                    "splitter" : ["best", "random"]},
                        "prmfixes" : {},
                        "perf"     : {},
                        "nom"      : "DecisionTreeClassifier",
                        "libelle"  : "Arbre de décision",
                        "classe"   : "sklearn.tree.DecisionTreeClassifier",
                        "instance" : DecisionTreeClassifier(random_state = 123), 
                        "grid"     : None
                    },

                    "ET" : {
                        "prmsgscv" : {},
                        "prmfixes" : {},
                        "perf"     : {},
                        "nom"      : "ExtraTreesClassifier",
                        "libelle"  : "Extra Trees Classifier",
                        "classe"   : "sklearn.ensemble.ExtraTreesClassifier",
                        "instance" : ExtraTreesClassifier(),
                        "grid"     : None
                    },

                    "BAG" : {
                        "prmsgscv" : {},
                        "prmfixes" : {},
                        "perf"     : {},
                        "nom"      : "BaggingClassifier",
                        "libelle"  : "Bagging Classifier",
                        "classe"   : "sklearn.neighbors.BaggingClassifier",
                        "instance" : BaggingClassifier(),
                        "grid"     : None
                    },

                    "KNN" : {
                        "prmsgscv" : {"metric" : ['minkowski', 'manhattan', 'chebyshev']},
                        "prmfixes" : {},
                        "perf"     : {},
                        "nom"      : "KNeighborsClassifier",
                        "libelle"  : "Plus proches voisins",
                        "classe"   : "sklearn.neighbors.KNeighborsClassifier",
                        "instance" : KNeighborsClassifier(),
                        "grid"     : None
                    },

                    "RF" : {
                        "prmsgscv" : {},
                        "prmfixes" : {},
                        "perf"     : {},
                        "nom"      : "RandomForestClassifier",
                        "libelle"  : "Forêt aléatoire",
                        "classe"   : "sklearn.neighbors.RandomForestClassifier",
                        "instance" : RandomForestClassifier(random_state =421), 
                        "grid"     : None
                    },

                    "GB" : {
                        "prmsgscv" : {"loss" : ["log_loss", "exponential"],
                                    "criterion" : ["friedman_mse", "squared_error"],},
                        "prmfixes" : {},
                        "perf"     : {},
                        "nom"      : "GradientBoostingClassifier",
                        "libelle"  : "Gradient Boosting Classifier",
                        "classe"   : "sklearn.neighbors.GradientBoostingClassifier",
                        "instance" : GradientBoostingClassifier(random_state = 421),
                        "grid"     : None
                    },

                    # Classifieur bidon à comparer aux autres modèles
                    "BIDON" : {
                        "prmsgscv" : {},
                        "prmfixes" : {},
                        "perf"     : {},
                        "nom"      : "DummyClassifier",
                        "libelle"  : "Dummy Classifier",
                        "classe"   : "sklearn.dummy.DummyClassifier",
                        "instance" : DummyClassifier(random_state = 421, strategy = "uniform"),
                        "grid"     : None
                    }

                    }
                            
                    ''')

        st.write("")
        st.write("5. Tableau des performances :")
        with st.expander("Afficher les détails"):    
            tab_perf_class = pd.read_csv("tab_perf_class.csv", sep = '\t', index_col=0)
            st.dataframe(tab_perf_class)

        st.write("")
        st.write("6. Visualisation graphique :")
        with st.expander("Afficher les détails"):     
            st.write("")
            image1 = Image.open("comparaison_scores.png")
            st.image(image1, caption="Comparaison des scores", width=680)
            st.write("=" * 84)
            st.write("")
            image2 = Image.open("evaluation_performances.png")
            st.image(image2, caption="Evaluation des performances", width=680)

        st.write("")
        st.write("7. Analyse SHAP :")
        with st.expander("Afficher les détails"):     
            st.write("")
            image3 = Image.open("shap_summary_LGBM.png")
            st.image(image3, caption="Labels explicatifs", width=680)


    elif option_model == 'Modélisation avancée (deep learning)':
        st.write("")
        st.session_state.df_shape = (177642, 265)
        st.write("1. Chargement du DataFrame prétraité:")
        with st.expander("Afficher les détails"):
            st.write(f"Dimension du dataframe 'df' :", st.session_state.df_shape)

        st.write("")
        st.write("2. Séparation des features et de la variable cible :")
        with st.expander("Afficher les détails"):
            st.code('''
                    X = df.drop('grav_grave', axis=1)
                    y = df['grav_grave']
                    ''')

        st.write("")
        st.write("3. Division en ensembles d'entraînement et de test avec stratification :")
        with st.expander("Afficher les détails"):
            st.code('''
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    ''')

        st.write("")
        st.write("4. Normalisation des features :")
        with st.expander("Afficher les détails"):
            st.code('''
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    ''')
        
        st.write("")   
        st.write("5. Création d'un modèle Keras :")
        with st.expander("Afficher les détails"):
            st.write("- Définfition du nombre de features :")
            st.code('''
                    input_shape = X_train_scaled.shape[1]
                    ''')

            st.write("- Fonction de construction du modèle Keras :")
            st.code('''
                    def create_model(optimizer='adam', init='glorot_uniform'):
                        model = Sequential([
                        Dense(64, activation='relu', input_shape=(input_shape,), kernel_initializer=init),
                        Dropout(0.4),
                        Dense(32, activation='relu', kernel_initializer=init),
                        Dropout(0.4),
                        Dense(16, activation='relu', kernel_initializer=init),
                        Dropout(0.4),
                        Dense(1, activation='sigmoid')
                    ])
                    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                    return model
                    ''')
            
        st.write("")   
        st.write("6. Création d'un wrapper KerasClassifier pour utiliser la bibliothèque de scikit-learn :")
        with st.expander("Afficher les détails"):
            st.code('''
                model = KerasClassifier(
                    model=create_model,
                    verbose=0,
                    optimizer='adam',
                    epochs=50,
                    batch_size=32,
                    model__init='glorot_uniform')
                    ''')

        st.write("")   
        st.write("7. Définition des hyperparamètres à tester avec GridSearchCV :")
        with st.expander("Afficher les détails"):
            st.code('''
                param_grid = {
                    'batch_size': [16, 32],
                    'epochs': [10, 50],
                    'optimizer': ['adam', 'rmsprop'],
                    'model__init': ['he_normal', 'he_uniform']}
                    ''')
            
        st.write("")   
        st.write("8. Configuration de GridSearchCV :")
        with st.expander("Afficher les détails"):
            st.code('''
                    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
                    ''')
            
        st.write("")   
        st.write("9. Entraînement du modèle avec recherche des hyperparamètres :")
        with st.expander("Afficher les détails"):
            st.code('''
                    grid_result = grid.fit(X_train_scaled, y_train)
                    ''')

        st.write("")
        if st.checkbox(f"Afficher les meilleurs paramètres et score") :
            st.write("Best score: 0.7820")
            st.write("Best parameters: {'batch_size': 32, 'epochs': 50, 'model__init': 'he_uniform', 'optimizer': 'rmsprop'}")

        st.write("")   
        st.write("10. Création d'un modèle Keras avec les meilleurs paramètres :")
        with st.expander("Afficher les détails"):
            st.code('''
                    best_params = grid_result.best_params_
                    best_model = create_model(optimizer=best_params['optimizer'], init=best_params['model__init'])
                    ''')
            
        st.write("")   
        st.write("11. Entraînement du meilleur modèle :")
        with st.expander("Afficher les détails"):
            st.code('''
                history = best_model.fit(
                    X_train_scaled, y_train,
                    validation_split=0.2,
                    epochs=best_params['epochs'],
                    batch_size=best_params['batch_size'],
                    verbose=1)
                    ''')

        st.write("")
        st.write("12. Performance du modèle :")
        if st.checkbox(f"Afficher l'évaluation sur l'ensemble de test :") :
            st.write("Test loss: 0.4792")
            st.write("Test accuracy: 0.7781")
        if st.checkbox(f"Afficher le rapport de classification") :
            dl_class_rep = pd.read_csv("dl_class_report.csv", sep = '\t', index_col=0)
            st.dataframe(dl_class_rep)
        if st.checkbox(f"Afficher la matrice de confusion") :
            st.write("[[13240  4525]")
            st.write("[ 3358 14406]]")
        if st.checkbox(f"Afficher le score AUC-ROC") :
            st.write("0.8570")

        st.write("")
        st.write("13. Visualisation graphique :")
        with st.expander("Afficher les détails"):     
            st.write("")
            image4 = Image.open("training_history.png")
            st.image(image4, caption="Historique d'entraînement", width=680)
            st.write("=" * 84)
            st.write("")
            image5 = Image.open("feature_importance.png")
            st.image(image5, width=680)

        st.write("")
        st.write("14. Analyse SHAP :")
        with st.expander("Afficher les détails"):     
            st.write("")
            image6 = Image.open("shap_summary_plot.png")
            st.image(image6, caption="Labels explicatifs", width=680)



##############################################################################
#
# Page : Prédiction
#
##############################################################################

if page == pages[4]:
    st.header("Prédiction (démo)")


    warnings.filterwarnings('ignore')

    try:
        # Charger les données
        data_path = os.path.join(os.path.dirname(__file__), "data.csv")
        df = pd.read_csv(data_path, sep="\t")
        df = df.astype(int)

        # Séparer les features et la variable cible
        X = df.drop('grav_grave', axis=1)
        y = df['grav_grave']

        # Obtenir le nombre de features pour input_shape
        input_shape = X.shape[1]

        def create_and_compile_model():
            model = Sequential([
                Dense(64, activation='relu', input_shape=(input_shape,)),
                Dropout(0.4),
                Dense(32, activation='relu'),
                Dropout(0.4),
                Dense(16, activation='relu'),
                Dropout(0.4),
                Dense(1, activation='sigmoid')])
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model

        # Vérifier si le modèle et le scaler existent
        model_architecture_path = 'model_architecture.json'
        model_weights_path = 'model_weights.h5'
        scaler_path = 'scaler_DP.joblib'

        if os.path.exists(model_architecture_path) and os.path.exists(model_weights_path) and os.path.exists(scaler_path):
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
                    with open('model_architecture.json', 'w') as f:
                        f.write(model_json)
                    
                    # Sauvegarder les poids
                    loaded_model.save_weights('model_weights.h5')
                    
                    # Sauvegarder le scaler
                    joblib.dump(scaler, 'scaler_DP.joblib')
                    
                    st.success("Modèle entraîné et sauvegardé avec succès!")

        if loaded_model is not None:
            # Interface de prédiction
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
                    "Octobre": "mois_10", "Novembre": "mois_11", "Décembre": "mois_12"}
                mois = st.selectbox("Mois", options=list(mois_dict.keys()))
                for col in features.columns:
                    if col.startswith('mois_'):
                        features[col] = 0
                features[mois_dict[mois]] = 1

                st.write("")

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

                st.write("")

                # Dispositif de sécurité
                secu_dict = {
                    "Ceinture": "secu_ceinture", "Casque": 'secu_casque', "Dispositif enfant": 'secu_dispenfant', 
                    "Gilet de sécurité": 'secu_gilet', "Airbag": 'secu_airbag23RM', "Gants": 'secu_gants'}
                secu = st.selectbox("Dispositif de sécurité", options=list(secu_dict.keys()))
                for col in features.columns:
                    if col.startswith('secu_'):
                        features[col] = 0
                features[secu_dict[secu]] = 1

                st.write("")        

                # Vitesse maximale autorisée
                vma_dict = {
                    "30 km/h": "vma_30m", "40 km/h": 'vma_40', "50 km/h": 'vma_50', "60 km/h": 'vma_60', "70 km/h": 'vma_70',
                    "80 km/h": 'vma_80', "90 km/h": 'vma_90', "110 km/h": 'vma_100', "130 km/h": 'vma_130' }
                vma = st.select_slider("Vitesse maximale autorisée (km/h)", options=list(vma_dict.keys()))
                for col in features.columns:
                    if col.startswith('vma_'):
                        features[col] = 0
                features[vma_dict[vma]] = 1

                st.write("")

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

                st.write("")

                # Motorisation du véhicule
                motor_dict = {
                    "Hydrocarbure": "motor_1", "Hybride électrique": "motor_2", "Electrique": "motor_3", "Hydrogène": "motor_4"}
                motor = st.selectbox("Motorisation du véhicule", options=list(motor_dict.keys()))
                for col in features.columns:
                    if col.startswith('motor_'):
                        features[col] = 0
                features[motor_dict[motor]] = 1

                st.write("")

                # Situation de l'accident
                situ_dict = {
                    "Sur chaussée": "situ_1", "Sur bande d'arrêt d'urgence": "situ_2", "Sur accotement": "situ_3", 
                    "Sur trottoir": "situ_4", "Sur piste cyclable": "situ_5", "Sur autre voie spéciale": "situ_6"}
                situ = st.selectbox("Situation de l'accident", options=list(situ_dict.keys()))
                for col in features.columns:
                    if col.startswith('situ_'):
                        features[col] = 0
                features[situ_dict[situ]] = 1

                st.write("")

                # Obstacle mobile heurté
                obsm_dict = {
                    "Piéton": "obsm_1", "Véhicule": "obsm_2", "Véhicule sur rail": "obsm_4", 
                    "Animal domestique": "obsm_5", "Animal sauvage": "obsm_6"}
                obsm = st.selectbox("Obstacle mobile heurté", options=list(obsm_dict.keys()))
                for col in features.columns:
                    if col.startswith('obsm_'):
                        features[col] = 0
                features[obsm_dict[obsm]] = 1

                st.write("")

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

                st.write("")

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
                lum_dict = {"Plein jour": "lum_1", "Crépuscule - aube": "lum_2", "Nuit sans éclairage public": "lum_3",
                    "Nuit avec éclairage public non allumé": "lum_4", "Nuit avec éclairage public allumé": "lum_5"}
                lum = st.selectbox("Luminosité", options=list(lum_dict.keys()))
                for col in features.columns:
                    if col.startswith('lum_'):
                        features[col] = 0
                features[lum_dict[lum]] = 1

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

    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")