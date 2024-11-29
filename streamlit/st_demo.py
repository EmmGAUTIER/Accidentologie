import streamlit as st
import json
import pandas as pd
import os
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

@st.cache_data
def read_models_DP():
    return None, None
"""        
@st.cache_data
def read_models_DP():

    rep_raw = "data/raw/"
    rep_processed = "data/processed/"
    rep_models = "models/"
    rep_figures = "reports/figures/"
    rep_ref = "references/"

    loaded_model = None
    scaler = None

    try:
        # Charger l'architecture du modèle
        with open(rep_models + "model_architecture.json", 'r') as f:
            model_json = f.read()
        loaded_model = model_from_json(model_json)
        # Charger les poids
        loaded_model.load_weights(rep_models + "model_weights.h5")
        
        # Compiler le modèle
        loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Charger le scaler
        scaler = joblib.load(rep_models + "scaler_DP.joblib")
        st.success("Modèle chargé avec succès!")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")


    return loaded_model, scaler
"""        

"""
def create_and_compile_model(input_shape):
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

"""

def st_demo():
    st.header("Prédiction (démo)")
    warnings.filterwarnings('ignore')


    #model, scaler = read_models_DP()

"""
    try:
        # Charger les données
        # data_path = os.path.join(os.path.dirname(__file__), "data.csv")
        # df = pd.read_csv(data_path, sep="\t")
        # df = df.astype(int)

        # Séparer les features et la variable cible
        # X = df.drop('grav_grave', axis=1)
        # y = df['grav_grave']

        # Obtenir le nombre de features pour input_shape
        # input_shape = X.shape[1]

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
                    loaded_model = create_and_compile_model(input_shape)
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
                        st.success(f"Risque faible d'accident grave (Probabilité : {proba_grave:.1%})")

                except Exception as e:
                    st.error(f"Erreur lors de la prédiction : {str(e)}")
                    st.write("Debug - Features utilisées:", features.columns[features.iloc[0] == 1].tolist())

"""

