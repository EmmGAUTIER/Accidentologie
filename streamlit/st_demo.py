import streamlit as st
import json
import pandas as pd
#import numpy as np

from sklearn.decomposition         import PCA
from sklearn.preprocessing         import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf

import joblib

# Lecture des modèles entraînés

@st.cache_data
def read_models():

    model_arch, model_weights, scaler_DP = None, None, None

    try:
        model_arch    = joblib.load(rep_models + "model_architecture.json")
        model_weights = joblib.load(rep_models + "model_weights.h5")
        scaler_DP     = joblib.load(rep_models + "scaler_DP.joblib")

    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")

    return model_arch, model_weights, scaler_DP

# Lecture du jeu de données avec une observation à zero
# ce jeu de données avec une seule observation est utilisé par la démo.

def st_demo():
    model_arch, model_weights, scaler_DP = read_models()
