import streamlit as st
import json
import pandas as pd
#import os
#import warnings
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split
#from tensorflow.keras.models import Sequential, model_from_json
#from tensorflow.keras.layers import Dense, Dropout
#import seaborn as sns
#import matplotlib.pyplot as plt
#import tensorflow as tf
import joblib

@st.cache_data
def read_models():
    model_arch, model_weights, scaler_DP = None, None, None
    try:
        model_arch = joblib.load(rep_models + "model_architecture.json")
        model_weights = joblib.load(rep_models + "model_weights.h5")
        scaler_DP = joblib.load(rep_models + "scaler_DP.joblib")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {str(e)}")
    return model_arch, model_weights, scaler_DP

def st_demo():
    st.header("Prédiction (démo)")

