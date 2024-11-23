import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import unicodedata

# Fonction pour charger les données
def load_data(filename):
    """Charge les données depuis un fichier Excel et renvoie un DataFrame."""
    try:
        data = pd.read_excel(filename)
        st.success("Données chargées avec succès.")
        return data
    except FileNotFoundError:
        st.error(f"Erreur : Le fichier {filename} est introuvable.")
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")

# Fonction de prétraitement des données
def preprocess_data(df, features):
    """Vérifie les colonnes et applique la normalisation aux caractéristiques spécifiées."""
    for feature in features:
        if feature not in df.columns:
            st.error(f"La colonne '{feature}' est manquante dans le fichier de données.")
            st.stop()
    
    x = df[features]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=features)

# Calcul de la matrice de covariance
def compute_covariance_matrix(df_scaled):
    """Calcule la matrice de covariance des données normalisées."""
    cov_matrix = np.cov(df_scaled.T)
    st.write("Matrice de covariance :", cov_matrix)
    return cov_matrix

# Calcul des composantes principales
def perform_pca(cov_matrix, n_components=2):
    """Calcule les valeurs et vecteurs propres, trie et projette les données."""
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    st.write("Valeurs propres :", eigenvalues)
    st.write("Vecteurs propres :", eigenvectors)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    projection_matrix = sorted_eigenvectors[:, :n_components]
    return projection_matrix, sorted_eigenvalues

# Projection des données sur les composantes principales
def project_data(df_scaled, projection_matrix):
    """Projette les données sur les composantes principales."""
    df_pca = df_scaled.dot(projection_matrix)
    st.write("Données projetées sur les composantes principales :", df_pca)
    return df_pca

# Interface utilisateur avec Streamlit
st.title("Analyse en Composantes Principales (ACP)")

# Charger le fichier de données
uploaded_file = st.file_uploader("Téléchargez le fichier Excel", type=["xlsx"])

if uploaded_file:
    # Charger les données
    df = load_data(uploaded_file)
    
    if df is not None:
        # Normaliser les noms de colonnes pour éviter les accents
        df.columns = [unicodedata.normalize('NFKD', col).encode('ascii', 'ignore').decode('utf-8') for col in df.columns]
        
        # Sélectionner les caractéristiques à analyser
        features = ["Poids", "Taille", "Age", "Note"]
        
        try:
            # Prétraitement des données
            df_scaled = preprocess_data(df, features)
            

            # Calcul de la matrice de covariance
            cov_matrix = compute_covariance_matrix(df_scaled)
            
            # Exécution de l'ACP
            projection_matrix, eigenvalues = perform_pca(cov_matrix, n_components=2)
            
            # Projection des données
            df_pca = project_data(df_scaled, projection_matrix)
            
            # Affichage des résultats
            st.write("Projection Matrix (Principal Components) :", projection_matrix)
            st.write("Eigenvalues (Valeurs propres) :", eigenvalues[:2])  # Seulement les deux plus grandes valeurs propres
            
        except ValueError as ve:
            st.error(f"Erreur de validation des données : {ve}")
        except Exception as e:
            st.error(f"Une erreur inattendue s'est produite : {e}")
