import streamlit as st
import joblib
import re
import numpy as np

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="wide"
)

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("Tfidf_Vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ======================
# CLEAN FUNCTION
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# ======================
# CONFIDENCE NORMALIZATION
# ======================
def normalize_confidence(decision_value):
    v = abs(float(decision_value)) * 3
    sigmoid = 1 / (1 + np.exp(-v))
    confidence = sigmoid * 100
    return round(min(max(confidence, 51), 99), 1)

# ======================
# TITLE
# ======================
st.title("💬 Analyse des avis clients")
st.markdown("### Dashboard intelligent basé sur SVM + TF-IDF")

st.divider()

# ======================
# LAYOUT
# ======================
col1, col2 = st.columns([2, 1])

# ======================
# INPUT
# ======================
with col1:
    avis = st.text_area(
        "✍️ Entrez votre avis",
        placeholder="Exemple : Ce produit est incroyable, livraison rapide et excellente qualité.",
        height=220
    )
    analyse = st.button("🔍 Analyser le sentiment", use_container_width=True)

# ======================
# RESULT SECTION
# ======================
with col2:
    st.subheader("📊 Résultat")

# ======================
# ANALYSIS
# ======================
if analyse:

    if avis.strip() == "":
        st.warning("⚠️ Veuillez entrer un texte.")

    else:
        # CLEAN
        cleaned = clean_text(avis)

        # VECTORIZE
        avis_vec = vectorizer.transform([cleaned])

        # PREDICTION
        prediction = model.predict(avis_vec)[0]

        # CONFIDENCE SCORE (SVM sigmoid normalisé)
        decision = model.decision_function(avis_vec)
        confidence = normalize_confidence(decision[0])
        confidence_display = int(confidence) / 100

        # ======================
        # POSITIVE
        # ======================
        if prediction == 1 or prediction == "positive":
            with col2:
                st.success("😊 Avis POSITIF")
                st.metric(
                    label="Confiance",
                    value=f"{confidence}%",
                    delta="Sentiment positif détecté"
                )
                st.progress(confidence_display)

        # ======================
        # NEGATIVE
        # ======================
        else:
            with col2:
                st.error("😡 Avis NÉGATIF")
                st.metric(
            label="Confiance",
            value=f"{confidence}%"
        )
                st.markdown("⬇️ Sentiment négatif détecté")
                st.progress(confidence_display)

        # ======================
        # DETAILS
        # ======================
        st.divider()

        with st.expander("📈 Détails de l'analyse"):
            detail_col1, detail_col2 = st.columns(2)

            with detail_col1:
                st.write("🔹 **Algorithme :** SVM (LinearSVC)")
                st.write("🔹 **Feature Extraction :** TF-IDF")
                st.write("🔹 **Type :** Classification binaire")

            with detail_col2:
                st.write(f"🔹 **Longueur du texte :** {len(avis)} caractères")
                st.write(f"🔹 **Nombre de mots :** {len(avis.split())}")
                st.write(f"🔹 **Score brut SVM :** {round(float(decision[0]), 4)}")

            st.text_area("Texte analysé", avis, height=100)

# ======================
# FOOTER
# ======================
st.divider()
st.caption("Projet PFE - Analyse des sentiments avec Machine Learning")