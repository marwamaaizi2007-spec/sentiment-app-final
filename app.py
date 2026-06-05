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
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ======================
# CLEAN FUNCTION
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

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

    analyse = st.button("🔍 Analyser le sentiment")

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

        # ======================
        # CONFIDENCE SCORE (SVM)
        # ======================
        decision = model.decision_function(avis_vec)

        confidence = round(abs(decision[0]) * 100, 2)

        # LIMIT DISPLAY
        confidence_display = min(int(confidence), 100)

        # ======================
        # POSITIVE
        # ======================
        if prediction == 1 or prediction == "positive":

            with col2:

                st.success("😊 Avis POSITIF")

                st.metric(
                    label="Confiance",
                    value=f"{confidence}%"
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

                st.progress(confidence_display)

        # ======================
        # DETAILS
        # ======================
        st.divider()

        with st.expander("📈 Détails du modèle"):

            st.write("🔹 Algorithme : SVM (LinearSVC)")
            st.write("🔹 Feature Extraction : TF-IDF")
            st.write("🔹 Type : Classification binaire")
            st.write(f"🔹 Longueur du texte : {len(avis)} caractères")

            st.text_area(
                "Texte analysé",
                avis,
                height=150
            )

# ======================
# FOOTER
# ======================
st.divider()

st.caption("Projet PFE - Analyse des sentiments avec Machine Learning")
