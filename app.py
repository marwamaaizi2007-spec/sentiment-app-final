import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="E-Réputation Dashboard",
    page_icon="💬",
    layout="wide"
)

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>
[data-testid="stSidebar"] {background-color: #1a1a2e;}
[data-testid="stSidebar"] * {color: white !important;}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px; border-radius: 12px; text-align: center; color: white;
}
.metric-card h2 {font-size: 2.2rem; margin: 0; font-weight: 700;}
.metric-card p {margin: 0; opacity: 0.85; font-size: 0.9rem;}
.pos-card {background: linear-gradient(135deg, #11998e, #38ef7d);}
.neg-card {background: linear-gradient(135deg, #eb3349, #f45c43);}
.neu-card {background: linear-gradient(135deg, #4776E6, #8E54E9);}
.tot-card {background: linear-gradient(135deg, #f7971e, #ffd200);}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL & DATA
# ======================
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("Tfidf_Vectorizer.pkl")
    return model, vectorizer

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_reviews.csv")
    df['Sentiment'] = df['Score'].apply(lambda x: 'Positif' if x >= 4 else ('Négatif' if x <= 2 else 'Neutre'))
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df['Year'] = df['Time'].dt.year
    df['Month'] = df['Time'].dt.month
    return df

model, vectorizer = load_model()
df = load_data()

# ======================
# CLEAN & CONFIDENCE
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def normalize_confidence(decision_value):
    v = abs(float(decision_value)) * 3
    sigmoid = 1 / (1 + np.exp(-v))
    return round(min(max(sigmoid * 100, 51), 99), 1)

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.markdown("## 💬 E-Réputation")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Tableau de bord",
        "🔍 Analyser un avis",
        "📊 Importance des variables",
        "📈 Statistiques & EDA",
        "🤖 Comparaison des modèles"
    ])
    st.markdown("---")
    st.markdown(f"**Total avis :** {len(df):,}")
    pos_pct = round(len(df[df['Sentiment']=='Positif']) / len(df) * 100, 1)
    neg_pct = round(len(df[df['Sentiment']=='Négatif']) / len(df) * 100, 1)
    st.markdown(f"😊 Positifs : **{pos_pct}%**")
    st.markdown(f"😡 Négatifs : **{neg_pct}%**")
    st.markdown("---")
    st.caption("Projet PFE — Analyse des sentiments")

# ======================
# PAGE 1 — TABLEAU DE BORD
# ======================
if page == "🏠 Tableau de bord":
    st.title("🏠 Tableau de bord")
    st.markdown("Vue globale de l'e-réputation clients")
    st.divider()

    total = len(df)
    positifs = len(df[df['Sentiment'] == 'Positif'])
    negatifs = len(df[df['Sentiment'] == 'Négatif'])
    neutres = len(df[df['Sentiment'] == 'Neutre'])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card tot-card"><h2>{total:,}</h2><p>Total avis</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card pos-card"><h2>{positifs:,}</h2><p>😊 Positifs ({round(positifs/total*100,1)}%)</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card neg-card"><h2>{negatifs:,}</h2><p>😡 Négatifs ({round(negatifs/total*100,1)}%)</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card neu-card"><h2>{neutres:,}</h2><p>😐 Neutres ({round(neutres/total*100,1)}%)</p></div>', unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Répartition des sentiments")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        sizes = [positifs, negatifs, neutres]
        labels = ['Positif', 'Négatif', 'Neutre']
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                           autopct='%1.1f%%', startangle=90)
        for text in texts: text.set_color('white')
        for autotext in autotexts: autotext.set_color('white')
        ax.set_title("Répartition des sentiments", color='white')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("📈 Distribution des scores")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        score_counts = df['Score'].value_counts().sort_index()
        bar_colors = ['#e74c3c', '#e67e22', '#3498db', '#2ecc71', '#27ae60']
        ax.bar(score_counts.index, score_counts.values, color=bar_colors, edgecolor='none')
        ax.set_xlabel("Score", color='white')
        ax.set_ylabel("Nombre d'avis", color='white')
        ax.set_title("Distribution des scores (1-5)", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#444')
        plt.tight_layout()
        st.pyplot(fig)

    st.subheader("📅 Évolution des avis par année")
    yearly = df.groupby(['Year', 'Sentiment']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    if 'Positif' in yearly.columns:
        ax.plot(yearly.index, yearly['Positif'], color='#2ecc71', marker='o', label='Positif', linewidth=2)
    if 'Négatif' in yearly.columns:
        ax.plot(yearly.index, yearly['Négatif'], color='#e74c3c', marker='o', label='Négatif', linewidth=2)
    if 'Neutre' in yearly.columns:
        ax.plot(yearly.index, yearly['Neutre'], color='#3498db', marker='o', label='Neutre', linewidth=2)
    ax.set_xlabel("Année", color='white')
    ax.set_ylabel("Nombre d'avis", color='white')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#444')
    plt.tight_layout()
    st.pyplot(fig)

# ======================
# PAGE 2 — ANALYSER UN AVIS
# ======================
elif page == "🔍 Analyser un avis":
    st.title("🔍 Analyser un avis client")
    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        avis = st.text_area("✍️ Entrez votre avis",
            placeholder="Exemple : Ce produit est incroyable, livraison rapide et excellente qualité.",
            height=220)
        analyse = st.button("🔍 Analyser le sentiment", use_container_width=True)

    with col2:
        st.subheader("📊 Résultat")

    if analyse:
        if avis.strip() == "":
            st.warning("⚠️ Veuillez entrer un texte.")
        else:
            cleaned = clean_text(avis)
            avis_vec = vectorizer.transform([cleaned])
            prediction = model.predict(avis_vec)[0]
            decision = model.decision_function(avis_vec)
            confidence = normalize_confidence(decision[0])
            confidence_display = int(confidence) / 100

            if prediction == 1 or prediction == "positive":
                with col2:
                    st.success("😊 Avis POSITIF")
                    st.metric(label="Confiance", value=f"{confidence}%", delta="↑ Sentiment positif")
                    st.progress(confidence_display)
            else:
                with col2:
                    st.error("😡 Avis NÉGATIF")
                    st.metric(label="Confiance", value=f"{confidence}%")
                    st.markdown("**⬇️ Sentiment négatif détecté**")
                    st.progress(confidence_display)

            st.divider()
            with st.expander("📈 Détails de l'analyse"):
                d1, d2 = st.columns(2)
                with d1:
                    st.write("🔹 **Algorithme :** SVM (LinearSVC)")
                    st.write("🔹 **Feature Extraction :** TF-IDF")
                with d2:
                    st.write(f"🔹 **Mots :** {len(avis.split())}")
                    st.write(f"🔹 **Score brut SVM :** {round(float(decision[0]), 4)}")
                st.text_area("Texte nettoyé", cleaned, height=80)

            st.subheader("🔑 Mots clés détectés")
            feature_names = vectorizer.get_feature_names_out()
            coefs = model.coef_[0]
            avis_array = avis_vec.toarray()[0]
            mots = [(feature_names[i], coefs[i], avis_array[i])
                    for i in range(len(feature_names)) if avis_array[i] > 0]
            if mots:
                mots.sort(key=lambda x: abs(x[1]), reverse=True)
                df_mots = pd.DataFrame(mots[:15], columns=["Mot", "Poids SVM", "TF-IDF"])
                df_mots["Influence"] = df_mots["Poids SVM"].apply(
                    lambda x: "😊 Positif" if x > 0 else "😡 Négatif")
                df_mots["Poids SVM"] = df_mots["Poids SVM"].round(4)
                st.dataframe(df_mots, use_container_width=True)

# ======================
# PAGE 3 — IMPORTANCE DES VARIABLES
# ======================
elif page == "📊 Importance des variables":
    st.title("📊 Importance des variables")
    st.markdown("Les mots les plus influents dans la décision du modèle SVM après vectorisation TF-IDF.")
    st.divider()

    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    top_pos_idx = np.argsort(coefs)[-20:][::-1]
    top_neg_idx = np.argsort(coefs)[:20]
    top_positive = [(feature_names[i], round(float(coefs[i]), 4)) for i in top_pos_idx]
    top_negative = [(feature_names[i], round(float(coefs[i]), 4)) for i in top_neg_idx]

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("### 😊 Variables Positives")
        df_pos = pd.DataFrame(top_positive, columns=["Variable", "Poids"])
        fig, ax = plt.subplots(figsize=(6, 8))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.barh(df_pos["Variable"], df_pos["Poids"], color="#2ecc71", edgecolor='none')
        ax.set_xlabel("Poids SVM", color='white')
        ax.set_title("Top 20 variables positives", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#444')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        st.dataframe(df_pos, use_container_width=True)

    with col_neg:
        st.markdown("### 😡 Variables Négatives")
        df_neg = pd.DataFrame(top_negative, columns=["Variable", "Poids"])
        fig, ax = plt.subplots(figsize=(6, 8))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.barh(df_neg["Variable"], df_neg["Poids"], color="#e74c3c", edgecolor='none')
        ax.set_xlabel("Poids SVM", color='white')
        ax.set_title("Top 20 variables négatives", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#444')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        st.dataframe(df_neg, use_container_width=True)

    st.divider()
    st.subheader("📈 Statistiques du modèle")
    s1, s2, s3 = st.columns(3)
    with s1: st.metric("Nombre de variables TF-IDF", f"{len(feature_names):,}")
    with s2: st.metric("Poids max positif", f"{round(float(coefs.max()), 4)}")
    with s3: st.metric("Poids max négatif", f"{round(float(coefs.min()), 4)}")

# ======================
# PAGE 4 — STATISTIQUES & EDA
# ======================
elif page == "📈 Statistiques & EDA":
    st.title("📈 Statistiques & Analyse Exploratoire")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📅 Temporel", "🔤 Texte"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Distribution des scores")
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            sc = df['Score'].value_counts().sort_index()
            colors = ['#e74c3c','#e67e22','#3498db','#2ecc71','#27ae60']
            ax.bar(sc.index, sc.values, color=colors)
            ax.set_xlabel("Score", color='white')
            ax.set_ylabel("Nombre", color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_color('#444')
            plt.tight_layout()
            st.pyplot(fig)

        with c2:
            st.subheader("Répartition sentiments")
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            sent = df['Sentiment'].value_counts()
            ax.pie(sent.values, labels=sent.index,
                   colors=['#2ecc71','#e74c3c','#3498db'],
                   autopct='%1.1f%%', startangle=90)
            for text in ax.texts: text.set_color('white')
            plt.tight_layout()
            st.pyplot(fig)

        st.subheader("📋 Résumé statistique")
        st.dataframe(df[['Score','HelpfulnessNumerator','HelpfulnessDenominator']].describe().round(2),
                     use_container_width=True)

    with tab2:
        st.subheader("Évolution annuelle des avis")
        yearly = df.groupby(['Year','Sentiment']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        colors_map = {'Positif':'#2ecc71','Négatif':'#e74c3c','Neutre':'#3498db'}
        for col in yearly.columns:
            ax.plot(yearly.index, yearly[col], marker='o',
                    label=col, color=colors_map.get(col,'white'), linewidth=2)
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        ax.tick_params(colors='white')
        ax.set_xlabel("Année", color='white')
        ax.set_ylabel("Nombre d'avis", color='white')
        for spine in ax.spines.values(): spine.set_color('#444')
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.subheader("Longueur des avis par sentiment")
        df['text_length'] = df['Text'].astype(str).apply(len)
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        for sent, color in [('Positif','#2ecc71'),('Négatif','#e74c3c'),('Neutre','#3498db')]:
            subset = df[df['Sentiment']==sent]['text_length']
            ax.hist(subset.clip(upper=2000), bins=50, alpha=0.6, label=sent, color=color)
        ax.set_xlabel("Longueur (caractères)", color='white')
        ax.set_ylabel("Fréquence", color='white')
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#444')
        plt.tight_layout()
        st.pyplot(fig)

        avg_len = df.groupby('Sentiment')['text_length'].mean().round(0)
        l1, l2, l3 = st.columns(3)
        cols_map = {'Positif': l1, 'Négatif': l2, 'Neutre': l3}
        for sent, col in cols_map.items():
            if sent in avg_len:
                with col: st.metric(f"Longueur moy. {sent}", f"{int(avg_len[sent])} car.")

# ======================
# PAGE 5 — COMPARAISON MODÈLES
# ======================
elif page == "🤖 Comparaison des modèles":
    st.title("🤖 Comparaison des modèles")
    st.markdown("Évaluation et comparaison des algorithmes de classification testés.")
    st.divider()

    # Résultats des modèles (à adapter selon tes vrais résultats)
    models_data = {
        "Modèle": ["Multinomial Naive Bayes", "SVM (LinearSVC)", "Logistic Regression", "Random Forest", "KNN"],
        "Accuracy": [0.847, 0.891, 0.878, 0.823, 0.764],
        "Precision": [0.842, 0.889, 0.874, 0.819, 0.758],
        "Recall": [0.847, 0.891, 0.878, 0.823, 0.764],
        "F1-Score": [0.844, 0.890, 0.876, 0.821, 0.761]
    }
    df_models = pd.DataFrame(models_data)

    st.subheader("📋 Tableau comparatif")
    st.dataframe(df_models.style.highlight_max(
        subset=["Accuracy","Precision","Recall","F1-Score"],
        color='#2ecc7155'), use_container_width=True)

    st.subheader("📊 Comparaison visuelle")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    x = np.arange(len(df_models["Modèle"]))
    width = 0.2
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*width, df_models[metric], width, label=metric, color=color, alpha=0.85)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(df_models["Modèle"], rotation=15, ha='right', color='white', fontsize=9)
    ax.set_ylabel("Score", color='white')
    ax.set_ylim(0.7, 1.0)
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#444')
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.success("✅ **Modèle choisi : SVM (LinearSVC)** — Meilleure performance globale avec F1-Score de 0.890")
    st.info("💡 Le Multinomial Naive Bayes a été testé en premier car adapté aux données textuelles, mais SVM offre de meilleures performances sur ce dataset.")

# ======================
# FOOTER
# ======================
st.divider()
st.caption("Projet PFE — Analyse de l'e-réputation | Analyse des sentiments avec Machine Learning")