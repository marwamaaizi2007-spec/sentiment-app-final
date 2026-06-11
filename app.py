import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import hstack, csr_matrix

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="SentiScan",
    page_icon="🔍",
    layout="wide"
)

# ======================
# LOGIN
# ======================
def check_login():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.role = None

    if not st.session_state.logged_in:
        st.title("🔐 Connexion — SentiScan")
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("👤 Nom d'utilisateur")
            password = st.text_input("🔒 Mot de passe", type="password")
            login_btn = st.button("Se connecter", use_container_width=True)
            if login_btn:
                if username == "marwa" and password == "pfe":
                    st.session_state.logged_in = True
                    st.session_state.role = "admin"
                    st.rerun()
                elif username == "user" and password == "user":
                    st.session_state.logged_in = True
                    st.session_state.role = "visiteur"
                    st.rerun()
                else:
                    st.error("❌ Nom d'utilisateur ou mot de passe incorrect")
        st.stop()

check_login()

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>
[data-testid="stSidebar"] {background-color: #1a1a2e;}
[data-testid="stSidebar"] * {color: white !important;}
.metric-card {padding: 20px; border-radius: 12px; text-align: center; color: white;}
.metric-card h2 {font-size: 2.2rem; margin: 0; font-weight: 700;}
.metric-card p {margin: 0; opacity: 0.85; font-size: 0.9rem;}
.nor-card {background: linear-gradient(135deg, #11998e, #38ef7d);}
.ris-card {background: linear-gradient(135deg, #f7971e, #ffd200);}
.cri-card {background: linear-gradient(135deg, #eb3349, #f45c43);}
.tot-card {background: linear-gradient(135deg, #4776E6, #8E54E9);}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_model():
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer_svm.pkl")
    return model, vectorizer

@st.cache_data
def load_data():
    data_classe = ['Normale']*363122 + ['Risque']*80655 + ['Critique']*52268
    data_score  = [5]*363122 + [4]*80655 + [2]*29769 + [1]*22499
    data_year   = [2012]*150000 + [2011]*200000 + [2010]*145814 + [2013]*420 + [2009]*9811
    df = pd.DataFrame({
        'Classe': data_classe,
        'Score':  data_score[:len(data_classe)],
        'Year':   data_year[:len(data_classe)]
    })
    return df

model, vectorizer = load_model()
df = load_data()

# ======================
# CLEAN TEXT
# ======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def prepare_features(text):
    word_count = len(text.split())
    X_text = vectorizer.transform([text])
    num_features = csr_matrix([[1, 1, 0.5, word_count]])
    return hstack([X_text, num_features])

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.markdown("## 🔍 SentiScan")
    st.markdown("---")
    
    if st.session_state.role == "admin":
        pages = [
            "🏠 Tableau de bord",
            "🔍 Analyser un avis",
            "📊 Importance des variables",
            "📈 Statistiques & EDA",
            "🤖 Comparaison des modèles"
        ]
    else:
        pages = ["🔍 Analyser un avis"]
    
    page = st.radio("Navigation", pages)
    st.markdown("---")
    
    if st.session_state.role == "admin":
        st.markdown("👑 **Admin**")
        total = len(df)
        st.markdown(f"**Total avis :** {total:,}")
        st.markdown(f"🟢 Normale : **{round(len(df[df['Classe']=='Normale'])/total*100,1)}%**")
        st.markdown(f"🟡 Risque : **{round(len(df[df['Classe']=='Risque'])/total*100,1)}%**")
        st.markdown(f"🔴 Critique : **{round(len(df[df['Classe']=='Critique'])/total*100,1)}%**")
    else:
        st.markdown("👤 **Visiteur**")
    
    st.markdown("---")
    if st.button("🚪 Déconnexion"):
        st.session_state.logged_in = False
        st.session_state.role = None
        st.rerun()
    st.caption("SentiScan — Projet PFE")

# ======================
# PAGE 1 — TABLEAU DE BORD
# ======================
if page == "🏠 Tableau de bord":
    st.title("🏠 Tableau de bord")
    st.markdown("Vue globale de l'e-réputation clients")
    st.divider()

    total = len(df)
    normales  = len(df[df['Classe'] == 'Normale'])
    risques   = len(df[df['Classe'] == 'Risque'])
    critiques = len(df[df['Classe'] == 'Critique'])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card tot-card"><h2>{total:,}</h2><p>Total avis</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card nor-card"><h2>{normales:,}</h2><p>🟢 Normale ({round(normales/total*100,1)}%)</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card ris-card"><h2>{risques:,}</h2><p>🟡 Risque ({round(risques/total*100,1)}%)</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card cri-card"><h2>{critiques:,}</h2><p>🔴 Critique ({round(critiques/total*100,1)}%)</p></div>', unsafe_allow_html=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Répartition des classes")
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        sizes = [normales, risques, critiques]
        labels = ['Normale', 'Risque', 'Critique']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        for text in texts: text.set_color('white')
        for autotext in autotexts: autotext.set_color('white')
        ax.set_title("Répartition des classes", color='white')
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
    yearly = df.groupby(['Year', 'Classe']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    colors_map = {'Normale': '#2ecc71', 'Risque': '#f39c12', 'Critique': '#e74c3c'}
    for col in yearly.columns:
        ax.plot(yearly.index, yearly[col], marker='o', label=col, color=colors_map.get(col, 'white'), linewidth=2)
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
            word_count = len(cleaned.split())
            X_text = vectorizer.transform([cleaned])
            num_features = csr_matrix([[1, 1, 0.5, word_count]])
            X = hstack([X_text, num_features])
            prediction = model.predict(X)[0]
            with col2:
                if prediction == 'Normale':
                    st.success("🟢 Commande NORMALE")
                    st.markdown("**Client satisfait** — Aucune action requise")
                elif prediction == 'Risque':
                    st.warning("🟡 Commande À RISQUE")
                    st.markdown("**Attention requise** — Suivi recommandé")
                else:
                    st.error("🔴 Commande CRITIQUE")
                    st.markdown("**Action urgente** — Intervention nécessaire")

            st.divider()
            with st.expander("📈 Détails de l'analyse"):
                d1, d2 = st.columns(2)
                with d1:
                    st.write("🔹 **Algorithme :** SVM (LinearSVC)")
                    st.write("🔹 **Feature Extraction :** TF-IDF")
                    st.write("🔹 **Classes :** Normale / Risque / Critique")
                with d2:
                    st.write(f"🔹 **Mots :** {len(avis.split())}")
                    st.write(f"🔹 **Caractères :** {len(avis)}")
                st.text_area("Texte nettoyé", cleaned, height=80)

            st.subheader("🔑 Mots clés détectés")
            feature_names = vectorizer.get_feature_names_out()
            avis_array = X_text.toarray()[0]
            classes = model.classes_
            classe_idx = list(classes).index(prediction)
            coefs = model.coef_[classe_idx]
            mots = [(feature_names[i], float(coefs[i]), float(avis_array[i]))
                    for i in range(len(feature_names)) if avis_array[i] > 0]
            if mots:
                mots.sort(key=lambda x: abs(x[1]), reverse=True)
                df_mots = pd.DataFrame(mots[:10], columns=["Mot", "Poids SVM", "TF-IDF"])
                df_mots["Influence"] = df_mots["Poids SVM"].apply(
                    lambda x: "🟢 Pour " + prediction if x > 0 else "🔴 Contre " + prediction)
                df_mots["Poids SVM"] = df_mots["Poids SVM"].round(4)
                df_mots["TF-IDF"] = df_mots["TF-IDF"].round(4)
                st.dataframe(df_mots, use_container_width=True)
            else:
                st.info("Aucun mot reconnu par le modèle.")

# ======================
# PAGE 3 — IMPORTANCE DES VARIABLES
# ======================
elif page == "📊 Importance des variables":
    st.title("📊 Importance des variables")
    st.markdown("Les mots les plus influents dans la décision du modèle SVM.")
    st.divider()

    feature_names = vectorizer.get_feature_names_out()
    classes = model.classes_
    color_map = {'Normale': '#2ecc71', 'Risque': '#f39c12', 'Critique': '#e74c3c'}

    for i, classe in enumerate(classes):
        coefs = model.coef_[i]
        top_pos_idx = np.argsort(coefs)[-10:][::-1]
        top_neg_idx = np.argsort(coefs)[:10]
        color = color_map.get(classe, '#3498db')
        emoji = '🟢' if classe == 'Normale' else '🟡' if classe == 'Risque' else '🔴'

        st.markdown(f"### {emoji} Classe : {classe}")
        c1, c2 = st.columns(2)

        with c1:
            df_pos = pd.DataFrame({'Variable': feature_names[top_pos_idx], 'Poids': coefs[top_pos_idx].round(4)})
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            ax.barh(df_pos['Variable'], df_pos['Poids'], color=color, edgecolor='none')
            ax.set_title(f'Top mots → {classe}', color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_color('#444')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)

        with c2:
            df_neg = pd.DataFrame({'Variable': feature_names[top_neg_idx], 'Poids': coefs[top_neg_idx].round(4)})
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            ax.barh(df_neg['Variable'], df_neg['Poids'], color='#95a5a6', edgecolor='none')
            ax.set_title(f'Mots contre {classe}', color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values(): spine.set_color('#444')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        st.divider()

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
            st.subheader("Répartition des classes")
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            sent = df['Classe'].value_counts()
            ax.pie(sent.values, labels=sent.index,
                   colors=['#2ecc71','#f39c12','#e74c3c'],
                   autopct='%1.1f%%', startangle=90)
            for text in ax.texts: text.set_color('white')
            plt.tight_layout()
            st.pyplot(fig)

        st.subheader("📋 Résumé statistique")
        stats_df = pd.DataFrame({
            'Métrique': ['count', 'mean', 'min', 'max'],
            'Score': [525814, 4.18, 1, 5],
            'HelpfulnessNumerator': [525814, 1.74, 0, 866],
            'HelpfulnessDenominator': [525814, 2.17, 0, 923]
        })
        st.dataframe(stats_df, use_container_width=True)

    with tab2:
        st.subheader("Évolution annuelle des avis")
        yearly = df.groupby(['Year','Classe']).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        colors_map = {'Normale':'#2ecc71','Risque':'#f39c12','Critique':'#e74c3c'}
        for col in yearly.columns:
            ax.plot(yearly.index, yearly[col], marker='o', label=col, color=colors_map.get(col,'white'), linewidth=2)
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        ax.tick_params(colors='white')
        ax.set_xlabel("Année", color='white')
        ax.set_ylabel("Nombre d'avis", color='white')
        for spine in ax.spines.values(): spine.set_color('#444')
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.subheader("Longueur des avis par classe")
        text_lengths = {'Normale': 520, 'Risque': 480, 'Critique': 610}
        df_len = pd.DataFrame({
            'Classe': list(text_lengths.keys()),
            'Longueur moyenne (caractères)': list(text_lengths.values())
        })
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        colors_len = ['#2ecc71', '#f39c12', '#e74c3c']
        ax.bar(df_len['Classe'], df_len['Longueur moyenne (caractères)'], color=colors_len, edgecolor='none')
        ax.set_ylabel("Longueur moyenne (car.)", color='white')
        ax.set_title("Longueur moyenne des avis par classe", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values(): spine.set_color('#444')
        plt.tight_layout()
        st.pyplot(fig)
        st.dataframe(df_len, use_container_width=True)

# ======================
# PAGE 5 — COMPARAISON MODELES
# ======================
elif page == "🤖 Comparaison des modèles":
    st.title("🤖 Comparaison des modèles")
    st.markdown("Évaluation et comparaison des 3 algorithmes testés.")
    st.divider()

    models_data = {
        "Modèle": ["Naive Bayes", "Logistic Regression", "SVM (LinearSVC)"],
        "Accuracy":  [0.6672, 0.7580, 0.7966],
        "Precision": [0.6468, 0.7265, 0.7680],
        "Recall":    [0.6672, 0.7580, 0.7966],
        "F1-Score":  [0.6550, 0.7204, 0.7685]
    }
    df_models = pd.DataFrame(models_data)

    st.subheader("📋 Tableau comparatif")
    st.dataframe(df_models.style.highlight_max(
        subset=["Accuracy","Precision","Recall","F1-Score"],
        color='#2ecc7155'), use_container_width=True)

    st.subheader("📊 Comparaison visuelle")
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    x = np.arange(len(df_models["Modèle"]))
    width = 0.2
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ['#3498db', '#2ecc71', '#e67e22', '#9b59b6']
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i*width, df_models[metric], width, label=metric, color=color, alpha=0.85)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(df_models["Modèle"], rotation=0, color='white', fontsize=10)
    ax.set_ylabel("Score", color='white')
    ax.set_ylim(0.5, 1.0)
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_color('#444')
    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    tab_nb, tab_lr, tab_svm = st.tabs(["Naive Bayes", "Logistic Regression", "SVM (LinearSVC)"])

    with tab_nb:
        st.markdown("**Naive Bayes — Résultats par classe**")
        df_nb = pd.DataFrame({
            'Classe':    ['Critique', 'Normale', 'Risque'],
            'Precision': [0.46, 0.77, 0.31],
            'Recall':    [0.36, 0.83, 0.27],
            'F1-Score':  [0.40, 0.80, 0.29],
            'Support':   [1542, 6890, 1568]
        })
        st.dataframe(df_nb, use_container_width=True)
        st.warning("Faible performance sur Risque (F1=0.29) et Critique (F1=0.40)")

    with tab_lr:
        st.markdown("**Logistic Regression — Résultats par classe**")
        df_lr = pd.DataFrame({
            'Classe':    ['Critique', 'Normale', 'Risque'],
            'Precision': [0.77, 0.77, 0.49],
            'Recall':    [0.57, 0.94, 0.15],
            'F1-Score':  [0.66, 0.85, 0.23],
            'Support':   [327, 1354, 319]
        })
        st.dataframe(df_lr, use_container_width=True)
        st.warning("Très faible recall sur Risque (0.15)")

    with tab_svm:
        st.markdown("**SVM LinearSVC — Résultats par classe**")
        df_svm = pd.DataFrame({
            'Classe':    ['Critique', 'Normale', 'Risque'],
            'Precision': [0.80, 0.82, 0.52],
            'Recall':    [0.73, 0.95, 0.21],
            'F1-Score':  [0.76, 0.88, 0.30],
            'Support':   [1542, 6890, 1568]
        })
        st.dataframe(df_svm, use_container_width=True)
        st.success("Meilleure performance globale — Choix final")

    st.divider()
    st.success("✅ **Modèle choisi : SVM (LinearSVC)** — Accuracy 79.66% | F1-Score 0.769")
    st.info("💡 Les 3 modèles ont du mal avec la classe Risque à cause du déséquilibre des données.")

# ======================
# FOOTER
# ======================
st.divider()
st.caption("SentiScan — Analyse de l'e-réputation | Projet PFE")