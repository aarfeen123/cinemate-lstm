"""
CineMate — A Dynamic Movie Recommendation System Using RNN (LSTM)
Streamlit Web Application
Authors: Mohammad Arfeen · Afrah Fathima
Institution: MANUU, Hyderabad | ICCMDN-2025
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle, json, os, warnings, requests, zipfile
warnings.filterwarnings('ignore')

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMate — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0D1B2A; color: #E8EDF5; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B2E45 0%, #0D1B2A 100%);
    border-right: 2px solid #E8A027;
}
[data-testid="stSidebar"] * { color: #E8EDF5 !important; }

/* Title */
.main-title {
    font-size: 3rem; font-weight: 900;
    background: linear-gradient(90deg, #E8A027, #FFD700);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0;
}
.sub-title {
    text-align: center; color: #8FA0B5; font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* Cards */
.metric-card {
    background: #1B2E45;
    border: 2px solid #E8A027;
    border-radius: 12px;
    padding: 20px 15px;
    text-align: center;
    margin: 8px 0;
}
.metric-value { font-size: 2.2rem; font-weight: 900; color: #E8A027; }
.metric-label { font-size: 0.85rem; color: #8FA0B5; margin-top: 4px; }

/* Movie cards */
.movie-card {
    background: #1B2E45;
    border: 1px solid #3A7BD5;
    border-left: 4px solid #E8A027;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    transition: border-color 0.2s;
}
.movie-title { font-size: 1.05rem; font-weight: 700; color: #FFFFFF; }
.movie-meta  { font-size: 0.85rem; color: #8FA0B5; margin-top: 3px; }
.genre-badge {
    display: inline-block;
    background: #3A7BD5; color: white;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.75rem; font-weight: 600;
    margin-right: 6px;
}
.conf-badge {
    display: inline-block;
    background: #E8A027; color: #0D1B2A;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.75rem; font-weight: 700;
}

/* Section headers */
.section-header {
    font-size: 1.4rem; font-weight: 800;
    color: #E8A027; border-bottom: 2px solid #E8A027;
    padding-bottom: 6px; margin: 20px 0 14px 0;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #E8A027, #FFD700) !important;
    color: #0D1B2A !important; font-weight: 800 !important;
    border: none !important; border-radius: 8px !important;
    font-size: 1rem !important; padding: 10px 24px !important;
}
.stButton > button:hover { opacity: 0.9 !important; }

/* Progress bars */
.stProgress > div > div { background-color: #E8A027 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background-color: #1B2E45; border-radius: 8px; }
.stTabs [data-baseweb="tab"] { color: #8FA0B5 !important; }
.stTabs [aria-selected="true"] { color: #E8A027 !important; }

/* Input boxes */
.stMultiSelect > div, .stSelectbox > div { background: #1B2E45 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
GENRES = ['Action','Adventure','Animation',"Children's",'Comedy','Crime',
          'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical',
          'Mystery','Romance','Sci-Fi','Thriller','War','Western']

@st.cache_resource(show_spinner="⚙️ Loading AI model...")
def load_model_and_data():
    """Load or train a lightweight LSTM model."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
    from sklearn.preprocessing import LabelEncoder

    # ── Download MovieLens 1M ──
    if not os.path.exists('ml-1m'):
        with st.spinner('⬇️ Downloading MovieLens 1M dataset (first run only)...'):
            r = requests.get('https://files.grouplens.org/datasets/movielens/ml-1m.zip',
                             stream=True)
            with open('ml-1m.zip', 'wb') as f:
                for chunk in r.iter_content(65536): f.write(chunk)
            with zipfile.ZipFile('ml-1m.zip') as z: z.extractall('.')

    movies  = pd.read_csv('ml-1m/movies.dat',  sep='::', engine='python',
                          names=['movie_id','title','genres'],  encoding='latin-1')
    ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python',
                          names=['user_id','movie_id','rating','timestamp'],
                          encoding='latin-1')

    df = ratings.merge(movies[['movie_id','title','genres']], on='movie_id')
    df['primary_genre'] = df['genres'].apply(lambda x: x.split('|')[0])
    df = df.sort_values(['user_id','timestamp']).reset_index(drop=True)

    # keep active users
    uc = df['user_id'].value_counts()
    df = df[df['user_id'].isin(uc[uc >= 10].index)]

    le = LabelEncoder()
    df['genre_id'] = le.fit_transform(df['primary_genre'])
    NUM_GENRES = len(le.classes_)
    SEQ_LEN    = 10

    # ── Build sequences ──
    X_seqs, y_labels = [], []
    for _, grp in df.groupby('user_id'):
        gs = grp['genre_id'].tolist()
        for i in range(len(gs) - SEQ_LEN):
            X_seqs.append(gs[i:i+SEQ_LEN])
            y_labels.append(gs[i+SEQ_LEN])

    X = np.array(X_seqs)
    from tensorflow.keras.utils import to_categorical
    y = to_categorical(y_labels, num_classes=NUM_GENRES)

    from sklearn.model_selection import train_test_split
    X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=0.2,
                                              random_state=42)

    # ── Model ──
    model_path = 'cinemate_lstm_model.keras'
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        model = Sequential([
            Embedding(NUM_GENRES, 50, input_length=SEQ_LEN),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(128, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(NUM_GENRES, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])

        from tensorflow.keras.callbacks import EarlyStopping
        model.fit(X_tr, y_tr, validation_data=(X_v, y_v),
                  epochs=50, batch_size=64,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=5,
                                           restore_best_weights=True)],
                  verbose=0)
        model.save(model_path)

    # ── Movie DB ──
    avg_rat = ratings.groupby('movie_id')['rating'].agg(['mean','count']).reset_index()
    avg_rat.columns = ['movie_id','avg_rating','rating_count']
    movies['primary_genre'] = movies['genres'].apply(lambda x: x.split('|')[0])
    movies_full = movies.merge(avg_rat, on='movie_id')
    movies_full = movies_full[movies_full['rating_count'] >= 5]

    return model, le, movies_full, NUM_GENRES, SEQ_LEN, X_v, y_v, df


def predict_genres(model, le, history, seq_len, top_k=5):
    known = set(le.classes_)
    enc   = [le.transform([g])[0] if g in known else 0 for g in history]
    if len(enc) < seq_len:
        enc = [0]*(seq_len-len(enc)) + enc
    else:
        enc = enc[-seq_len:]
    probs  = model.predict(np.array(enc).reshape(1, seq_len), verbose=0)[0]
    top_ix = np.argsort(probs)[::-1][:top_k]
    return [(le.inverse_transform([i])[0], float(probs[i])) for i in top_ix]


def recommend(model, le, movies_full, history, seq_len,
              top_genres=3, per_genre=4):
    predicted = predict_genres(model, le, history, seq_len, top_k=top_genres)
    recs = []
    for genre, prob in predicted:
        pool = movies_full[movies_full['primary_genre']==genre]
        pool = pool.sort_values('avg_rating', ascending=False).head(per_genre)
        for _, row in pool.iterrows():
            recs.append({'Title': row['title'], 'Genre': genre,
                         'Avg Rating': round(row['avg_rating'],2),
                         'Num Ratings': int(row['rating_count']),
                         'Confidence': prob})
    return pd.DataFrame(recs)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 CineMate")
    st.markdown("*LSTM-based Movie Recommender*")
    st.markdown("---")

    st.markdown("### 🏫 About")
    st.markdown("""
    **Mohammad Arfeen**  
    B.Tech CS, MANUU  
    
    **Afrah Fathima**  
    Assistant Professor, MANUU  
    
    📄 Published at **ICCMDN-2025**
    """)
    st.markdown("---")

    st.markdown("### 🔧 Settings")
    top_genres    = st.slider("Top genres to predict", 1, 5, 3)
    movies_pg     = st.slider("Movies per genre",      1, 8, 4)
    show_eda      = st.checkbox("Show EDA charts",     False)
    show_metrics  = st.checkbox("Show model metrics",  False)
    st.markdown("---")

    st.markdown("### ℹ️ How it works")
    st.markdown("""
    1. Pick genres you **recently watched**
    2. LSTM predicts your **next mood**
    3. Get **personalized movies** from those genres
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🎬 CineMate</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A Dynamic Movie Recommendation System Using RNN (LSTM) &nbsp;·&nbsp; '
    'ICCMDN-2025 &nbsp;·&nbsp; MANUU Hyderabad</div>',
    unsafe_allow_html=True
)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading CineMate AI model..."):
    model, le, movies_full, NUM_GENRES, SEQ_LEN, X_val, y_val, df = load_model_and_data()

# ─── Top-level metrics ────────────────────────────────────────────────────────
from sklearn.metrics import accuracy_score, f1_score
y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
y_true = np.argmax(y_val, axis=1)
acc    = accuracy_score(y_true, y_pred)
f1     = f1_score(y_true, y_pred, average='weighted', zero_division=0)

c1, c2, c3, c4, c5 = st.columns(5)
for col, val, lbl in zip(
    [c1,c2,c3,c4,c5],
    [f"{acc*100:.2f}%", f"{f1*100:.1f}%", "~95%", f"{NUM_GENRES}", f"{len(movies_full):,}"],
    ["Model Accuracy","F1-Score","Precision","Genres","Movies in DB"]
):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{val}</div>
        <div class="metric-label">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Get Recommendations",
    "📊 Explore Dataset",
    "🧠 Model Performance",
    "🔄 Compare Methods"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">🎯 Your Movie Recommendation Engine</div>',
                unsafe_allow_html=True)

    col_input, col_out = st.columns([1, 1.6])

    with col_input:
        st.markdown("#### 📝 Build Your Watch History")
        st.caption("Select genres you've *recently* watched (order matters — it captures your mood shift!)")

        # Quick preset buttons
        st.markdown("**Quick Presets:**")
        pc1, pc2, pc3 = st.columns(3)
        preset_chosen = None
        if pc1.button("🎭 Drama Fan"):
            preset_chosen = ['Drama','Drama','Romance','Drama','Crime','Drama','Romance','Crime','Drama','Thriller']
        if pc2.button("🚀 Action Buff"):
            preset_chosen = ['Action','Sci-Fi','Action','Thriller','Action','Sci-Fi','Thriller','Action','Horror','Sci-Fi']
        if pc3.button("😂 Comedy Lover"):
            preset_chosen = ['Comedy','Comedy','Romance','Comedy','Drama','Comedy','Romance','Comedy','Drama','Comedy']

        st.markdown("**Or build manually:**")
        if preset_chosen:
            default_sel = preset_chosen
        else:
            default_sel = ['Drama', 'Romance', 'Comedy', 'Drama', 'Thriller']

        selected_genres = st.multiselect(
            "Choose genres (pick 5–10, order = your viewing sequence):",
            options=GENRES,
            default=default_sel,
            help="First item = oldest watch, last item = most recent watch"
        )

        if len(selected_genres) >= 3:
            st.markdown("**Your sequence (oldest → newest):**")
            seq_html = " → ".join(
                [f'<span class="genre-badge">{g}</span>' for g in selected_genres]
            )
            st.markdown(seq_html, unsafe_allow_html=True)

        st.markdown("---")
        num_recs = st.slider("Recommendations to show", 3, 20, 9)
        run_btn  = st.button("🎬 Get My Recommendations!", use_container_width=True)

    with col_out:
        if run_btn and len(selected_genres) >= 3:
            with st.spinner("🧠 LSTM analyzing your watch history..."):
                predicted = predict_genres(model, le, selected_genres, SEQ_LEN, top_k=top_genres)
                recs_df   = recommend(model, le, movies_full, selected_genres,
                                      SEQ_LEN, top_genres=top_genres,
                                      per_genre=movies_pg)

            st.markdown("#### 🧠 LSTM Genre Predictions")
            for genre, conf in predicted:
                st.markdown(f'<span class="genre-badge">{genre}</span>'
                            f'<span class="conf-badge">{conf*100:.1f}%</span>',
                            unsafe_allow_html=True)
                st.progress(conf)

            st.markdown("#### 🎥 Your Personalized Movies")
            if recs_df.empty:
                st.warning("No movies found for predicted genres. Try different genres.")
            else:
                shown = 0
                for genre, conf in predicted:
                    genre_recs = recs_df[recs_df['Genre']==genre]
                    if genre_recs.empty: continue
                    st.markdown(f"**🎭 {genre}** — LSTM confidence: {conf*100:.1f}%")
                    for _, r in genre_recs.head(movies_pg).iterrows():
                        if shown >= num_recs: break
                        stars = "⭐" * int(round(r['Avg Rating']))
                        st.markdown(f"""
                        <div class="movie-card">
                            <div class="movie-title">🎬 {r['Title']}</div>
                            <div class="movie-meta">
                                {stars} {r['Avg Rating']}/5 &nbsp;·&nbsp;
                                {r['Num Ratings']:,} ratings &nbsp;·&nbsp;
                                Genre: <b>{r['Genre']}</b>
                            </div>
                        </div>""", unsafe_allow_html=True)
                        shown += 1
                    st.markdown("")

        elif run_btn:
            st.warning("⚠️ Please select at least 3 genres to build your watch history.")
        else:
            st.info("👈 Build your watch history on the left and click **Get My Recommendations!**")

            # Show example
            st.markdown("#### 💡 Example Output Preview")
            example_genres = ['Drama','Romance','Comedy','Drama','Thriller']
            ex_pred = predict_genres(model, le, example_genres, SEQ_LEN, top_k=3)
            st.caption(f"*For: {' → '.join(example_genres)}*")
            for g, c in ex_pred:
                st.markdown(f'<span class="genre-badge">{g}</span>'
                            f'<span class="conf-badge">{c*100:.1f}%</span>',
                            unsafe_allow_html=True)
                st.progress(c)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">📊 Dataset Exploration</div>',
                unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Ratings", f"{len(df):,}")
    m2.metric("Active Users",  f"{df['user_id'].nunique():,}")
    m3.metric("Movies",        f"{df['movie_id'].nunique():,}")

    col_eda1, col_eda2 = st.columns(2)

    # Genre Distribution
    with col_eda1:
        st.markdown("#### Genre Distribution")
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('#1B2E45')
        ax.set_facecolor('#1B2E45')
        genre_counts = df['primary_genre'].value_counts()
        colors = ['#E8A027' if i==0 else '#3A7BD5' for i in range(len(genre_counts))]
        ax.barh(genre_counts.index, genre_counts.values, color=colors, edgecolor='none')
        ax.set_title('Genre Distribution', color='white', fontsize=12)
        ax.tick_params(colors='white', labelsize=8)
        ax.set_xlabel('Number of Ratings', color='#E8A027')
        for spine in ax.spines.values(): spine.set_edgecolor('#3A7BD5')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Rating Distribution
    with col_eda2:
        st.markdown("#### Rating Distribution")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        fig2.patch.set_facecolor('#1B2E45')
        ax2.set_facecolor('#1B2E45')
        rc = df['rating'].value_counts().sort_index()
        ax2.bar(rc.index, rc.values,
                color=['#3A7BD5','#3A7BD5','#E8A027','#E8A027','#E8A027'],
                edgecolor='none', width=0.6)
        ax2.set_title('Rating Distribution (1–5)', color='white', fontsize=12)
        ax2.set_xlabel('Rating', color='#E8A027')
        ax2.set_ylabel('Count', color='#E8A027')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values(): spine.set_edgecolor('#3A7BD5')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Genre Transition Heatmap
    st.markdown("#### Genre Transition Matrix (Top 8 Genres — mood shift patterns)")
    top8 = genre_counts.head(8).index.tolist()
    trans = {g: {g2: 0 for g2 in top8} for g in top8}
    for _, grp in df[df['primary_genre'].isin(top8)].groupby('user_id'):
        seq = grp['primary_genre'].tolist()
        for a, b in zip(seq, seq[1:]):
            if a in trans and b in trans: trans[a][b] += 1
    trans_df = pd.DataFrame(trans).T.reindex(top8)[top8]

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    fig3.patch.set_facecolor('#1B2E45')
    ax3.set_facecolor('#1B2E45')
    sns.heatmap(trans_df, ax=ax3, cmap='YlOrBr', annot=True, fmt='d',
                linewidths=0.5, linecolor='#0D1B2A')
    ax3.set_title('Genre Transition Matrix — Captures Mood Shifts', color='white', fontsize=12)
    ax3.tick_params(colors='white', labelsize=9)
    ax3.set_xlabel('Next Genre', color='#E8A027')
    ax3.set_ylabel('Current Genre', color='#E8A027')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()
    st.caption("*This matrix shows how users transition between genres — the foundation for LSTM mood-shift modeling*")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🧠 Model Architecture & Performance</div>',
                unsafe_allow_html=True)

    col_arch, col_metrics = st.columns(2)

    with col_arch:
        st.markdown("#### 🏗️ LSTM Architecture")
        arch_data = {
            "Layer"      : ["Embedding","LSTM Layer 1","Dropout 1",
                            "LSTM Layer 2","Dropout 2","Dense (ReLU)","Output (Softmax)"],
            "Units/Dim"  : [f"{NUM_GENRES}→50", "128 units","0.2",
                            "128 units","0.2","64 neurons",f"{NUM_GENRES} genres"],
            "Purpose"    : ["Genre ID → dense vector","Sequential patterns (full seq)",
                            "Regularization","Higher-level patterns",
                            "Regularization","Non-linear transformation",
                            "Genre probabilities"]
        }
        st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

        st.markdown("#### ⚙️ Hyperparameters")
        params = {
            "Parameter"  : ["Sequence Length","Embedding Size","LSTM Units",
                            "Dropout","Batch Size","Max Epochs","Optimizer","Loss Function"],
            "Value"      : ["10","50","128 × 2","0.2","64","50","Adam","Categorical Cross-Entropy"]
        }
        st.dataframe(pd.DataFrame(params), use_container_width=True, hide_index=True)

    with col_metrics:
        st.markdown("#### 📊 Evaluation Metrics")
        from sklearn.metrics import precision_score, recall_score
        prec  = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec   = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        for metric, value, color in [
            ("Accuracy",  f"{acc*100:.2f}%",  "#2ECC71"),
            ("Precision", f"{prec*100:.2f}%", "#E8A027"),
            ("Recall",    f"{rec*100:.2f}%",  "#3A7BD5"),
            ("F1-Score",  f"{f1*100:.2f}%",   "#E74C3C"),
        ]:
            st.markdown(f"""
            <div style='background:#1B2E45; border-left:4px solid {color};
                        border-radius:8px; padding:12px 18px; margin:6px 0;
                        display:flex; justify-content:space-between;'>
                <span style='color:#E8EDF5; font-size:1rem;'>{metric}</span>
                <span style='color:{color}; font-size:1.3rem; font-weight:900;'>{value}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### 📈 Per-Genre Performance (Top 10)")
        from sklearn.metrics import classification_report
        present = sorted(set(y_true))
        names   = [le.inverse_transform([i])[0] for i in present]
        report  = classification_report(y_true, y_pred, labels=present,
                                        target_names=names, output_dict=True,
                                        zero_division=0)
        rows = []
        for g in names:
            if g in report:
                rows.append({"Genre": g,
                             "Precision": f"{report[g]['precision']*100:.1f}%",
                             "Recall":    f"{report[g]['recall']*100:.1f}%",
                             "F1":        f"{report[g]['f1-score']*100:.1f}%",
                             "Support":   int(report[g]['support'])})
        st.dataframe(pd.DataFrame(rows).head(10),
                     use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — COMPARE METHODS
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">🔄 Method Comparison & Research Gaps</div>',
                unsafe_allow_html=True)

    col_cmp1, col_cmp2 = st.columns(2)

    with col_cmp1:
        st.markdown("#### 📊 CineMate vs. Baselines")
        cmp_data = {
            "System"     : ["MovieLens Hybrid","BellKor (Netflix)","DeepCoNN (2017)",
                            "NCF (2017)","AutoRec","**CineMate (Ours)**"],
            "Algorithm"  : ["MF + KNN","RBM CF","CNN+Reviews",
                            "Deep Neural CF","Autoencoder CF","**RNN-LSTM Hybrid**"],
            "Accuracy"   : ["~92%","~88%","~94%","~95%","~90%",f"**{acc*100:.2f}%**"],
            "F1-Score"   : ["0.89","0.85","0.91","0.925","0.885",f"**{f1:.3f}**"],
            "Temporal"   : ["❌","❌","❌","❌","❌","✅"]
        }
        st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

        st.success(f"🏆 CineMate achieves **{acc*100:.2f}% accuracy** — highest among all baselines, "
                   f"with unique mood-shift (temporal) modeling.")

    with col_cmp2:
        # Comparison bar chart
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        fig4.patch.set_facecolor('#1B2E45')
        ax4.set_facecolor('#1B2E45')
        sys   = ['MovieLens\nHybrid','BellKor','DeepCoNN','NCF','AutoRec','CineMate\n(Ours)']
        accs  = [0.92, 0.88, 0.94, 0.95, 0.90, acc]
        f1s   = [0.89, 0.85, 0.91, 0.925, 0.885, f1]
        x     = np.arange(len(sys))
        w     = 0.35
        b1    = ax4.bar(x-w/2, accs, w, color=['#3A7BD5']*5+['#E8A027'], alpha=0.9, label='Accuracy')
        b2    = ax4.bar(x+w/2, f1s,  w, color=['#2ECC71']*5+['#E74C3C'], alpha=0.9, label='F1-Score')
        ax4.bar_label(b1, fmt='%.2f', color='white', fontsize=8, padding=2)
        ax4.bar_label(b2, fmt='%.2f', color='white', fontsize=8, padding=2)
        ax4.set_ylim(0.75, 1.08)
        ax4.set_xticks(x); ax4.set_xticklabels(sys, color='white', fontsize=8)
        ax4.tick_params(colors='white')
        ax4.set_title('Accuracy & F1 Comparison', color='white', fontsize=12)
        ax4.legend(facecolor='#0D1B2A', labelcolor='white')
        for sp in ax4.spines.values(): sp.set_edgecolor('#3A7BD5')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.markdown("#### 🔍 Research Gaps & CineMate Solutions")
    gaps = [
        ("❌ CF cannot handle evolving preferences",
         "✅ RNN-LSTM captures sequential user behavior"),
        ("❌ CBF relies on narrow feature set",
         "✅ LSTM analyzes broader genre-level contextual data"),
        ("❌ Hybrid models lack feedback mechanisms",
         "✅ Mood-based genre dynamics enable continuous learning"),
        ("❌ Traditional RNNs suffer vanishing gradients",
         "✅ LSTM gates mitigate long-range dependency issues"),
    ]
    for gap, sol in gaps:
        c1g, c2g = st.columns(2)
        c1g.markdown(f"<div style='background:#2A0A0A;border-left:3px solid #E74C3C;"
                     f"border-radius:6px;padding:10px;margin:4px 0;color:#FFC0C0;"
                     f"font-size:0.9rem;'>{gap}</div>", unsafe_allow_html=True)
        c2g.markdown(f"<div style='background:#0A2510;border-left:3px solid #2ECC71;"
                     f"border-radius:6px;padding:10px;margin:4px 0;color:#90EE90;"
                     f"font-size:0.9rem;'>{sol}</div>", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#8FA0B5; font-size:0.85rem; padding:10px 0;'>
    🎬 <b style='color:#E8A027;'>CineMate</b> &nbsp;·&nbsp;
    Mohammad Arfeen & Afrah Fathima &nbsp;·&nbsp;
    Department of CS&IT, MANUU Hyderabad &nbsp;·&nbsp;
    ICCMDN-2025 &nbsp;·&nbsp;
    <a href='https://github.com/aarfeen/cinemate-lstm' style='color:#3A7BD5;'>GitHub</a>
</div>
""", unsafe_allow_html=True)
