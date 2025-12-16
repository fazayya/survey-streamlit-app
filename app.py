import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Survey Data Analysis App",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS for Bigger Font and Pink Background
# --------------------------------------------------
st.markdown("""
<style>
.big-font { font-size:18px !important; font-weight: bold; }
body {
    background-color: #c2185b;  /* Pink tua untuk background utama */
    color: #000000;  /* Font hitam */
}
[data-testid="stSidebar"] {
    background-color: #ff69b4;  /* Hot Pink untuk sidebar */
    color: #000000;  /* Font hitam */
}
[data-testid="stHeader"] {
    background-color: #c2185b;  /* Pink tua untuk header */
    color: #000000;  /* Font hitam */
}
[data-testid="stAppViewContainer"] {
    background-color: #c2185b;  /* Pink tua untuk konten utama */
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "language" not in st.session_state:
    st.session_state.language = "EN"

# --------------------------------------------------
# Language Dictionary (Tambah deskripsi singkat)
# --------------------------------------------------
LANG = {
    "EN": {
        "title": "Survey Data Analysis App",
        "subtitle": "Upload your survey dataset and generate comprehensive statistical insights.",
        "description": "This app allows you to analyze survey data with descriptive statistics, frequency tables, visualizations, and correlation analysis. Perfect for researchers and students exploring various trends and insights.",
        "upload": "Upload your survey dataset (Excel or CSV)",
        "preview": "Preview of Dataset",
        "dataset_size": "Dataset size",
        "desc_stats": "Descriptive Statistics (Each Item + Composite Scores)",
        "freq_table": "Frequency & Percentage Tables",
        "viz": "Visualizations (Histogram & Boxplot)",
        "corr": "Association Analysis (Correlation)",
        "interpretation": "Interpretation",
        "variable_x": "Variable X – Level of Fear of Missing Out (FOMO)",
        "variable_y": "Variable Y – Gen Z Mental Health",
        "mean": "Mean",
        "median": "Median",
        "std": "Standard Deviation",
        "min": "Minimum",
        "max": "Maximum",
        "composite": "Composite Score",
        "no_numeric": "No numeric variables found in the dataset.",
        "member_title": "Team Members",
        "select_corr": "Select correlation method",
        "corr_coeff": "Correlation Coefficient (r)",
        "p_value": "p-value",
        "direction": "Direction",
        "strength": "Strength",
        "positive": "Positive",
        "negative": "Negative",
        "very_weak": "Very Weak",
        "weak": "Weak",
        "moderate": "Moderate",
        "strong": "Strong",
        "corr_interpret_en": "The {method} correlation shows a {strength} {direction} relationship between Variable X and Variable Y (r = {r:.3f}, p-value = {p:.4f}).",
        "corr_interpret_id": "Korelasi {method} menunjukkan hubungan {direction} dengan kekuatan {strength} antara Variabel X dan Variabel Y (r = {r:.3f}, p-value = {p:.4f})."
    },
    "ID": {
        "title": "Aplikasi Analisis Data Survei",
        "subtitle": "Unggah data survei dan hasilkan analisis statistik yang komprehensif.",
        "description": "Aplikasi ini memungkinkan Anda menganalisis data survei dengan statistik deskriptif, tabel frekuensi, visualisasi, dan analisis korelasi. Cocok untuk peneliti dan mahasiswa yang mengeksplorasi berbagai tren dan wawasan.",
        "upload": "Unggah dataset survei (Excel atau CSV)",
        "preview": "Pratinjau Dataset",
        "dataset_size": "Ukuran dataset",
        "desc_stats": "Statistik Deskriptif (Setiap Item + Skor Komposit)",
        "freq_table": "Tabel Frekuensi & Persentase",
        "viz": "Visualisasi (Histogram & Boxplot)",
        "corr": "Analisis Asosiasi (Korelasi)",
        "interpretation": "Interpretasi",
        "variable_x": "Variabel X – Tingkat Fear of Missing Out (FOMO)",
        "variable_y": "Variabel Y – Kesehatan Mental Gen Z",
        "mean": "Rata-rata",
        "median": "Median",
        "std": "Simpangan Baku",
        "min": "Minimum",
        "max": "Maksimum",
        "composite": "Skor Komposit",
        "no_numeric": "Tidak ditemukan variabel numerik.",
        "member_title": "Anggota Tim",
        "select_corr": "Pilih metode korelasi",
        "corr_coeff": "Koefisien Korelasi (r)",
        "p_value": "p-value",
        "direction": "Arah",
        "strength": "Kekuatan",
        "positive": "Positif",
        "negative": "Negatif",
        "very_weak": "Sangat Lemah",
        "weak": "Lemah",
        "moderate": "Sedang",
        "strong": "Kuat",
        "corr_interpret_en": "The {method} correlation shows a {strength} {direction} relationship between Variable X and Variable Y (r = {r:.3f}, p-value = {p:.4f}).",
        "corr_interpret_id": "Korelasi {method} menunjukkan hubungan {direction} dengan kekuatan {strength} antara Variabel X dan Variabel Y (r = {r:.3f}, p-value = {p:.4f})."
    }
}

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    page = st.radio("Menu", ["App", "Member"], label_visibility="collapsed")
    st.divider()
    st.session_state.language = st.selectbox(
        "Language / Bahasa",
        ["EN", "ID"],
        index=0 if st.session_state.language == "EN" else 1
    )

lang = st.session_state.language

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title(LANG[lang]["title"])
st.caption(LANG[lang]["subtitle"])
st.write(LANG[lang]["description"])  # Tambah deskripsi singkat di bawah subtitle
st.markdown("---")

# ==================================================
# PAGE: APP
# ==================================================
if page == "App":

    uploaded_file = st.file_uploader(
        LANG[lang]["upload"],
        type=["xlsx", "csv"]
    )

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)

    if st.session_state.df is not None:
        df = st.session_state.df

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning(LANG[lang]["no_numeric"])
            st.stop()

        half = len(numeric_cols) // 2
        x_vars = numeric_cols[:half]
        y_vars = numeric_cols[half:]

        # Composite Scores
        df["Composite_X"] = df[x_vars].mean(axis=1)
        df["Composite_Y"] = df[y_vars].mean(axis=1)

        # ---------------- Preview ----------------
        st.subheader(LANG[lang]["preview"])
        st.dataframe(df, use_container_width=True)
        st.write(f"{LANG[lang]['dataset_size']}: **{df.shape[0]} × {df.shape[1]}**")
        st.markdown("---")

        # ---------------- Descriptive Statistics ----------------
        st.subheader(LANG[lang]["desc_stats"])

        desc = pd.DataFrame({
            LANG[lang]["mean"]: df[numeric_cols].mean(),
            LANG[lang]["median"]: df[numeric_cols].median(),
            LANG[lang]["std"]: df[numeric_cols].std(),
            LANG[lang]["min"]: df[numeric_cols].min(),
            LANG[lang]["max"]: df[numeric_cols].max()
        })

        st.dataframe(desc, use_container_width=True)

        st.markdown(f"### **{LANG[lang]['composite']}**")
        st.dataframe(df[["Composite_X", "Composite_Y"]].describe().T)

        st.markdown("---")

        # ---------------- Frequency Tables ----------------
        st.subheader(LANG[lang]["freq_table"])

        st.markdown(f'<p class="big-font">{LANG[lang]["variable_x"]}</p>', unsafe_allow_html=True)
        for i, col in enumerate(x_vars, start=1):
            # Ekstrak nama kolom tanpa angka di depan (jika ada format "angka. teks")
            display_name = col.split('. ', 1)[1] if '. ' in col and col[0].isdigit() else col
            st.markdown(f'<p class="big-font">{i}. {display_name}</p>', unsafe_allow_html=True)
            freq = df[col].value_counts().sort_index()
            percent = (freq / freq.sum()) * 100
            st.dataframe(pd.DataFrame({
                "Frequency": freq,
                "Percentage (%)": percent.round(2)
            }), use_container_width=True)
            st.write("")  # Tambah spasi

        st.markdown(f'<p class="big-font">{LANG[lang]["variable_y"]}</p>', unsafe_allow_html=True)
        for i, col in enumerate(y_vars, start=1):
            # Ekstrak nama kolom tanpa angka di depan
            display_name = col.split('. ', 1)[1] if '. ' in col and col[0].isdigit() else col
            st.markdown(f'<p class="big-font">{i}. {display_name}</p>', unsafe_allow_html=True)
            freq = df[col].value_counts().sort_index()
            percent = (freq / freq.sum()) * 100
            st.dataframe(pd.DataFrame({
                "Frequency": freq,
                "Percentage (%)": percent.round(2)
            }), use_container_width=True)
            st.write("")  # Tambah spasi

        st.markdown("---")

        # ---------------- Visualizations ----------------
        st.subheader(LANG[lang]["viz"])

        st.markdown(f'<p class="big-font">{LANG[lang]["variable_x"]}</p>', unsafe_allow_html=True)
        for i, col in enumerate(x_vars, start=1):
            # Ekstrak nama kolom tanpa angka di depan
            display_name = col.split('. ', 1)[1] if '. ' in col and col[0].isdigit() else col
            st.markdown(f'<p class="big-font">{i}. {display_name}</p>', unsafe_allow_html=True)
            # Gunakan columns untuk side-by-side
            col_hist, col_box = st.columns(2)
            with col_hist:
                fig, ax = plt.subplots()
                ax.hist(df[col].dropna(), bins=10)
                ax.set_title("Histogram")
                st.pyplot(fig)
            with col_box:
                fig, ax = plt.subplots()
                ax.boxplot(df[col].dropna())
                ax.set_title("Boxplot")
                st.pyplot(fig)
            st.write("")  # Tambah spasi

        st.markdown(f'<p class="big-font">{LANG[lang]["variable_y"]}</p>', unsafe_allow_html=True)
        for i, col in enumerate(y_vars, start=1):
            # Ekstrak nama kolom tanpa angka di depan
            display_name = col.split('. ', 1)[1] if '. ' in col and col[0].isdigit() else col
            st.markdown(f'<p class="big-font">{i}. {display_name}</p>', unsafe_allow_html=True)
            # Gunakan columns untuk side-by-side
            col_hist, col_box = st.columns(2)
            with col_hist:
                fig, ax = plt.subplots()
                ax.hist(df[col].dropna(), bins=10)
                ax.set_title("Histogram")
                st.pyplot(fig)
            with col_box:
                fig, ax = plt.subplots()
                ax.boxplot(df[col].dropna())
                ax.set_title("Boxplot")
                st.pyplot(fig)
            st.write("")  # Tambah spasi

        st.markdown("---")

        # ---------------- Association Analysis ----------------
        st.subheader(LANG[lang]["corr"])

        method = st.selectbox(
            LANG[lang]["select_corr"],
            ["Pearson", "Spearman"]
        )

        if method == "Pearson":
            corr_value, p_value = pearsonr(df["Composite_X"], df["Composite_Y"])
        else:
            corr_value, p_value = spearmanr(df["Composite_X"], df["Composite_Y"])

        fig, ax = plt.subplots()
        ax.scatter(df["Composite_X"], df["Composite_Y"])
        ax.set_xlabel("Composite X")
        ax.set_ylabel("Composite Y")
        st.pyplot(fig)

        st.markdown("### **Correlation Output**")
        st.markdown(f"**{LANG[lang]['corr_coeff']}:** {corr_value:.3f}")
        st.markdown(f"**{LANG[lang]['p_value']}:** {p_value:.4f}")

        direction = LANG[lang]["positive"] if corr_value > 0 else LANG[lang]["negative"]
        abs_corr = abs(corr_value)

        if abs_corr < 0.2:
            strength = LANG[lang]["very_weak"]
        elif abs_corr < 0.4:
            strength = LANG[lang]["weak"]
        elif abs_corr < 0.6:
            strength = LANG[lang]["moderate"]
        else:
            strength = LANG[lang]["strong"]

        st.markdown(f"**{LANG[lang]['direction']}:** {direction}")
        st.markdown(f"**{LANG[lang]['strength']}:** {strength}")

        st.subheader(LANG[lang]["interpretation"])

        if lang == "EN":
            st.write(LANG[lang]["corr_interpret_en"].format(method=method, strength=strength.lower(), direction=direction.lower(), r=corr_value, p=p_value))
        else:
            st.write(LANG[lang]["corr_interpret_id"].format(method=method, strength=strength.lower(), direction=direction.lower(), r=corr_value, p=p_value))

# ==================================================
# PAGE: MEMBER
# ==================================================
else:
    st.subheader(LANG[lang]["member_title"])
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.image("assets/fazayya.jpg", width=200)
        st.markdown("### **Fazayya Syauqii Wardhani**")
        if lang == "EN":
            st.markdown("**Role:** Project Leader & Data Analyst")
            st.markdown(
                "**Responsibilities:**\n"
                "- Coordinating team workflow\n"
                "- Designing statistical analysis structure\n"
                "- Interpreting results and conclusions"
            )
        else:
            st.markdown("**Peran:** Ketua Proyek & Analis Data")
            st.markdown(
                "**Tugas:**\n"
                "- Mengkoordinasikan alur kerja tim\n"
                "- Menyusun struktur analisis statistik\n"
                "- Menginterpretasikan hasil dan kesimpulan"
            )

    with col2:
        st.image("assets/lulu.jpg", width=200)
        st.markdown("### **Lulu Zenover**")
        if lang == "EN":
            st.markdown("**Role:** Data Collection & Instrument Design")
            st.markdown(
                "**Responsibilities:**\n"
                "- Designing survey instruments\n"
                "- Managing data collection process\n"
                "- Validating raw data"
            )
        else:
            st.markdown("**Peran:** Pengumpulan Data & Desain Instrumen")
            st.markdown(
                "**Tugas:**\n"
                "- Menyusun instrumen survei\n"
                "- Mengelola proses pengumpulan data\n"
                "- Validasi data mentah"
            )

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.image("assets/nazwa.jpg", width=200)
        st.markdown("### **Nazwa Safa Davina**")
        if lang == "EN":
            st.markdown("**Role:** Statistical Analysis Support")
            st.markdown(
                "**Responsibilities:**\n"
                "- Assisting descriptive and correlation analysis\n"
                "- Reviewing statistical outputs\n"
                "- Supporting interpretation section"
            )
        else:
            st.markdown("**Peran:** Pendukung Analisis Statistik")
            st.markdown(
                "**Tugas:**\n"
                "- Membantu analisis deskriptif dan korelasi\n"
                "- Meninjau hasil statistik\n"
                "- Mendukung bagian interpretasi"
            )

    with col4:
        st.image("assets/raisyah.jpg", width=200)
        st.markdown("### **Raisyah Aditya Sufah**")
        if lang == "EN":
            st.markdown("**Role:** Application Developer & UI Support")
            st.markdown(
                "**Responsibilities:**\n"
                "- Developing Streamlit application\n"
                "- Managing user interface layout\n"
                "- Ensuring application usability"
            )
        else:
            st.markdown("**Peran:** Pengembang Aplikasi & UI")
            st.markdown(
                "**Tugas:**\n"
                "- Mengembangkan aplikasi Streamlit\n"
                "- Mengatur tata letak antarmuka\n"
                "- Memastikan aplikasi mudah digunakan"
            )
