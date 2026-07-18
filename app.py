import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Konfigurasi Halaman Dasbor
st.set_page_config(
    page_title="UNIB Smart Academic Predictor",
    page_icon="📊",
    layout="wide"
)

# Judul Utama Dasbor
st.title("📊 UNIB Smart Academic Predictor")
st.markdown("""
**Inovasi Layanan Biro Akademik & Kemahasiswaan Universitas Bengkulu**  
*Early Warning System (EWS)* Berbasis Sains Data untuk Deteksi Dini Risiko Kelulusan Mahasiswa.
""")
st.markdown("---")

# ==========================================
# 1. ENGINE SAINS DATA (PROSES DI BALIK LAYAR)
# ==========================================
@st.cache_data
def load_data_and_model():
    np.random.seed(42)
    n_samples = 200

    ipk_sem2 = np.random.uniform(2.0, 4.0, n_samples)
    kehadiran = np.random.uniform(60, 100, n_samples)
    sks_gagal = np.random.randint(0, 12, n_samples)
    lama_skripsi = np.random.uniform(3, 18, n_samples)

    target = []
    for i in range(n_samples):
        if ipk_sem2[i] < 2.75 or kehadiran[i] < 75 or sks_gagal[i] > 6:
            target.append(1)  # Berisiko / Terlambat
        else:
            target.append(0)  # Aman / Tepat Waktu

    df = pd.DataFrame({
        'IPK_Semester_2': ipk_sem2,
        'Persentase_Kehadiran': kehadiran,
        'SKS_Gagal': sks_gagal,
        'Lama_Skripsi_Bulan': lama_skripsi,
        'Status_Risiko': target
    })

    X = df.drop('Status_Risiko', axis=1)
    y = df['Status_Risiko']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, df, X.columns

model_ews, df_akademik, X_columns = load_data_and_model()

# ==========================================
# 2. ANTARMUKA INPUT (SIDEBAR PANEL)
# ==========================================
st.sidebar.header("📥 Input Data Akademik Mahasiswa")
st.sidebar.write("Simulasi pengecekan status mahasiswa secara individu:")

input_ipk = st.sidebar.slider("IPK Semester 2", 2.0, 4.0, 3.0, 0.01)
input_hadir = st.sidebar.slider("Persentase Kehadiran (%)", 50, 100, 85, 1)
input_sks = st.sidebar.slider("Jumlah SKS Gagal/Mengulang", 0, 12, 2, 1)
input_skripsi = st.sidebar.slider("Durasi Skripsi (Bulan)", 3, 24, 6, 1)

# ==========================================
# 3. PANEL HASIL PREDIKSI (UTAMA)
# ==========================================
st.subheader("🔮 Hasil Deteksi Dini Sistem (Real-Time Prediction)")

data_input = pd.DataFrame([[input_ipk, input_hadir, input_sks, input_skripsi]], columns=X_columns)
prediksi = model_ews.predict(data_input)
probabilitas = model_ews.predict_proba(data_input)[0][1] * 100

col_pred, col_prob = st.columns(2)

with col_pred:
    if prediksi[0] == 1:
        st.error("⚠️ STATUS: MAHASISWA BERISIKO (EARLY WARNING)")
        st.write("Sistem merekomendasikan intervensi segera oleh Dosen Pembimbing Akademik dan Kaprodi.")
    else:
        st.success("✅ STATUS: AMAN / TEPAT WAKTU")
        st.write("Mahasiswa menunjukkan performa akademik yang sesuai dengan standar kelulusan.")

with col_prob:
    st.metric(label="Tingkat Risiko Kelulusan", value=f"{probabilitas:.2f}%")
    st.progress(int(probabilitas))

st.markdown("---")

# ==========================================
# 4. PANEL ANALISIS & VISUALISASI DATA
# ==========================================
st.subheader("📊 Analisis Data Agregat Makro")

tab1, tab2 = st.tabs(["Faktor Berpengaruh", "Sebaran Data Historis"])

with tab1:
    st.markdown("**Indikator Utama Pemicu Risiko Keterlambatan Lulus**")
    
    importances = model_ews.feature_importances_
    features = ['IPK Sem 2', 'Kehadiran (%)', 'SKS Gagal', 'Durasi Skripsi']
    df_imp = pd.DataFrame({'Indikator': features, 'Kepentingan': importances}).sort_values(by='Kepentingan', ascending=False)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x='Kepentingan', y='Indikator', data=df_imp, palette='Blues_r', ax=ax)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")
    st.pyplot(fig)
    st.caption("Grafik ini membantu Biro Akademik menentukan prioritas kebijakan intervensi berdasarkan faktor paling kritis[cite: 1].")

with tab2:
    st.markdown("**Simulasi Data 200 Mahasiswa Terdaftar**")
    st.dataframe(df_akademik.style.format({
        'IPK_Semester_2': '{:.2f}',
        'Persentase_Kehadiran': '{:.1f}%',
        'Lama_Skripsi_Bulan': '{:.1f} bulan'
    }), height=250)