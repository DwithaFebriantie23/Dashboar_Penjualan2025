import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import time

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Dashboard Skripsi - Market Basket Analysis",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# CACHE DATA & PEMROSESAN UTAMA
# ---------------------------------------------------------
@st.cache_data
def load_and_process_data():
    file_path = "Penjualan2025.xlsx"
    df_raw = pd.read_excel(file_path, sheet_name="Rekap 2025", header=1)
    df_selected = df_raw[['TANGGAL', 'CUSTOMER', 'ID TRANSAKSI', 'ITEM', 'QTY', 'TOTAL']].copy()
    
    # Preprocessing
    df_clean = df_selected.dropna(subset=['ID TRANSAKSI', 'ITEM']).drop_duplicates()
    df_clean['ID TRANSAKSI'] = df_clean['ID TRANSAKSI'].astype(str).str.strip()
    df_clean['ITEM'] = df_clean['ITEM'].astype(str).str.strip()
    
    # Transformation Matrix
    basket = (df_clean.groupby(['ID TRANSAKSI', 'ITEM'])['ITEM']
              .count().unstack().reset_index().fillna(0)
              .set_index('ID TRANSAKSI'))
    basket_sets = basket.map(lambda x: 1 if x > 0 else 0).astype(bool)
    
    return df_clean, basket_sets

try:
    df_clean, basket_sets = load_and_process_data()
except Exception as e:
    st.error(f"Gagal memuat file dataset: {e}")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR NAVIGASI
# ---------------------------------------------------------
st.sidebar.title("📌 Navigasi Menu")
menu = st.sidebar.radio("Pilih Menu Utama:", ["🏠 Home / Beranda", "📈 Hasil Analisis & Mining"])

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips Skripsi:** Gunakan menu di atas untuk beralih antara profil penelitian dan hasil eksperimen algoritma.")

# ---------------------------------------------------------
# HALAMAN 1: HOME (RINGKASAN DATA & IDENTITAS)
# ---------------------------------------------------------
if menu == "🏠 Home / Beranda":
    st.title("🎓 Dashboard Penelitian Skripsi")
    st.subheader("Analisis Perbandingan Apriori dan FP-Growth untuk Penentuan Paket Penjualan")
    
    st.markdown("---")
    
    # Kolom Identitas Mahasiswa
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 👤 **Identitas Mahasiswa**")
        st.markdown("""
        * **Nama Lengkap:** Dwitha Febriantie
        * **NIM:** 43230130
        * **Program Studi:** Sistem Informasi
        * **Fakultas / Universitas:** STMIK IKMI Cirebon
        * **Semester:** 8 (Akhir)
        """)
    
    with col2:
        st.markdown("### 📋 **Informasi Skripsi**")
        st.markdown("""
        * **Judul Skripsi:** Analisis Perbandingan Apriori dan FP-Growth untuk Penentuan Paket Penjualan (Studi Kasus: PT. Cakrawala Kemasindo Lestari)
        * **Metodologi:** Knowledge Discovery in Databases (KDD)
        * **Dataset:** Rekapitulasi Penjualan Tahun 2025 (`Penjualan2025.xlsx`)
        """)
        
    st.markdown("---")
    
    # Ringkasan Metrik Dataset
    st.subheader("📊 Ringkasan Dataset Penjualan")
    # Baris Pertama
    m1, m2 = st.columns(2)
    m1.metric("Total Baris Transaksi Bersih", f"{len(df_clean):,} baris")
    m2.metric("Jumlah Transaksi Unik", f"{basket_sets.shape[0]:,} Transaksi")

    st.markdown("---")

    # Baris Kedua
    m3, m4 = st.columns(2)
    m3.metric("Jumlah Produk Unik (Item)", f"{basket_sets.shape[1]:,} Produk")
    m4.metric("Aturan Asosiasi Terbentuk", "18 Aturan")

# ---------------------------------------------------------
# HALAMAN 2: HASIL ANALISIS & MINING
# ---------------------------------------------------------
elif menu == "📈 Hasil Analisis & Mining":
    st.title("📈 Hasil Data Mining & Market Basket Analysis")
    st.write("Eksperimen perbandingan algoritma **Apriori** dan **FP-Growth** berdasarkan dataset penjualan.")
    
    st.markdown("---")
    
    # Sidebar Filter Parameter
    st.sidebar.subheader("⚙️ Parameter Mining")
    min_sup = st.sidebar.slider("Minimum Support", 0.001, 0.050, 0.010, 0.001, format="%.3f")
    min_conf = st.sidebar.slider("Minimum Confidence", 0.05, 0.90, 0.10, 0.05)
    
    # Eksekusi Algoritma
    with st.spinner("Sedang menjalankan algoritma Apriori dan FP-Growth..."):
        # Apriori
        t0 = time.time()
        freq_ap = apriori(basket_sets, min_support=min_sup, use_colnames=True)
        rules_ap = association_rules(freq_ap, metric="confidence", min_threshold=min_conf)
        t_ap = time.time() - t0
        
        # FP-Growth
        t0 = time.time()
        freq_fp = fpgrowth(basket_sets, min_support=min_sup, use_colnames=True)
        rules_fp = association_rules(freq_fp, metric="confidence", min_threshold=min_conf)
        t_fp = time.time() - t0

    # 1. Tabel Perbandingan Kinerja
    st.subheader("⏱️ Tabel Perbandingan Performa Algoritma")
    comp_df = pd.DataFrame({
        'Algoritma': ['Apriori', 'FP-Growth'],
        'Jumlah Frequent Itemsets': [len(freq_ap), len(freq_fp)],
        'Jumlah Association Rules': [len(rules_ap), len(rules_fp)],
        'Waktu Eksekusi (detik)': [round(t_ap, 4), round(t_fp, 4)]
    })
    st.dataframe(comp_df, use_container_width=True)
    
    # Grafik Waktu Eksekusi
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(x='Algoritma', y='Waktu Eksekusi (detik)', data=comp_df, palette="viridis", ax=ax)
    ax.set_title("Perbandingan Kecepatan Waktu Eksekusi")
    st.pyplot(fig)
    
    st.markdown("---")
    
    # 2. Grafik Top 10 Produk Terlaris
    st.subheader("🏆 Top 10 Produk Paling Sering Dibeli")
    top_items = basket_sets.sum().sort_values(ascending=False).head(10)
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(x=top_items.values, y=top_items.index, palette="Blues_r", ax=ax2)
    ax2.set_xlabel("Frekuensi Transaksi")
    ax2.set_ylabel("Nama Produk")
    st.pyplot(fig2)
    
    st.markdown("---")
    
    # 3. Top Aturan Asosiasi (Association Rules)
    st.subheader("📋 Top Aturan Asosiasi Terbaik (Lift Ratio > 1.0)")
    if not rules_fp.empty:
        strong_rules = rules_fp[rules_fp['lift'] > 1.0].sort_values(by='lift', ascending=False).copy()
        strong_rules['Jika Membeli (Antecedent)'] = strong_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        strong_rules['Maka Membeli (Consequent)'] = strong_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        display_cols = ['Jika Membeli (Antecedent)', 'Maka Membeli (Consequent)', 'support', 'confidence', 'lift']
        st.dataframe(strong_rules[display_cols].head(10), use_container_width=True)
    else:
        st.warning("Tidak ditemukan aturan asosiasi yang memenuhi parameter support dan confidence tersebut. Coba turunkan nilainya di sidebar.")