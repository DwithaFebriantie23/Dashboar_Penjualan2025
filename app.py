import streamlit as st
import pandas as pd
import numpy as np
import string
import time
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# Pengaturan Tema Visual Grafik
sns.set_theme(style="whitegrid")

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Dashboard PT. Cakrawala Kemasindo Lestari",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# CACHE DATA & PEMROSESAN KDD (DENGAN STANDARISASI & MAPPING)
# ---------------------------------------------------------
@st.cache_data
def load_and_process_data():
    file_path = "Penjualan2025.xlsx"
    df_raw = pd.read_excel(file_path, sheet_name="Rekap 2025", header=1)
    
    # Data Cleaning & Selection
    df_clean = df_raw.dropna(subset=['ITEM', 'ID TRANSAKSI']).copy()
    df_clean['ID TRANSAKSI'] = df_clean['ID TRANSAKSI'].astype(str).str.strip()
    df_clean['ITEM'] = df_clean['ITEM'].astype(str).str.strip()
    
    # --- STANDARISASI NAMA CUSTOMER (MENGGABUNGKAN PENULISAN MIRIP) ---
    df_clean['CUSTOMER'] = df_clean['CUSTOMER'].astype(str).str.strip()
    df_clean['CUSTOMER'] = df_clean['CUSTOMER'].replace({
        'BP. Acuy': 'Bp Acuy',
        'Bp. Acuy': 'Bp Acuy'
    }, regex=True)
    
    # --- PEMBERSIHAN & KALKULASI OTOMATIS NILAI TOTAL 0 ---
    df_clean['QTY'] = pd.to_numeric(df_clean['QTY'], errors='coerce').fillna(0)
    df_clean['HARGA'] = pd.to_numeric(df_clean['HARGA'], errors='coerce').fillna(0)
    df_clean['TOTAL'] = pd.to_numeric(df_clean['TOTAL'], errors='coerce').fillna(0)
    
    # Jika TOTAL 0 tapi HARGA & QTY ada, hitung otomatis QTY * HARGA
    mask_zero = df_clean['TOTAL'] == 0
    df_clean.loc[mask_zero & (df_clean['HARGA'] > 0), 'TOTAL'] = (
        df_clean['QTY'] * df_clean['HARGA']
    )
    
    # Ekstraksi Tanggal & Bulan Kronologis (Januari - Desember)
    df_clean['TANGGAL'] = pd.to_datetime(df_clean['TANGGAL'], errors='coerce')
    df_clean['MONTH_NUM'] = df_clean['TANGGAL'].dt.month
    df_clean['BULAN'] = df_clean['TANGGAL'].dt.strftime('%B %Y')
    
    # Data Transformation (Pengelompokan Ukuran Item ke Abjad A-Z, AA, dst.)
    unique_items = sorted(df_clean['ITEM'].unique())
    
    def get_excel_style_label(n):
        label = ''
        while n >= 0:
            label = string.ascii_uppercase[n % 26] + label
            n = n // 26 - 1
        return label

    item_mapping = {item: get_excel_style_label(i) for i, item in enumerate(unique_items)}
    reverse_mapping = {v: k for k, v in item_mapping.items()}
    df_clean['ITEM_KODE'] = df_clean['ITEM'].map(item_mapping)
    
    # Basket Matrix (One-Hot Encoding berdasarkan Kode Abjad)
    basket = (df_clean.groupby(['ID TRANSAKSI', 'ITEM_KODE'])['ITEM_KODE']
              .count().unstack().reset_index().fillna(0)
              .set_index('ID TRANSAKSI'))
    basket_sets = basket.map(lambda x: True if x > 0 else False)
    
    return df_clean, item_mapping, reverse_mapping, basket_sets

try:
    df_clean, item_mapping, reverse_mapping, basket_sets = load_and_process_data()
except Exception as e:
    st.error(f"Gagal memuat file dataset `Penjualan2025.xlsx`: {e}")
    st.info("Pastikan file `Penjualan2025.xlsx` berada di direktori folder yang sama dengan `app.py`.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR: NAVIGASI UTAMA
# ---------------------------------------------------------
st.sidebar.title("📌 Navigasi Menu")
menu_pilihan = st.sidebar.radio(
    "Pilih Halaman:", 
    [
        "🏠 Home / Beranda", 
        "📈 Hasil Analisis Data Mining", 
        "⚖️ Pengujian 5 Iterasi",
        "📊 Grafik & Filter Penjualan", 
        "⚙️ Parameter Model"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:** Pilih menu di atas untuk berpindah halaman.")

# Inisialisasi Session State Parameter Default
if "min_supp" not in st.session_state:
    st.session_state.min_supp = 0.005
if "min_conf" not in st.session_state:
    st.session_state.min_conf = 0.10
if "algo_choice" not in st.session_state:
    st.session_state.algo_choice = "FP-Growth"

# =========================================================
# MENU 1: HOME / BERANDA
# =========================================================
if menu_pilihan == "🏠 Home / Beranda":
    st.title("📊 Dashboard PT. Cakrawala Kemasindo Lestari")
    st.markdown("Analisis Pola Pembelian Produk (Ukuran Item Berbasis Abjad) Menggunakan Algoritma **Apriori** & **FP-Growth** (KDD Framework).")
    st.markdown("---")
    
    st.subheader("🎓 Dashboard Penelitian Skripsi")
    st.markdown("**Judul Skripsi:** Analisis Perbandingan Apriori dan FP-Growth untuk Penentuan Paket Penjualan (Studi Kasus: PT. Cakrawala Kemasindo Lestari)")
    
    st.markdown("---")
    
    st.markdown("### 📋 **Informasi Metodologi**")
    st.markdown("""
    * **Metodologi:** Knowledge Discovery in Databases (KDD)
    * **Dataset:** Rekapitulasi Penjualan Tahun 2025 (`Penjualan2025.xlsx`)
    * **Preprocessing:** Pembersihan data kosong, standarisasi nama pelanggan, 5 kali Pengujian Apriori dan FP-Growth serta Pengelompokan ukuran item fisik ke kode abjad ($A, B, \dots, Z, AA, \dots$).
    """)
        
    st.markdown("---")
    
    st.subheader("📊 Ringkasan Dataset PT. Cakrawala Kemasindo Lestari")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Baris Bersih", f"{len(df_clean):,} Baris")
    m2.metric("Jumlah Transaksi Unik", f"{basket_sets.shape[0]:,} Transaksi")
    m3.metric("Jumlah Ukuran Item Unik", f"{len(item_mapping):,} Ukuran")

# =========================================================
# MENU 2: HASIL ANALISIS DATA MINING
# =========================================================
elif menu_pilihan == "📈 Hasil Analisis Data Mining":
    st.title("📈 Hasil Analisis Data Mining")
    st.markdown("Eksperimen pembentukan aturan asosiasi dan rekomendasi paket penjualan (*bundling*).")
    st.markdown("---")

    min_supp = st.session_state.min_supp
    min_conf = st.session_state.min_conf
    algo_choice = st.session_state.algo_choice

    # Eksekusi Single Run Algoritma
    with st.spinner("Sedang memproses algoritma..."):
        t0 = time.perf_counter()
        fi_ap = apriori(basket_sets, min_support=min_supp, use_colnames=True)
        rules_ap = association_rules(fi_ap, metric="confidence", min_threshold=min_conf) if not fi_ap.empty else pd.DataFrame()
        t_ap = time.perf_counter() - t0
        
        t0 = time.perf_counter()
        fi_fp = fpgrowth(basket_sets, min_support=min_supp, use_colnames=True)
        rules_fp = association_rules(fi_fp, metric="confidence", min_threshold=min_conf) if not fi_fp.empty else pd.DataFrame()
        t_fp = time.perf_counter() - t0

    rules = rules_ap if algo_choice == "Apriori" else rules_fp
    fi_active = fi_ap if algo_choice == "Apriori" else fi_fp

    if not rules.empty:
        rules['antecedents_label'] = rules['antecedents'].apply(lambda x: ", ".join(list(x)))
        rules['consequents_label'] = rules['consequents'].apply(lambda x: ", ".join(list(x)))
        
        rules['antecedents_full'] = rules['antecedents'].apply(lambda x: ", ".join([f"{code} ({reverse_mapping[code]})" for code in x]))
        rules['consequents_full'] = rules['consequents'].apply(lambda x: ", ".join([f"{code} ({reverse_mapping[code]})" for code in x]))
        rules = rules.sort_values(by='lift', ascending=False)

    # Metrik Ringkasan Atas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transaksi", f"{basket_sets.shape[0]:,}")
    c2.metric("Frequent Itemsets", f"{len(fi_active):,}")
    c3.metric(f"Association Rules ({algo_choice})", f"{len(rules):,}")
    c4.metric("Waktu Eksekusi Single Run", f"{t_ap if algo_choice=='Apriori' else t_fp:.4f} s")

    st.markdown("---")

    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "🚀 Top 10 Rules (Bundling)", 
        "📈 Semua Aturan Asosiasi", 
        "🔤 Kamus Kode Ukuran", 
        "📋 Data Transaksi Berkode"
    ])

    with sub_tab1:
        st.subheader("🔥 Top 10 Association Rules Terbaik (Berdasarkan Lift Ratio)")
        st.markdown(f"Parameter Aktif: **Min Support = {min_supp}** | **Min Confidence = {min_conf}** | **Algoritma = {algo_choice}**")
        
        if not rules.empty:
            top_10 = rules.head(10)[['antecedents_full', 'consequents_full', 'support', 'confidence', 'lift']].copy()
            top_10.columns = ['Antecedents (Kode & Ukuran Asli)', 'Consequents (Kode & Ukuran Asli)', 'Support', 'Confidence', 'Lift Ratio']
            st.dataframe(top_10, use_container_width=True)
        else:
            st.warning("Tidak ada aturan asosiasi yang ditemukan dengan parameter tersebut.")

    with sub_tab2:
        st.subheader(f"Seluruh Aturan Asosiasi ({len(rules)} Rules)")
        if not rules.empty:
            st.dataframe(rules[['antecedents_full', 'consequents_full', 'support', 'confidence', 'lift']], use_container_width=True)
        else:
            st.warning("Data aturan asosiasi kosong.")

    with sub_tab3:
        st.subheader("🔤 Kamus Referensi Pemetaan Ukuran Item ke Kode Abjad")
        df_kamus = pd.DataFrame(list(item_mapping.items()), columns=['UKURAN_ITEM_ASLI', 'KODE_ABJAD'])
        df_kamus = df_kamus.sort_values(by='KODE_ABJAD').reset_index(drop=True)
        st.dataframe(df_kamus, use_container_width=True)

    with sub_tab4:
        st.subheader("📋 Pratinjau Data Transaksi Hasil Clean & Mapping Kode")
        st.dataframe(df_clean[['TANGGAL', 'CUSTOMER', 'ID TRANSAKSI', 'ITEM', 'ITEM_KODE', 'QTY']].head(100), use_container_width=True)

# =========================================================
# MENU 3: PENGUJIAN 5 ITERASI & PERBANDINGAN KINERJA
# =========================================================
elif menu_pilihan == "⚖️ Pengujian 5 Iterasi":
    st.title("⚖️ Hasil Pengujian 5 Iterasi Komparatif")
    st.markdown("Berikut adalah tabel rincian hasil pengujian komparatif 5 iterasi yang disesuaikan dengan hasil eksperimen terbaru pada Google Colab.")
    st.markdown("---")

    min_supp = st.session_state.min_supp
    min_conf = st.session_state.min_conf

    df_iter = pd.DataFrame([
        {'Pengujian': 1, 'Waktu Apriori (detik)': 0.111713, 'Waktu FP-Growth (detik)': 0.296784, 'Frequent Itemsets Apriori': 126.0, 'Frequent Itemsets FP-Growth': 126.0, 'Association Rules Apriori': 86.0, 'Association Rules FP-Growth': 86.0},
        {'Pengujian': 2, 'Waktu Apriori (detik)': 0.029206, 'Waktu FP-Growth (detik)': 0.165974, 'Frequent Itemsets Apriori': 126.0, 'Frequent Itemsets FP-Growth': 126.0, 'Association Rules Apriori': 86.0, 'Association Rules FP-Growth': 86.0},
        {'Pengujian': 3, 'Waktu Apriori (detik)': 0.020686, 'Waktu FP-Growth (detik)': 0.152908, 'Frequent Itemsets Apriori': 126.0, 'Frequent Itemsets FP-Growth': 126.0, 'Association Rules Apriori': 86.0, 'Association Rules FP-Growth': 86.0},
        {'Pengujian': 4, 'Waktu Apriori (detik)': 0.020072, 'Waktu FP-Growth (detik)': 0.153038, 'Frequent Itemsets Apriori': 126.0, 'Frequent Itemsets FP-Growth': 126.0, 'Association Rules Apriori': 86.0, 'Association Rules FP-Growth': 86.0},
        {'Pengujian': 5, 'Waktu Apriori (detik)': 0.019503, 'Waktu FP-Growth (detik)': 0.453314, 'Frequent Itemsets Apriori': 126.0, 'Frequent Itemsets FP-Growth': 126.0, 'Association Rules Apriori': 86.0, 'Association Rules FP-Growth': 86.0}
    ])

    summary_df = pd.DataFrame([
        {
            'Pengujian': 'Rata-rata',
            'Waktu Apriori (detik)': 0.040236,
            'Waktu FP-Growth (detik)': 0.244403,
            'Frequent Itemsets Apriori': 126.0,
            'Frequent Itemsets FP-Growth': 126.0,
            'Association Rules Apriori': 86.0,
            'Association Rules FP-Growth': 86.0
        },
        {
            'Pengujian': 'Standar Deviasi',
            'Waktu Apriori (detik)': 0.040154,
            'Waktu FP-Growth (detik)': 0.131584,
            'Frequent Itemsets Apriori': float('nan'),
            'Frequent Itemsets FP-Growth': float('nan'),
            'Association Rules Apriori': float('nan'),
            'Association Rules FP-Growth': float('nan')
        }
    ])

    df_tabel_komparasi = pd.concat([df_iter, summary_df], ignore_index=True)

    st.subheader("📊 Tabel Perbandingan Hasil Algoritma Apriori dan FP-Growth")
    st.dataframe(df_tabel_komparasi.style.format({
        'Waktu Apriori (detik)': '{:.6f}',
        'Waktu FP-Growth (detik)': '{:.6f}',
        'Frequent Itemsets Apriori': '{:.1f}',
        'Frequent Itemsets FP-Growth': '{:.1f}',
        'Association Rules Apriori': '{:.1f}',
        'Association Rules FP-Growth': '{:.1f}'
    }, na_rep="NaN"), use_container_width=True)

    st.markdown("---")
    st.markdown("### Ringkasan Jumlah Hasil Algoritma")
    col_sum1, col_sum2 = st.columns(2)
    with col_sum1:
        st.markdown("**APRIORI**")
        st.write(f"Jumlah Frequent Itemsets : {int(df_tabel_komparasi.loc[0, 'Frequent Itemsets Apriori'])}")
        st.write(f"Jumlah Association Rules : {int(df_tabel_komparasi.loc[0, 'Association Rules Apriori'])}")
    with col_sum2:
        st.markdown("**FP-GROWTH**")
        st.write(f"Jumlah Frequent Itemsets : {int(df_tabel_komparasi.loc[0, 'Frequent Itemsets FP-Growth'])}")
        st.write(f"Jumlah Association Rules : {int(df_tabel_komparasi.loc[0, 'Association Rules FP-Growth'])}")

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("📊 Grouped Bar Chart per Iterasi")
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(df_iter))
        width = 0.35

        bars1 = ax.bar(x - width/2, df_iter['Waktu Apriori (detik)'], width, label='Apriori', color='#1f77b4')
        bars2 = ax.bar(x + width/2, df_iter['Waktu FP-Growth (detik)'], width, label='FP-Growth', color='#ff7f0e')

        ax.set_ylabel('Waktu Eksekusi (Detik)', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Pengujian {i}" for i in df_iter['Pengujian']], rotation=15, fontsize=8)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        for bar in bars1:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f"{yval:.4f}s", ha='center', va='bottom', fontsize=7, fontweight='bold')

        for bar in bars2:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f"{yval:.4f}s", ha='center', va='bottom', fontsize=7, fontweight='bold')

        st.pyplot(fig)

    with col_g2:
        st.subheader("📈 Scatter Plot Support vs Confidence")
        fi_ap = apriori(basket_sets, min_support=min_supp, use_colnames=True)
        r_ap = association_rules(fi_ap, metric="confidence", min_threshold=min_conf) if not fi_ap.empty else pd.DataFrame()
        
        if not r_ap.empty:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sc = ax2.scatter(r_ap['support'], r_ap['confidence'], c=r_ap['lift'], cmap='viridis', s=60, alpha=0.85)
            plt.colorbar(sc, label='Lift Ratio')
            ax2.set_xlabel('Support')
            ax2.set_ylabel('Confidence')
            ax2.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig2)
        else:
            st.info("Scatter plot memerlukan data aturan asosiasi.")

# =========================================================
# MENU 4: GRAFIK & FILTER PENJUALAN
# =========================================================
elif menu_pilihan == "📊 Grafik & Filter Penjualan":
    st.title("📊 Filter Data Penjualan & Grafik Tren Bulanan")
    st.markdown("Gunakan kotak filter di bawah ini untuk menyaring data transaksi berdasarkan Nama Pelanggan, Bulan Transaksi, atau Ukuran Item (Kode Abjad).")
    st.markdown("---")

    st.subheader("🔍 Parameter Filter Interaktif")
    f_col1, f_col2, f_col3 = st.columns(3)

    list_pelanggan = ['Semua Pelanggan'] + sorted(df_clean['CUSTOMER'].unique().tolist())

    bulan_urutan = {
        1: 'January 2025', 2: 'February 2025', 3: 'March 2025', 4: 'April 2025',
        5: 'May 2025', 6: 'June 2025', 7: 'July 2025', 8: 'August 2025',
        9: 'September 2025', 10: 'October 2025', 11: 'November 2025', 12: 'December 2025'
    }
    available_months_num = sorted(df_clean['MONTH_NUM'].dropna().unique().astype(int).tolist())
    list_bulan = ['Semua Bulan'] + [df_clean[df_clean['MONTH_NUM'] == m]['BULAN'].iloc[0] for m in available_months_num if m in bulan_urutan]

    formatted_items = [f"{code} - {item}" for item, code in sorted(item_mapping.items(), key=lambda x: x[1])]
    list_ukuran = ['Semua Ukuran'] + formatted_items

    with f_col1:
        pilih_pelanggan = st.selectbox("Pilih Pelanggan:", list_pelanggan)
    with f_col2:
        pilih_bulan = st.selectbox("Pilih Bulan Transaksi:", list_bulan)
    with f_col3:
        pilih_ukuran = st.selectbox("Pilih Ukuran Item (Kode Abjad):", list_ukuran)

    st.markdown("---")

    df_filtered = df_clean.copy()
    if pilih_pelanggan != 'Semua Pelanggan':
        df_filtered = df_filtered[df_filtered['CUSTOMER'] == pilih_pelanggan]
    if pilih_bulan != 'Semua Bulan':
        df_filtered = df_filtered[df_filtered['BULAN'] == pilih_bulan]
    if pilih_ukuran != 'Semua Ukuran':
        selected_item_name = pilih_ukuran.split(" - ", 1)[1]
        df_filtered = df_filtered[df_filtered['ITEM'] == selected_item_name]

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    col_f1.metric("Total Baris Sesuai Filter", f"{len(df_filtered):,}")
    col_f2.metric("Jumlah Transaksi Unik", f"{df_filtered['ID TRANSAKSI'].nunique():,}")
    col_f3.metric("Total Qty Terjual", f"{df_filtered['QTY'].sum():,.0f}" if 'QTY' in df_filtered.columns else "N/A")
    col_f4.metric("Total Nilai Penjualan", f"Rp {df_filtered['TOTAL'].sum():,.0f}" if 'TOTAL' in df_filtered.columns else "N/A")

    st.markdown("---")

    st.subheader("📈 Grafik Tren Penjualan per Bulan")
    if 'TANGGAL' in df_clean.columns and 'QTY' in df_clean.columns:
        df_trend_all = pd.DataFrame({'MONTH_NUM': range(1, 13)})
        df_grouped = df_filtered.groupby('MONTH_NUM')['QTY'].sum().reset_index()
        df_trend_merged = pd.merge(df_trend_all, df_grouped, on='MONTH_NUM', how='left').fillna(0)
        
        month_names_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                           7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        df_trend_merged['Bulan_Label'] = df_trend_merged['MONTH_NUM'].map(month_names_map)

        fig_trend, ax_trend = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=df_trend_merged, x='Bulan_Label', y='QTY', marker='o', color='#1f77b4', linewidth=2.5, ax=ax_trend)
        ax_trend.set_title("Total Kuantitas Penjualan per Bulan (Berdasarkan Filter Aktif)", fontsize=12, fontweight='bold')
        ax_trend.set_xlabel("Bulan", fontweight='bold')
        ax_trend.set_ylabel("Total Qty Penjualan", fontweight='bold')
        ax_trend.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_trend)
    else:
        st.info("Kolom 'TANGGAL' atau 'QTY' tidak tersedia untuk membuat grafik tren.")

    st.markdown("---")
    st.subheader("📋 Tabel Rincian Transaksi Sesuai Filter")
    if not df_filtered.empty:
        st.dataframe(df_filtered[['TANGGAL', 'CUSTOMER', 'ID TRANSAKSI', 'ITEM', 'ITEM_KODE', 'QTY', 'HARGA', 'TOTAL']], use_container_width=True)
    else:
        st.warning("⚠️ Tidak ada data transaksi yang cocok dengan kombinasi filter yang Anda pilih.")

# =========================================================
# MENU 5: PARAMETER MODEL
# =========================================================
elif menu_pilihan == "⚙️ Parameter Model":
    st.title("⚙️ Pengaturan & Simulasi Parameter Model")
    st.markdown("Atur parameter uji minimum *Support* dan *Confidence* di bawah ini untuk melihat hasil simulasi pembentukan aturan asosiasi secara *real-time*.")
    st.markdown("---")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.session_state.algo_choice = st.selectbox("Pilih Algoritma Utama:", ["FP-Growth", "Apriori"], index=0 if st.session_state.algo_choice=="FP-Growth" else 1)
    with col_p2:
        st.session_state.min_supp = st.slider("Minimum Support", 0.001, 0.050, st.session_state.min_supp, 0.001, format="%.3f")
    with col_p3:
        st.session_state.min_conf = st.slider("Minimum Confidence", 0.1, 0.9, st.session_state.min_conf, 0.1)

    st.markdown("---")
    st.info(f"⚡ Menjalankan simulasi menggunakan algoritma **{st.session_state.algo_choice}** dengan Min Support = **{st.session_state.min_supp}** dan Min Confidence = **{st.session_state.min_conf}**...")

    with st.spinner("Memproses perhitungan data mining..."):
        start_time = time.perf_counter()
        if st.session_state.algo_choice == "Apriori":
            fi_sim = apriori(basket_sets, min_support=st.session_state.min_supp, use_colnames=True)
            rules_sim = association_rules(fi_sim, metric="confidence", min_threshold=st.session_state.min_conf) if not fi_sim.empty else pd.DataFrame()
        else:
            fi_sim = fpgrowth(basket_sets, min_support=st.session_state.min_supp, use_colnames=True)
            rules_sim = association_rules(fi_sim, metric="confidence", min_threshold=st.session_state.min_conf) if not fi_sim.empty else pd.DataFrame()
        sim_duration = time.perf_counter() - start_time

    m_s1, m_s2, m_s3 = st.columns(3)
    m_s1.metric("Waktu Eksekusi Simulasi", f"{sim_duration:.4f} detik")
    m_s2.metric("Jumlah Frequent Itemsets", f"{len(fi_sim):,}")
    m_s3.metric("Jumlah Association Rules", f"{len(rules_sim):,}")

    st.markdown("---")
    st.subheader("📋 Hasil Preview Aturan Asosiasi (Simulasi)")

    if not rules_sim.empty:
        rules_sim['antecedents_full'] = rules_sim['antecedents'].apply(lambda x: ", ".join([f"{code} ({reverse_mapping[code]})" for code in x]))
        rules_sim['consequents_full'] = rules_sim['consequents'].apply(lambda x: ", ".join([f"{code} ({reverse_mapping[code]})" for code in x]))
        rules_sim = rules_sim.sort_values(by='lift', ascending=False)
        
        preview_df = rules_sim[['antecedents_full', 'consequents_full', 'support', 'confidence', 'lift']].head(15).copy()
        preview_df.columns = ['Antecedents (Kode & Ukuran)', 'Consequents (Kode & Ukuran)', 'Support', 'Confidence', 'Lift Ratio']
        st.dataframe(preview_df, use_container_width=True)

        csv_data = rules_sim.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Unduh Hasil Simulasi Aturan Asosiasi (CSV)",
            data=csv_data,
            file_name=f"Simulasi_Rules_{st.session_state.algo_choice}_Supp{st.session_state.min_supp}_Conf{st.session_state.min_conf}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Tidak ada aturan asosiasi yang terbentuk dengan kombinasi parameter tersebut. Coba turunkan nilai *Support* atau *Confidence*.")
