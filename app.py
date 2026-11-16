import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# 1. KONFIGURASI DASHBOARD
# =========================================================

st.set_page_config(
    page_title="Dashboard Penjualan PT CKL",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
FILE_EXCEL = BASE_DIR / "Penjualan2025.xlsx"


# =========================================================
# 2. MEMBACA DAN MEMBERSIHKAN DATA EXCEL
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_excel(
        FILE_EXCEL,
        sheet_name="Rekap 2025",
        header=1
    )

    # Menghapus data tanpa ID transaksi dan ITEM
    df = df.dropna(
        subset=["ID TRANSAKSI", "ITEM"]
    ).copy()

    # Normalisasi ID transaksi
    def normalisasi_id(value):
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        return str(value).strip()

    df["ID TRANSAKSI"] = (
        df["ID TRANSAKSI"].apply(normalisasi_id)
    )

    # Mempertahankan nama item asli
    df["ITEM"] = (
        df["ITEM"].astype(str).str.strip()
    )

    # Membersihkan nama pelanggan
    df["CUSTOMER"] = (
        df["CUSTOMER"]
        .fillna("Tidak Diketahui")
        .astype(str)
        .str.strip()
    )

    # Mengubah tanggal
    df["TANGGAL"] = pd.to_datetime(
        df["TANGGAL"],
        errors="coerce"
    )

    # Mengubah kolom numerik
    for kolom in ["QTY", "HARGA", "TOTAL"]:
        df[kolom] = pd.to_numeric(
            df[kolom],
            errors="coerce"
        )

    # Mengeluarkan data TOTAL = 0
    df = df[
        df["TOTAL"].notna() &
        (df["TOTAL"] != 0)
    ].copy()

    # Membuat kolom bulan
    df["BULAN"] = (
        df["TANGGAL"].dt.to_period("M")
    )

    return df


# =========================================================
# 3. MEMUAT DATA
# =========================================================

try:

    df = load_data()

except FileNotFoundError:

    st.error(
        "File Penjualan2025.xlsx tidak ditemukan. "
        "Pastikan file Excel berada satu folder dengan app.py."
    )

    st.stop()

except Exception as e:

    st.error(
        f"Terjadi kesalahan saat membaca Excel: {e}"
    )

    st.stop()


# =========================================================
# 4. FUNGSI FORMAT ANGKA
# =========================================================

def rupiah(nilai):

    return (
        "Rp "
        + f"{nilai:,.0f}".replace(",", ".")
    )


def angka(nilai):

    return f"{nilai:,.0f}".replace(",", ".")


# =========================================================
# 5. DATA HASIL PENELITIAN APRIORI
# =========================================================

DATA_PENGUJIAN = pd.DataFrame({

    "Pengujian": [
        "Pengujian 1",
        "Pengujian 2",
        "Pengujian 3",
        "Pengujian 4",
        "Pengujian 5"
    ],

    "Min Support": [
        0.005, 0.010, 0.015, 0.020, 0.025
    ],

    "Min Confidence": [
        0.10, 0.20, 0.30, 0.40, 0.50
    ],

    "Frequent Itemsets": [
        125, 45, 19, 11, 6
    ],

    "Aturan Asosiasi": [
        86, 20, 2, 0, 0
    ]

})

DATA_KANDIDAT = pd.DataFrame({

    "Kandidat Paket": [
        "100x3x780 + 100x3x1100 + 100x3x1130",
        "80x80x5x520 + 80x80x5x800",
        "100x100x8x200 + 100x100x8x250",
        "50x50x3x125 + 50x50x3x240",
        "100x100x5x1150 + 100x100x8x250"
    ],

    "Support": [
        0.012216,
        0.010471,
        0.015707,
        0.010471,
        0.011344
    ],

    "Confidence": [
        "100%",
        "92,31%",
        "54,55%–78,26%",
        "44,44%–63,16%",
        "39,39%–46,43%"
    ],

    "Lift": [
        81.857143,
        81.372781,
        27.177866,
        26.807018,
        16.123377
    ],

    "Jumlah Transaksi": [
        14, 12, 18, 12, 13
    ],

    "Jumlah Pelanggan": [
        2, 3, 1, 2, 1
    ]

})

DATA_FREQUENT = pd.DataFrame({
    "Itemsets": [
        "('100x3x780',)",
        "('100x3x1100',)",
        "('100x3x780', '100x3x1100')",
        "('80x80x5x520', '80x80x5x800')",
        "('100x100x8x200', '100x100x8x250')"
    ],
    "Support": [0.035, 0.028, 0.012, 0.010, 0.015]
})

DATA_RULES = pd.DataFrame({
    "Antecedent (If)": ["100x3x780", "80x80x5x520", "100x100x8x200", "50x50x3x125", "100x100x5x1150"],
    "Consequent (Then)": ["100x3x1100", "80x80x5x800", "100x100x8x250", "50x50x3x240", "100x100x8x250"],
    "Support": [0.0122, 0.0104, 0.0157, 0.0104, 0.0113],
    "Confidence": ["100,00%", "92,31%", "78,26%", "63,16%", "46,43%"],
    "Lift Ratio": [81.85, 81.37, 27.17, 26.80, 16.12]
})


# =========================================================
# 6. NAVIGASI SIDEBAR
# =========================================================

st.sidebar.title("📊 Dashboard PT CKL")

menu = st.sidebar.radio(
    "Menu Navigasi",
    [
        "🏠 Beranda",
        "📈 Analisis Penjualan",
        "🔍 Hasil Apriori"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Sistem Informasi | Data Science")
st.sidebar.caption("PT Cakrawala Kemasindo Lestari")


# =========================================================
# 7. MENU 1: BERANDA
# =========================================================

if menu == "🏠 Beranda":

    st.title("📊 Dashboard Penjualan PT CKL")
    st.write(
        "Dashboard ini menyajikan ringkasan informasi penjualan "
        "PT Cakrawala Kemasindo Lestari tahun 2025 "
        "serta hasil implementasi algoritma Apriori."
    )

    st.divider()

    total_penjualan = df["TOTAL"].sum()
    total_transaksi = df["ID TRANSAKSI"].nunique()
    total_pelanggan = df["CUSTOMER"].nunique()
    total_produk = df["ITEM"].nunique()

    # METRIK 2 BARIS
    row1_c1, row1_c2 = st.columns(2)
    row1_c1.metric("Total Penjualan", rupiah(total_penjualan))
    row1_c2.metric("Jumlah Transaksi", angka(total_transaksi))

    row2_c1, row2_c2 = st.columns(2)
    row2_c1.metric("Jumlah Pelanggan", angka(total_pelanggan))
    row2_c2.metric("Jenis Produk", angka(total_produk))

    st.divider()

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("Grafik Penjualan Bulanan")
        data_bulan = (
            df.dropna(subset=["TANGGAL"])
            .groupby("BULAN")["TOTAL"]
            .sum()
            .reset_index()
        )
        data_bulan["BULAN"] = data_bulan["BULAN"].astype(str)

        if not data_bulan.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(data_bulan["BULAN"], data_bulan["TOTAL"], color="#1f77b4")
            ax.set_ylabel("Total Penjualan (Rp)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with col_chart2:
        st.subheader("Kontribusi per Bulan")
        data_tahunan = df[df["TANGGAL"].dt.year == 2025].copy()
        penjualan_bulanan = (
            data_tahunan.groupby(data_tahunan["TANGGAL"].dt.month)["TOTAL"]
            .sum()
            .reindex(range(1, 13), fill_value=0)
        )
        nama_bulan = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
        data_pie = penjualan_bulanan[penjualan_bulanan > 0]

        if not data_pie.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.pie(
                data_pie.values,
                labels=[nama_bulan[i - 1] for i in data_pie.index],
                autopct="%1.1f%%",
                startangle=90
            )
            ax.axis("equal")
            st.pyplot(fig)
            plt.close(fig)


# =========================================================
# 8. MENU 2: ANALISIS PENJUALAN (DENGAN TAB)
# =========================================================

elif menu == "📈 Analisis Penjualan":

    st.title("📈 Analisis Penjualan")
    
    tab_analisis, tab_karakteristik = st.tabs([
        "📊 Statistik Penjualan", 
        "📁 Karakteristik Dataset"
    ])

    # ================= TAB 1: STATISTIK PENJUALAN =================
    with tab_analisis:
        st.subheader("Filter Data Interaktif")
        
        col1, col2, col3 = st.columns(3)
        daftar_bulan = sorted(df["BULAN"].dropna().astype(str).unique())
        daftar_customer = sorted(df["CUSTOMER"].dropna().unique().tolist())
        daftar_item = sorted(df["ITEM"].dropna().unique().tolist())

        with col1:
            pilihan_bulan = st.multiselect("Pilih Bulan", options=daftar_bulan, default=daftar_bulan, key="f_bulan")
        with col2:
            pilihan_customer = st.multiselect("Pilih Pelanggan", options=daftar_customer, key="f_customer")
        with col3:
            pilihan_item = st.multiselect("Pilih UK Produk", options=daftar_item, key="f_item")

        # LOGIKA FILTER FLEKSIBEL
        data_filter = df.copy()

        if pilihan_bulan:
            data_filter = data_filter[
                data_filter["BULAN"].astype(str).isin(pilihan_bulan)
            ]

        if pilihan_customer:
            data_filter = data_filter[
                data_filter["CUSTOMER"].isin(pilihan_customer)
            ]

        if pilihan_item:
            data_filter = data_filter[
                data_filter["ITEM"].isin(pilihan_item)
            ]

        st.divider()

        f_row1_c1, f_row1_c2 = st.columns(2)
        f_row1_c1.metric("Total Penjualan", rupiah(data_filter["TOTAL"].sum()))
        f_row1_c2.metric("Jumlah Transaksi", angka(data_filter["ID TRANSAKSI"].nunique()))

        f_row2_c1, f_row2_c2 = st.columns(2)
        f_row2_c1.metric("Jumlah Pelanggan", angka(data_filter["CUSTOMER"].nunique()))
        f_row2_c2.metric("Total QTY", angka(data_filter["QTY"].sum()))

        st.divider()

        st.subheader("10 Produk dengan QTY Tertinggi")
        produk_terlaris = (
            data_filter.groupby("ITEM")["QTY"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        if not produk_terlaris.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(produk_terlaris["ITEM"].iloc[::-1], produk_terlaris["QTY"].iloc[::-1])
            ax.set_xlabel("Total QTY")
            ax.set_ylabel("UK Produk")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Tidak terdapat data produk sesuai kombinasi filter yang dipilih.")

    # ================= TAB 2: KARAKTERISTIK DATASET =================
    with tab_karakteristik:
        st.subheader("Ringkasan Karakteristik Dataset")
        
        c_row1_c1, c_row1_c2, c_row1_c3 = st.columns(3)
        c_row1_c1.metric("Jumlah Baris Data", angka(len(df)))
        c_row1_c2.metric("Total Transaksi Unik", angka(df["ID TRANSAKSI"].nunique()))
        c_row1_c3.metric("Total Jenis Produk", angka(df["ITEM"].nunique()))

        st.divider()

        st.subheader("Informasi Kolom Dataset")
        informasi = pd.DataFrame({
            "Nama Kolom": df.columns,
            "Jumlah Data Terisi": [df[kolom].notna().sum() for kolom in df.columns],
            "Tipe Data": [str(df[kolom].dtype) for kolom in df.columns]
        })
        st.dataframe(informasi, use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("Tabel Seluruh Data Transaksi")
        st.dataframe(
            df[["TANGGAL", "CUSTOMER", "ID TRANSAKSI", "ITEM", "QTY", "HARGA", "TOTAL"]],
            use_container_width=True,
            hide_index=True
        )

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 Unduh Seluruh Data Penjualan (CSV)",
            data=csv,
            file_name="data_penjualan_lengkap.csv",
            mime="text/csv"
        )


# =========================================================
# 9. MENU 3: HASIL APRIORI (DENGAN TAB)
# =========================================================

elif menu == "🔍 Hasil Apriori":

    st.title("🔍 Hasil Penelitian Apriori")

    tab_hasil, tab_kandidat, tab_pengujian = st.tabs([
        "📊 Parameter & Pola", 
        "📦 Kandidat Paket", 
        "⚙️ Evaluasi Pengujian"
    ])

    # ================= TAB 1: PARAMETER & POLA =================
    with tab_hasil:
        st.subheader("Parameter Terpilih & Pembentukan Pola")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Min Support", "0,010")
        col2.metric("Min Confidence", "0,20")
        col3.metric("Frequent Itemsets", "45")
        col4.metric("Aturan Asosiasi", "20")

        st.divider()

        st.subheader("Daftar Frequent Itemsets")
        st.write("Kombinasi produk yang memenuhi nilai minimum support (0,010):")
        st.dataframe(DATA_FREQUENT, use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("Tabel Aturan Asosiasi (Association Rules)")
        st.write("Menampilkan informasi anteseden, konsekuen, support, confidence, dan lift ratio:")
        st.dataframe(DATA_RULES, use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("Rata-rata Nilai Aturan Asosiasi")
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Support", "0,012129")
        col2.metric("Average Confidence", "85,54%")
        col3.metric("Average Lift", "64,262390")

        st.divider()

        st.subheader("Interpretasi Hasil")
        st.write(
            """
            Berdasarkan konfigurasi parameter terpilih, algoritma Apriori menghasilkan 
            45 frequent itemsets dan 20 aturan asosiasi. Nilai lift seluruh aturan lebih dari 1, 
            menunjukkan adanya asosiasi positif yang kuat antar produk.
            """
        )

    # ================= TAB 2: KANDIDAT PAKET =================
    with tab_kandidat:
        st.subheader("5 Kombinasi Kandidat Paket Penjualan")
        st.dataframe(DATA_KANDIDAT, use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("Visualisasi Jumlah Transaksi per Kandidat")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(
            DATA_KANDIDAT["Kandidat Paket"].iloc[::-1],
            DATA_KANDIDAT["Jumlah Transaksi"].iloc[::-1],
            color="#2ca02c"
        )
        ax.set_xlabel("Jumlah Transaksi")
        ax.set_ylabel("Kandidat Paket")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ================= TAB 3: EVALUASI PENGUJIAN =================
    with tab_pengujian:
        st.subheader("Perbandingan Pengujian Parameter")
        st.dataframe(DATA_PENGUJIAN, use_container_width=True, hide_index=True)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Frequent Itemsets**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(DATA_PENGUJIAN["Pengujian"], DATA_PENGUJIAN["Frequent Itemsets"], color="orange")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.write("**Aturan Asosiasi**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(DATA_PENGUJIAN["Pengujian"], DATA_PENGUJIAN["Aturan Asosiasi"], color="purple")
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
