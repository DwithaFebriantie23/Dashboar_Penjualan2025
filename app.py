
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# 1. KONFIGURASI DASHBOARD
# =========================================================

st.set_page_config(
    page_title="Dashboard Penjualan PT. Cakrawala Kemasindo Lestari",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
FILE_EXCEL = BASE_DIR / "Penjualan2025(4).xlsx"


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

    # Menghapus data yang tidak memiliki ID transaksi atau ITEM
    df = df.dropna(
        subset=["ID TRANSAKSI", "ITEM"]
    ).copy()

    # Mengubah format ID transaksi menjadi teks
    def normalisasi_id(value):
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        return str(value).strip()

    df["ID TRANSAKSI"] = (
        df["ID TRANSAKSI"].apply(normalisasi_id)
    )

    # Mempertahankan nama ITEM asli
    df["ITEM"] = (
        df["ITEM"].astype(str).str.strip()
    )

    # Membersihkan nama pelanggan
    df["CUSTOMER"] = (
        df["CUSTOMER"].fillna("Tidak Diketahui")
        .astype(str)
        .str.strip()
    )

    # Mengubah tanggal menjadi format datetime
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

    # Mengeluarkan baris dengan TOTAL = 0
    df = df[df["TOTAL"] != 0].copy()

    # Menghapus data yang tidak memiliki tanggal
    # agar dapat digunakan dalam grafik bulanan
    df["BULAN"] = df["TANGGAL"].dt.to_period("M")

    return df


# =========================================================
# 3. MEMUAT DATA
# =========================================================

st.sidebar.title("📊 Dashboard PT. CKL")

try:
    df = load_data()

except FileNotFoundError:
    st.error(
        "File Penjualan2025(4).xlsx tidak ditemukan. "
        "Pastikan file Excel berada satu folder dengan app.py."
    )
    st.stop()

except Exception as e:
    st.error(f"Terjadi kesalahan saat membaca Excel: {e}")
    st.stop()


# =========================================================
# 4. DATA HASIL PENELITIAN APRIORI DARI COLAB
# =========================================================

DATA_PENELITIAN = {
    "data_awal": 1889,
    "data_dikeluarkan": 6,
    "data_bersih": 1883,
    "transaksi": 1146,
    "produk": 422,
    "matriks_basket": "1.146 × 422"
}


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


# =========================================================
# 5. FUNGSI FORMAT ANGKA
# =========================================================

def rupiah(nilai):
    return "Rp " + f"{nilai:,.0f}".replace(",", ".")


def angka(nilai):
    return f"{nilai:,.0f}".replace(",", ".")


# =========================================================
# 6. NAVIGASI SIDEBAR
# =========================================================

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Beranda",
        "Karakteristik Data",
        "Analisis Penjualan",
        "Hasil Apriori",
        "Kandidat Paket Penjualan",
        "Pengujian Parameter"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Sistem Informasi | Data Science"
)

st.sidebar.caption(
    "Studi Kasus: PT. Cakrawala Kemasindo Lestari"
)


# =========================================================
# 7. HALAMAN BERANDA
# =========================================================

if menu == "Beranda":

    st.title("📊 Dashboard Penjualan PT. Cakrawala Kemasindo Lestari")

    st.write(
        "Dashboard ini menyajikan informasi penjualan "
        "PT Cakrawala Kemasindo Lestari tahun 2025 "
        "serta hasil penelitian menggunakan algoritma Apriori."
    )

    st.divider()

    total_penjualan = df["TOTAL"].sum()
    total_transaksi = df["ID TRANSAKSI"].nunique()
    total_pelanggan = df["CUSTOMER"].nunique()
    total_produk = df["ITEM"].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Penjualan",
        rupiah(total_penjualan)
    )

    col2.metric(
        "Jumlah Transaksi",
        angka(total_transaksi)
    )

    col3.metric(
        "Jumlah Pelanggan",
        angka(total_pelanggan)
    )

    col4.metric(
        "Jenis Produk",
        angka(total_produk)
    )

    st.divider()

    st.subheader("Ringkasan Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            """
            **Analisis Penjualan**

            Menampilkan informasi transaksi penjualan
            berdasarkan periode, pelanggan, dan produk.
            """
        )

    with col2:
        st.success(
            """
            **Hasil Algoritma Apriori**

            Menampilkan hasil pengujian parameter,
            aturan asosiasi, dan kandidat paket penjualan.
            """
        )

    st.subheader("Grafik Penjualan Bulanan")

    data_bulan = (
        df.dropna(subset=["TANGGAL"])
        .groupby("BULAN")["TOTAL"]
        .sum()
        .reset_index()
    )

    data_bulan["BULAN"] = (
        data_bulan["BULAN"].astype(str)
    )

    if not data_bulan.empty:

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.bar(
            data_bulan["BULAN"],
            data_bulan["TOTAL"]
        )

        ax.set_xlabel("Bulan")
        ax.set_ylabel("Total Penjualan (Rp)")
        ax.set_title("Total Penjualan PT CKL Tahun 2025")

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)


# =========================================================
# 8. HALAMAN KARAKTERISTIK DATA
# =========================================================

elif menu == "Karakteristik Data":

    st.title("📁 Karakteristik Data Penjualan")

    st.write(
        "Halaman ini menampilkan karakteristik data "
        "penjualan yang dibaca dari file Excel."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Jumlah Baris Data",
        angka(len(df))
    )

    col2.metric(
        "Jumlah Transaksi",
        angka(df["ID TRANSAKSI"].nunique())
    )

    col3.metric(
        "Jumlah Jenis Produk",
        angka(df["ITEM"].nunique())
    )

    st.divider()

    st.subheader("Informasi Kolom Dataset")

    informasi = pd.DataFrame({
        "Nama Kolom": df.columns,
        "Jumlah Data Terisi": [
            df[kolom].notna().sum()
            for kolom in df.columns
        ],
        "Tipe Data": [
            str(df[kolom].dtype)
            for kolom in df.columns
        ]
    })

    st.dataframe(
        informasi,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Data Penjualan")

    st.dataframe(
        df[
            [
                "TANGGAL",
                "CUSTOMER",
                "ID TRANSAKSI",
                "ITEM",
                "QTY",
                "HARGA",
                "TOTAL"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Data ditampilkan berdasarkan isi file Excel "
        "setelah proses pembersihan."
    )


# =========================================================
# 9. HALAMAN ANALISIS PENJUALAN
# =========================================================

elif menu == "Analisis Penjualan":

    st.title("📈 Analisis Penjualan")

    st.write(
        "Gunakan filter berikut untuk melihat penjualan "
        "berdasarkan bulan, pelanggan, dan produk."
    )

    # -------------------------
    # FILTER DATA
    # -------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        daftar_bulan = sorted(
            df["BULAN"].dropna().astype(str).unique()
        )

        pilihan_bulan = st.multiselect(
            "Pilih Bulan",
            options=daftar_bulan,
            default=daftar_bulan
        )

    with col2:

        daftar_customer = sorted(
            df["CUSTOMER"].dropna().unique().tolist()
        )

        pilihan_customer = st.multiselect(
            "Pilih Pelanggan",
            options=daftar_customer
        )

    with col3:

        daftar_item = sorted(
            df["ITEM"].dropna().unique().tolist()
        )

        pilihan_item = st.multiselect(
            "Pilih UK Produk",
            options=daftar_item
        )

    # -------------------------
    # PROSES FILTER
    # -------------------------

    data_filter = df.copy()

    if pilihan_bulan:
        data_filter = data_filter[
            data_filter["BULAN"].astype(str).isin(
                pilihan_bulan
            )
        ]
    else:
        data_filter = data_filter.iloc[0:0]

    if pilihan_customer:
        data_filter = data_filter[
            data_filter["CUSTOMER"].isin(
                pilihan_customer
            )
        ]

    if pilihan_item:
        data_filter = data_filter[
            data_filter["ITEM"].isin(
                pilihan_item
            )
        ]

    st.divider()

    # -------------------------
    # RINGKASAN HASIL FILTER
    # -------------------------

    total_penjualan = data_filter["TOTAL"].sum()
    total_transaksi = data_filter["ID TRANSAKSI"].nunique()
    total_pelanggan = data_filter["CUSTOMER"].nunique()
    total_qty = data_filter["QTY"].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Penjualan",
        rupiah(total_penjualan)
    )

    col2.metric(
        "Jumlah Transaksi",
        angka(total_transaksi)
    )

    col3.metric(
        "Jumlah Pelanggan",
        angka(total_pelanggan)
    )

    col4.metric(
        "Total QTY",
        angka(total_qty)
    )

    st.divider()

    # -------------------------
    # GRAFIK PENJUALAN BULANAN
    # -------------------------

    st.subheader("Total Penjualan per Bulan")

    penjualan_bulanan = (
        data_filter.dropna(subset=["TANGGAL"])
        .groupby("BULAN")["TOTAL"]
        .sum()
        .reset_index()
    )

    penjualan_bulanan["BULAN"] = (
        penjualan_bulanan["BULAN"].astype(str)
    )

    if not penjualan_bulanan.empty:

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.bar(
            penjualan_bulanan["BULAN"],
            penjualan_bulanan["TOTAL"]
        )

        ax.set_xlabel("Bulan")
        ax.set_ylabel("Total Penjualan (Rp)")
        ax.set_title("Penjualan Berdasarkan Bulan")

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning("Tidak ada data untuk ditampilkan.")

    st.divider()

    # -------------------------
    # PENJUALAN BERDASARKAN PELANGGAN
    # -------------------------

    st.subheader("Total Penjualan Berdasarkan Pelanggan")

    penjualan_customer = (
        data_filter.groupby("CUSTOMER")["TOTAL"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    if not penjualan_customer.empty:

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.barh(
            penjualan_customer["CUSTOMER"].iloc[::-1],
            penjualan_customer["TOTAL"].iloc[::-1]
        )

        ax.set_xlabel("Total Penjualan (Rp)")
        ax.set_ylabel("Nama Pelanggan")
        ax.set_title("Penjualan Berdasarkan Pelanggan")

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            penjualan_customer,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("Belum ada data pelanggan yang sesuai filter.")

    st.divider()

    # -------------------------
    # PRODUK TERLARIS
    # -------------------------

    st.subheader("Produk Berdasarkan Total QTY")

    produk_terlaris = (
        data_filter.groupby("ITEM")["QTY"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    if not produk_terlaris.empty:

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.barh(
            produk_terlaris["ITEM"].iloc[::-1],
            produk_terlaris["QTY"].iloc[::-1]
        )

        ax.set_xlabel("Total QTY")
        ax.set_ylabel("UK Produk")
        ax.set_title("10 Produk dengan QTY Tertinggi")

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            produk_terlaris,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # -------------------------
    # TABEL DETAIL DATA
    # -------------------------

    st.subheader("Detail Data Berdasarkan Filter")

    st.write(
        f"Jumlah baris data yang sesuai: {len(data_filter)}"
    )

    st.dataframe(
        data_filter[
            [
                "TANGGAL",
                "CUSTOMER",
                "ID TRANSAKSI",
                "ITEM",
                "QTY",
                "HARGA",
                "TOTAL"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # Tombol unduh hasil filter
    csv = data_filter.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        label="📥 Unduh Data Hasil Filter",
        data=csv,
        file_name="data_penjualan_hasil_filter.csv",
        mime="text/csv"
    )


# =========================================================
# 10. HALAMAN HASIL APRIORI
# =========================================================

elif menu == "Hasil Apriori":

    st.title("🔍 Hasil Penelitian Algoritma Apriori")

    st.write(
        "Halaman ini menyajikan ringkasan hasil pengolahan "
        "algoritma Apriori yang telah dilakukan di Google Colab."
    )

    st.subheader("Parameter Terpilih")

    col1, col2 = st.columns(2)

    col1.metric(
        "Minimum Support",
        "0,010"
    )

    col2.metric(
        "Minimum Confidence",
        "0,20"
    )

    st.divider()

    st.subheader("Hasil Pembentukan Pola")

    col1, col2 = st.columns(2)

    col1.metric(
        "Frequent Itemsets",
        "45"
    )

    col2.metric(
        "Aturan Asosiasi",
        "20"
    )

    st.divider()

    st.subheader("Rata-rata Nilai Aturan Asosiasi")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Support",
        "0,012129"
    )

    col2.metric(
        "Average Confidence",
        "85,54%"
    )

    col3.metric(
        "Average Lift",
        "64,262390"
    )

    st.divider()

    st.subheader("Interpretasi Hasil")

    st.write(
        """
        Berdasarkan konfigurasi parameter terpilih, algoritma
        Apriori menghasilkan 45 frequent itemsets dan 20 aturan
        asosiasi.

        Nilai lift seluruh aturan yang dihasilkan lebih dari 1.
        Hal ini menunjukkan adanya asosiasi positif antara
        produk-produk yang muncul dalam aturan tersebut.

        Hasil ini kemudian digunakan untuk mengidentifikasi
        kombinasi produk yang berpotensi menjadi kandidat
        paket penjualan PT CKL.
        """
    )


# =========================================================
# 11. HALAMAN KANDIDAT PAKET PENJUALAN
# =========================================================

elif menu == "Kandidat Paket Penjualan":

    st.title("📦 Kandidat Paket Penjualan")

    st.write(
        "Berikut merupakan lima kombinasi produk yang "
        "diperoleh dari hasil pengolahan aturan asosiasi Apriori."
    )

    st.info(
        "Kandidat paket merupakan rekomendasi berdasarkan "
        "pola transaksi historis, bukan paket yang sudah "
        "diterapkan oleh perusahaan."
    )

    st.dataframe(
        DATA_KANDIDAT,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Visualisasi Jumlah Transaksi")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.barh(
        DATA_KANDIDAT["Kandidat Paket"].iloc[::-1],
        DATA_KANDIDAT["Jumlah Transaksi"].iloc[::-1]
    )

    ax.set_xlabel("Jumlah Transaksi")
    ax.set_ylabel("Kandidat Paket")
    ax.set_title("Jumlah Transaksi pada Setiap Kandidat Paket")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Kesimpulan Kandidat")

    st.write(
        """
        - Kandidat 1 memiliki nilai confidence dan lift tertinggi.
        - Kandidat 2 muncul pada jumlah pelanggan terbanyak.
        - Kandidat 3 memiliki support dan jumlah transaksi tertinggi.

        Dengan demikian, pemilihan kandidat paket tidak hanya
        mempertimbangkan satu nilai metrik, tetapi juga
        frekuensi kemunculan dan sebaran pelanggan.
        """
    )


# =========================================================
# 12. HALAMAN PENGUJIAN PARAMETER
# =========================================================

elif menu == "Pengujian Parameter":

    st.title("⚙️ Pengujian Parameter Algoritma Apriori")

    st.write(
        "Halaman ini menampilkan perbandingan lima konfigurasi "
        "minimum support dan minimum confidence."
    )

    st.dataframe(
        DATA_PENGUJIAN,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Grafik Jumlah Frequent Itemsets")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        DATA_PENGUJIAN["Pengujian"],
        DATA_PENGUJIAN["Frequent Itemsets"]
    )

    ax.set_xlabel("Konfigurasi Pengujian")
    ax.set_ylabel("Jumlah Frequent Itemsets")
    ax.set_title("Perbandingan Frequent Itemsets")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Grafik Jumlah Aturan Asosiasi")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        DATA_PENGUJIAN["Pengujian"],
        DATA_PENGUJIAN["Aturan Asosiasi"]
    )

    ax.set_xlabel("Konfigurasi Pengujian")
    ax.set_ylabel("Jumlah Aturan Asosiasi")
    ax.set_title("Perbandingan Aturan Asosiasi")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Parameter yang Dipilih")

    st.success(
        """
        Pengujian kedua dipilih dengan minimum support 0,010
        dan minimum confidence 0,20.

        Konfigurasi ini menghasilkan 45 frequent itemsets
        dan 20 aturan asosiasi. Konfigurasi tersebut dipilih
        sebagai keseimbangan antara jumlah pola yang ditemukan
        dan jumlah aturan asosiasi yang dihasilkan.
        """
    )


# =========================================================
# 13. FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Dashboard Penjualan dan Market Basket Analysis"
)

st.sidebar.caption(
    "PT Cakrawala Kemasindo Lestari | 2025"
)
