
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1. KONFIGURASI DASHBOARD
# =========================================================

st.set_page_config(
    page_title="Dashboard Analisis Apriori - PT CKL",
    page_icon="📊",
    layout="wide"
)

NAMA_PERUSAHAAN = "PT Cakrawala Kemasindo Lestari"

# =========================================================
# 2. DATA HASIL PENELITIAN DARI GOOGLE COLAB
# Seluruh data di bawah merupakan hasil penelitian,
# bukan hasil pengolahan ulang dataset Excel.
# =========================================================

DATA_PENELITIAN = {
    "data_awal": 1889,
    "data_dikeluarkan": 6,
    "data_bersih": 1883,
    "jumlah_transaksi": 1146,
    "jumlah_item": 422,
    "jumlah_kolom": 7,
    "ukuran_matriks": "1.146 x 422",
    "rata_item": 1.63,
    "std_item": 1.21,
    "min_item": 1,
    "median_item": 1,
    "q1_item": 1,
    "q3_item": 2,
    "max_item": 12
}

PARAMETER_TERPILIH = {
    "minimum_support": 0.010,
    "minimum_confidence": 0.20,
    "frequent_itemsets": 45,
    "aturan_asosiasi": 20,
    "rata_support": 0.012129,
    "rata_confidence": 0.855423,
    "rata_lift": 64.262390
}

# Hasil lima konfigurasi pengujian dari Colab
DATA_PENGUJIAN = pd.DataFrame([
    {
        "Pengujian": "Pengujian 1",
        "Minimum Support": 0.005,
        "Minimum Confidence": 0.10,
        "Frequent Itemsets": 125,
        "Aturan Asosiasi": 86
    },
    {
        "Pengujian": "Pengujian 2",
        "Minimum Support": 0.010,
        "Minimum Confidence": 0.20,
        "Frequent Itemsets": 45,
        "Aturan Asosiasi": 20
    },
    {
        "Pengujian": "Pengujian 3",
        "Minimum Support": 0.015,
        "Minimum Confidence": 0.30,
        "Frequent Itemsets": 19,
        "Aturan Asosiasi": 2
    },
    {
        "Pengujian": "Pengujian 4",
        "Minimum Support": 0.020,
        "Minimum Confidence": 0.40,
        "Frequent Itemsets": 11,
        "Aturan Asosiasi": 0
    },
    {
        "Pengujian": "Pengujian 5",
        "Minimum Support": 0.025,
        "Minimum Confidence": 0.50,
        "Frequent Itemsets": 6,
        "Aturan Asosiasi": 0
    }
])

# Lima kandidat paket penjualan hasil penelitian
DATA_KANDIDAT = pd.DataFrame([
    {
        "Kandidat": "Kandidat 1",
        "Kombinasi Produk": (
            "100x3x780 + 100x3x1100 + 100x3x1130"
        ),
        "Support": 0.012216,
        "Confidence": "100%",
        "Lift": 81.857143,
        "Jumlah Transaksi": 14,
        "Jumlah Pelanggan": 2
    },
    {
        "Kandidat": "Kandidat 2",
        "Kombinasi Produk": (
            "80x80x5x520 + 80x80x5x800"
        ),
        "Support": 0.010471,
        "Confidence": "92,31%",
        "Lift": 81.372781,
        "Jumlah Transaksi": 12,
        "Jumlah Pelanggan": 3
    },
    {
        "Kandidat": "Kandidat 3",
        "Kombinasi Produk": (
            "100x100x8x200 + 100x100x8x250"
        ),
        "Support": 0.015707,
        "Confidence": "54,55% - 78,26%",
        "Lift": 27.177866,
        "Jumlah Transaksi": 18,
        "Jumlah Pelanggan": 1
    },
    {
        "Kandidat": "Kandidat 4",
        "Kombinasi Produk": (
            "50x50x3x125 + 50x50x3x240"
        ),
        "Support": 0.010471,
        "Confidence": "44,44% - 63,16%",
        "Lift": 26.807018,
        "Jumlah Transaksi": 12,
        "Jumlah Pelanggan": 2
    },
    {
        "Kandidat": "Kandidat 5",
        "Kombinasi Produk": (
            "100x100x5x1150 + 100x100x8x250"
        ),
        "Support": 0.011344,
        "Confidence": "39,39% - 46,43%",
        "Lift": 16.123377,
        "Jumlah Transaksi": 13,
        "Jumlah Pelanggan": 1
    }
])


# =========================================================
# 3. FUNGSI FORMAT ANGKA
# =========================================================

def format_ribuan(nilai):
    return f"{nilai:,}".replace(",", ".")


def format_persen(nilai):
    return f"{nilai:.2%}".replace(".", ",")


def format_desimal(nilai):
    return f"{nilai:.6f}".replace(".", ",")


# =========================================================
# 4. SIDEBAR DAN NAVIGASI
# =========================================================

st.sidebar.title("📊 Dashboard PT CKL")
st.sidebar.caption("Analisis Pola Pembelian Produk")

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "🏠 Beranda",
        "📁 Karakteristik Data",
        "📈 Hasil Apriori",
        "📦 Kandidat Paket Penjualan",
        "⚙️ Pengujian Parameter",
        "📊 Grafik dan Analisis"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    """
    **Informasi Penelitian**

    - Metode: KDD
    - Algoritma: Apriori
    - Dataset: Penjualan Tahun 2025
    - Objek: PT CKL

    Dashboard menampilkan hasil pengolahan
    yang telah dilakukan melalui Google Colab.
    """
)


# =========================================================
# 5. HALAMAN BERANDA
# =========================================================

if menu == "🏠 Beranda":

    st.title("Dashboard Analisis Pola Pembelian")
    st.subheader(NAMA_PERUSAHAAN)

    st.write(
        """
        Dashboard ini menyajikan hasil penelitian analisis
        pola pembelian produk menggunakan algoritma Apriori
        berdasarkan data transaksi penjualan PT CKL tahun 2025.

        Hasil analisis digunakan untuk mengidentifikasi
        frequent itemsets, aturan asosiasi, serta kandidat
        paket penjualan berdasarkan pola pembelian pelanggan.
        """
    )

    st.divider()

    st.subheader("Ringkasan Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Data Bersih",
        format_ribuan(DATA_PENELITIAN["data_bersih"])
    )

    col2.metric(
        "Transaksi Unik",
        format_ribuan(DATA_PENELITIAN["jumlah_transaksi"])
    )

    col3.metric(
        "UK Asli",
        format_ribuan(DATA_PENELITIAN["jumlah_item"])
    )

    st.divider()

    st.subheader("Ringkasan Hasil Algoritma Apriori")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Frequent Itemsets",
        PARAMETER_TERPILIH["frequent_itemsets"]
    )

    col2.metric(
        "Aturan Asosiasi",
        PARAMETER_TERPILIH["aturan_asosiasi"]
    )

    col3.metric(
        "Kandidat Paket",
        len(DATA_KANDIDAT)
    )

    st.divider()

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

    st.info(
        """
        Pengujian kedua dipilih karena menghasilkan
        45 frequent itemsets dan 20 aturan asosiasi.
        Hasil tersebut digunakan untuk analisis pola
        pembelian dan penentuan kandidat paket penjualan.
        """
    )

    st.caption(
        "Sumber: Hasil pengolahan data penelitian menggunakan Google Colab, 2026."
    )


# =========================================================
# 6. HALAMAN KARAKTERISTIK DATA
# =========================================================

elif menu == "📁 Karakteristik Data":

    st.title("Karakteristik Data Penelitian")

    st.write(
        """
        Halaman ini menampilkan ringkasan karakteristik
        dataset setelah tahap preprocessing dan transformasi
        data transaksi.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Data Awal",
        format_ribuan(DATA_PENELITIAN["data_awal"])
    )

    col2.metric(
        "Data Dikeluarkan",
        format_ribuan(DATA_PENELITIAN["data_dikeluarkan"])
    )

    col3.metric(
        "Data Bersih",
        format_ribuan(DATA_PENELITIAN["data_bersih"])
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Transaksi Unik",
        format_ribuan(DATA_PENELITIAN["jumlah_transaksi"])
    )

    col2.metric(
        "Jenis UK",
        format_ribuan(DATA_PENELITIAN["jumlah_item"])
    )

    col3.metric(
        "Atribut Dataset",
        DATA_PENELITIAN["jumlah_kolom"]
    )

    st.divider()

    st.subheader("Ringkasan Preprocessing")

    tabel_preprocessing = pd.DataFrame({
        "Keterangan": [
            "Jumlah data awal",
            "Baris TOTAL = 0 yang dikeluarkan",
            "Jumlah data setelah preprocessing",
            "Jumlah transaksi unik",
            "Jumlah UK asli",
            "Nilai kosong setelah preprocessing",
            "Data duplikat identik yang dipertahankan"
        ],
        "Hasil": [
            "1.889 baris",
            "6 baris",
            "1.883 baris",
            "1.146 transaksi",
            "422 item",
            "0",
            "1 kejadian"
        ]
    })

    st.dataframe(
        tabel_preprocessing,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Karakteristik Jumlah Item per Transaksi")

    tabel_karakteristik = pd.DataFrame({
        "Statistik": [
            "Rata-rata",
            "Standar deviasi",
            "Minimum",
            "Kuartil pertama (Q1)",
            "Median (Q2)",
            "Kuartil ketiga (Q3)",
            "Maksimum"
        ],
        "Jumlah Jenis Item": [
            "1,63",
            "1,21",
            "1",
            "1",
            "1",
            "2",
            "12"
        ]
    })

    st.dataframe(
        tabel_karakteristik,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Hasil Transformasi Data")

    st.metric(
        "Ukuran Matriks Transaksi",
        DATA_PENELITIAN["ukuran_matriks"]
    )

    st.write(
        """
        Matriks transaksi menggunakan representasi biner.
        Nilai 1 menunjukkan bahwa UK terdapat dalam transaksi,
        sedangkan nilai 0 menunjukkan bahwa UK tidak terdapat
        dalam transaksi.
        """
    )


# =========================================================
# 7. HALAMAN HASIL APRIORI
# =========================================================

elif menu == "📈 Hasil Apriori":

    st.title("Hasil Analisis Algoritma Apriori")

    st.write(
        """
        Hasil berikut merupakan ringkasan pengujian kedua,
        yaitu konfigurasi minimum support 0,010 dan
        minimum confidence 0,20.
        """
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Frequent Itemsets",
        PARAMETER_TERPILIH["frequent_itemsets"]
    )

    col2.metric(
        "Aturan Asosiasi",
        PARAMETER_TERPILIH["aturan_asosiasi"]
    )

    st.divider()

    st.subheader("Ringkasan Evaluasi Aturan Asosiasi")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rata-rata Support",
        format_persen(PARAMETER_TERPILIH["rata_support"])
    )

    col2.metric(
        "Rata-rata Confidence",
        format_persen(PARAMETER_TERPILIH["rata_confidence"])
    )

    col3.metric(
        "Rata-rata Lift",
        format_desimal(PARAMETER_TERPILIH["rata_lift"])
    )

    st.divider()

    st.subheader("Interpretasi Hasil")

    st.markdown(
        """
        - **Support** menggambarkan proporsi transaksi yang
          memuat kombinasi produk.
        - **Confidence** menggambarkan tingkat kepercayaan
          aturan asosiasi berdasarkan arah aturan.
        - **Lift** menunjukkan kekuatan keterkaitan produk
          dibandingkan dengan kemunculan secara independen.
        """
    )

    st.info(
        """
        Hasil pengujian kedua menghasilkan 20 aturan asosiasi.
        Rata-rata support sebesar 0,012129, rata-rata confidence
        sebesar 0,855423, dan rata-rata lift sebesar 64,262390.
        """
    )

    st.caption(
        "Catatan: halaman ini menampilkan ringkasan metrik hasil Colab. "
        "Rincian 20 baris aturan asosiasi dapat ditambahkan setelah "
        "tabel lengkap hasil Colab dimasukkan ke dalam kode."
    )


# =========================================================
# 8. HALAMAN KANDIDAT PAKET PENJUALAN
# =========================================================

elif menu == "📦 Kandidat Paket Penjualan":

    st.title("Kandidat Paket Penjualan")

    st.write(
        """
        Lima kandidat paket penjualan berikut diperoleh
        dari pengelompokan aturan asosiasi pada pengujian
        kedua algoritma Apriori.
        """
    )

    st.dataframe(
        DATA_KANDIDAT,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Detail Kandidat Paket")

    pilihan = st.selectbox(
        "Pilih kandidat paket",
        DATA_KANDIDAT["Kandidat"].tolist()
    )

    detail = DATA_KANDIDAT[
        DATA_KANDIDAT["Kandidat"] == pilihan
    ].iloc[0]

    st.markdown(f"### {detail['Kandidat']}")

    st.write(
        f"**Kombinasi Produk:** {detail['Kombinasi Produk']}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Support",
        format_persen(detail["Support"])
    )

    col2.metric(
        "Confidence",
        detail["Confidence"]
    )

    col3.metric(
        "Lift",
        format_desimal(detail["Lift"])
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Jumlah Transaksi",
        int(detail["Jumlah Transaksi"])
    )

    col2.metric(
        "Jumlah Pelanggan",
        int(detail["Jumlah Pelanggan"])
    )

    if detail["Jumlah Pelanggan"] == 1:
        st.warning(
            "Kombinasi ini ditemukan pada satu pelanggan. "
            "Pertimbangkan kebutuhan pelanggan tersebut "
            "sebelum menjadikannya penawaran umum."
        )
    else:
        st.success(
            "Kombinasi ini ditemukan pada lebih dari satu pelanggan."
        )

    st.caption(
        "Kandidat paket merupakan rekomendasi berdasarkan pola "
        "transaksi historis, bukan paket yang sudah diterapkan."
    )


# =========================================================
# 9. HALAMAN PENGUJIAN PARAMETER
# =========================================================

elif menu == "⚙️ Pengujian Parameter":

    st.title("Hasil Pengujian Parameter Apriori")

    st.write(
        """
        Pengujian dilakukan menggunakan lima konfigurasi
        minimum support dan minimum confidence.
        """
    )

    tabel_pengujian = DATA_PENGUJIAN.copy()

    tabel_pengujian["Minimum Support"] = (
        tabel_pengujian["Minimum Support"].map(
            lambda x: f"{x:.3f}".replace(".", ",")
        )
    )

    tabel_pengujian["Minimum Confidence"] = (
        tabel_pengujian["Minimum Confidence"].map(
            lambda x: f"{x:.2f}".replace(".", ",")
        )
    )

    st.dataframe(
        tabel_pengujian,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Perbandingan Frequent Itemsets")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(
        DATA_PENGUJIAN["Pengujian"],
        DATA_PENGUJIAN["Frequent Itemsets"]
    )

    ax.set_xlabel("Konfigurasi Pengujian")
    ax.set_ylabel("Jumlah Frequent Itemsets")
    ax.set_title("Jumlah Frequent Itemsets pada Lima Pengujian")

    plt.xticks(rotation=20)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Perbandingan Aturan Asosiasi")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(
        DATA_PENGUJIAN["Pengujian"],
        DATA_PENGUJIAN["Aturan Asosiasi"]
    )

    ax.set_xlabel("Konfigurasi Pengujian")
    ax.set_ylabel("Jumlah Aturan Asosiasi")
    ax.set_title("Jumlah Aturan Asosiasi pada Lima Pengujian")

    plt.xticks(rotation=20)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Parameter yang Dipilih")

    st.success(
        """
        Pengujian kedua dipilih menggunakan minimum support
        0,010 dan minimum confidence 0,20. Konfigurasi ini
        menghasilkan 45 frequent itemsets dan 20 aturan asosiasi.
        """
    )

    st.caption(
        "Minimum support dan minimum confidence dinaikkan "
        "secara bersamaan. Oleh karena itu, hasil pengujian "
        "menggambarkan perubahan gabungan kedua parameter."
    )


# =========================================================
# 10. HALAMAN GRAFIK DAN ANALISIS
# =========================================================

elif menu == "📊 Grafik dan Analisis":

    st.title("Grafik dan Analisis Hasil Penelitian")

    st.write(
        """
        Halaman ini menyajikan visualisasi hasil penelitian
        berdasarkan lima kandidat paket penjualan dan
        hasil pengujian algoritma Apriori.
        """
    )

    st.divider()

    st.subheader("Jumlah Transaksi pada Setiap Kandidat")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        DATA_KANDIDAT["Kandidat"],
        DATA_KANDIDAT["Jumlah Transaksi"]
    )

    ax.set_xlabel("Jumlah Transaksi")
    ax.set_ylabel("Kandidat Paket")
    ax.set_title("Jumlah Transaksi per Kandidat Paket")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Jumlah Pelanggan pada Setiap Kandidat")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        DATA_KANDIDAT["Kandidat"],
        DATA_KANDIDAT["Jumlah Pelanggan"]
    )

    ax.set_xlabel("Kandidat Paket")
    ax.set_ylabel("Jumlah Pelanggan")
    ax.set_title("Persebaran Pelanggan pada Kandidat Paket")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Perbandingan Nilai Lift")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.barh(
        DATA_KANDIDAT["Kandidat"],
        DATA_KANDIDAT["Lift"]
    )

    ax.set_xlabel("Nilai Lift")
    ax.set_ylabel("Kandidat Paket")
    ax.set_title("Nilai Lift pada Lima Kandidat Paket")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Kesimpulan Visualisasi")

    st.markdown(
        """
        - Kandidat 1 memiliki nilai confidence dan lift tertinggi.
        - Kandidat 3 memiliki support dan jumlah transaksi tertinggi.
        - Kandidat 2 memiliki jumlah pelanggan terbanyak.
        - Seluruh kandidat memiliki nilai lift lebih dari 1.
        """
    )

    st.info(
        """
        Hasil visualisasi menunjukkan bahwa setiap kandidat
        memiliki keunggulan yang berbeda. Penentuan paket yang
        akan diterapkan tetap perlu mempertimbangkan kebutuhan
        pelanggan dan kondisi operasional perusahaan.
        """
    )


# =========================================================
# 11. FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Dashboard Penelitian Skripsi"
)

st.sidebar.caption(
    "PT Cakrawala Kemasindo Lestari"
)

st.sidebar.caption(
    "Hasil Analisis Data Penjualan Tahun 2025"
)
