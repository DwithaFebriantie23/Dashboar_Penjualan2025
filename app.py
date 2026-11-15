
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules

# =========================================================
# KONFIGURASI DASHBOARD
# =========================================================
st.set_page_config(
    page_title="Dashboard Analisis Apriori - PT CKL",
    page_icon="📊",
    layout="wide"
)

NAMA_PERUSAHAAN = "PT Cakrawala Kemasindo Lestari"
MIN_SUPPORT_FINAL = 0.010
MIN_CONFIDENCE_FINAL = 0.20

KONFIGURASI_PENGUJIAN = [
    {"Pengujian": 1, "Minimum Support": 0.005, "Minimum Confidence": 0.10},
    {"Pengujian": 2, "Minimum Support": 0.010, "Minimum Confidence": 0.20},
    {"Pengujian": 3, "Minimum Support": 0.015, "Minimum Confidence": 0.30},
    {"Pengujian": 4, "Minimum Support": 0.020, "Minimum Confidence": 0.40},
    {"Pengujian": 5, "Minimum Support": 0.025, "Minimum Confidence": 0.50},
]

KANDIDAT_PAKET = [
    {
        "Kandidat": "Kandidat 1",
        "Produk": ["100x3x780", "100x3x1100", "100x3x1130"],
    },
    {
        "Kandidat": "Kandidat 2",
        "Produk": ["80x80x5x520", "80x80x5x800"],
    },
    {
        "Kandidat": "Kandidat 3",
        "Produk": ["100x100x8x200", "100x100x8x250"],
    },
    {
        "Kandidat": "Kandidat 4",
        "Produk": ["50x50x3x125", "50x50x3x240"],
    },
    {
        "Kandidat": "Kandidat 5",
        "Produk": ["100x100x5x1150", "100x100x8x250"],
    },
]


# =========================================================
# FUNGSI PREPROCESSING DAN TRANSFORMASI
# ITEM dipertahankan sebagai UK asli, tanpa kode label.
# =========================================================
def ubah_id_ke_teks(nilai):
    if isinstance(nilai, (datetime, pd.Timestamp)):
        return nilai.strftime("%Y-%m-%d")
    return str(nilai).strip()


@st.cache_data(show_spinner=False)
def load_and_prepare_data(file_bytes=None):
    if file_bytes is not None:
        df_raw = pd.read_excel(
            BytesIO(file_bytes),
            sheet_name="Rekap 2025",
            header=1
        )
    else:
        file_path = Path(__file__).resolve().parent / "Penjualan2025.xlsx"
        if not file_path.exists():
            raise FileNotFoundError(
                "File Penjualan2025.xlsx tidak ditemukan. "
                "Letakkan file Excel di folder yang sama dengan app.py "
                "atau unggah melalui menu di sidebar."
            )
        df_raw = pd.read_excel(
            file_path,
            sheet_name="Rekap 2025",
            header=1
        )

    kolom_wajib = [
        "TANGGAL", "CUSTOMER", "ID TRANSAKSI",
        "ITEM", "QTY", "HARGA", "TOTAL"
    ]

    kolom_hilang = [
        kolom for kolom in kolom_wajib
        if kolom not in df_raw.columns
    ]

    if kolom_hilang:
        raise ValueError(
            f"Kolom dataset tidak lengkap: {kolom_hilang}"
        )

    # Enam baris dengan TOTAL = 0 dikeluarkan.
    df_clean = df_raw.loc[df_raw["TOTAL"] != 0].copy()

    # Normalisasi ID transaksi.
    df_clean["ID TRANSAKSI"] = (
        df_clean["ID TRANSAKSI"].apply(ubah_id_ke_teks)
    )

    # Menggunakan UK asli tanpa kode label buatan.
    df_clean["ITEM"] = (
        df_clean["ITEM"].astype(str).str.strip()
    )

    # Normalisasi nama pelanggan untuk menghitung pelanggan unik.
    df_clean["CUSTOMER_NORMALIZED"] = (
        df_clean["CUSTOMER"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )

    df_clean["TANGGAL"] = pd.to_datetime(
        df_clean["TANGGAL"],
        errors="coerce"
    )

    df_clean["QTY"] = pd.to_numeric(
        df_clean["QTY"],
        errors="coerce"
    ).fillna(0)

    df_clean["HARGA"] = pd.to_numeric(
        df_clean["HARGA"],
        errors="coerce"
    )

    df_clean["TOTAL"] = pd.to_numeric(
        df_clean["TOTAL"],
        errors="coerce"
    ).fillna(0)

    # Membentuk matriks transaksi berdasarkan ID TRANSAKSI dan ITEM.
    basket_matrix = (
        df_clean
        .groupby(["ID TRANSAKSI", "ITEM"], sort=False)
        .size()
        .unstack(fill_value=0)
    )

    # Mengubah nilai menjadi matriks biner (0 dan 1).
    basket_matrix = (basket_matrix > 0).astype(int)

    return df_raw, df_clean, basket_matrix


# =========================================================
# FUNGSI ALGORITMA APRIORI
# =========================================================
@st.cache_data(show_spinner=False)
def jalankan_apriori(basket_matrix, min_support, min_confidence):
    basket_bool = basket_matrix.astype(bool)

    frequent_itemsets = apriori(
        basket_bool,
        min_support=min_support,
        use_colnames=True
    )

    if len(frequent_itemsets) >= 2:
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )
    else:
        rules = pd.DataFrame()

    return frequent_itemsets, rules


# =========================================================
# FUNGSI PENDUKUNG
# =========================================================
def format_produk(itemset):
    return " + ".join(sorted(list(itemset)))


def kategori_rekomendasi(jumlah_pelanggan):
    if jumlah_pelanggan == 1:
        return "Khusus pelanggan"
    if jumlah_pelanggan >= 3:
        return "Lintas pelanggan"
    return "Lintas pelanggan (terbatas)"

# =========================================================
# FUNGSI MEMBENTUK TABEL FREQUENT ITEMSETS
# =========================================================
def buat_tabel_itemsets(frequent_itemsets):
    if frequent_itemsets.empty:
        return pd.DataFrame(
            columns=["Produk", "Jumlah Produk", "Support"]
        )

    hasil = frequent_itemsets.copy()

    hasil["Produk"] = hasil["itemsets"].apply(format_produk)
    hasil["Jumlah Produk"] = hasil["itemsets"].apply(len)
    hasil["Support"] = hasil["support"]

    hasil = hasil[
        ["Produk", "Jumlah Produk", "Support"]
    ].sort_values(
        by=["Support", "Jumlah Produk"],
        ascending=[False, False]
    )

    return hasil.reset_index(drop=True)


# =========================================================
# FUNGSI MEMBENTUK TABEL ATURAN ASOSIASI
# =========================================================
def buat_tabel_aturan(rules):
    kolom = [
        "Antecedents",
        "Consequents",
        "Support",
        "Confidence",
        "Lift"
    ]

    if rules.empty:
        return pd.DataFrame(columns=kolom)

    hasil = rules.copy()

    hasil["Antecedents"] = hasil["antecedents"].apply(
        format_produk
    )

    hasil["Consequents"] = hasil["consequents"].apply(
        format_produk
    )

    hasil = hasil[
        [
            "Antecedents",
            "Consequents",
            "support",
            "confidence",
            "lift"
        ]
    ].rename(
        columns={
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift"
        }
    )

    hasil = hasil.sort_values(
        by=["Lift", "Confidence"],
        ascending=False
    )

    return hasil.reset_index(drop=True)


# =========================================================
# FUNGSI MENGHITUNG JUMLAH TRANSAKSI DAN PELANGGAN
# =========================================================
def hitung_transaksi_dan_pelanggan(df_clean, daftar_produk):
    transaksi_produk = (
        df_clean.groupby("ID TRANSAKSI")["ITEM"]
        .apply(set)
    )

    transaksi_yang_memuat_semua_produk = transaksi_produk[
        transaksi_produk.apply(
            lambda item: set(daftar_produk).issubset(item)
        )
    ]

    daftar_id = transaksi_yang_memuat_semua_produk.index

    data_transaksi = df_clean[
        df_clean["ID TRANSAKSI"].isin(daftar_id)
    ]

    jumlah_transaksi = len(daftar_id)

    jumlah_pelanggan = (
        data_transaksi["CUSTOMER_NORMALIZED"]
        .dropna()
        .nunique()
    )

    return jumlah_transaksi, jumlah_pelanggan


# =========================================================
# FUNGSI MEMBENTUK TABEL KANDIDAT PAKET PENJUALAN
# =========================================================
def buat_tabel_kandidat(
    df_clean,
    basket_matrix,
    rules
):
    daftar_hasil = []
    jumlah_total_transaksi = len(basket_matrix)

    for kandidat in KANDIDAT_PAKET:
        produk = kandidat["Produk"]
        himpunan_produk = set(produk)

        # Memastikan seluruh produk tersedia dalam dataset.
        produk_tersedia = all(
            item in basket_matrix.columns
            for item in produk
        )

        if not produk_tersedia:
            continue

        # Menghitung jumlah transaksi yang memuat seluruh produk.
        jumlah_transaksi, jumlah_pelanggan = (
            hitung_transaksi_dan_pelanggan(
                df_clean,
                produk
            )
        )

        support_paket = (
            jumlah_transaksi / jumlah_total_transaksi
            if jumlah_total_transaksi > 0 else 0
        )

        # Mengambil aturan yang berasal dari kombinasi produk tersebut.
        aturan_kandidat = pd.DataFrame()

        if not rules.empty:
            aturan_kandidat = rules[
                rules.apply(
                    lambda baris: (
                        set(baris["antecedents"])
                        | set(baris["consequents"])
                    ) == himpunan_produk,
                    axis=1
                )
            ]

        if aturan_kandidat.empty:
            confidence = np.nan
            lift = np.nan
            jumlah_aturan = 0
        else:
            confidence_min = aturan_kandidat["confidence"].min()
            confidence_max = aturan_kandidat["confidence"].max()

            if np.isclose(confidence_min, confidence_max):
                confidence = confidence_max
            else:
                confidence = (
                    f"{confidence_min:.2%} - "
                    f"{confidence_max:.2%}"
                )

            lift = aturan_kandidat["lift"].max()
            jumlah_aturan = len(aturan_kandidat)

        daftar_hasil.append({
            "Kandidat": kandidat["Kandidat"],
            "Kombinasi Produk": " + ".join(produk),
            "Jumlah Produk": len(produk),
            "Jumlah Transaksi": jumlah_transaksi,
            "Jumlah Pelanggan": jumlah_pelanggan,
            "Support": support_paket,
            "Confidence": confidence,
            "Lift": lift,
            "Jumlah Aturan": jumlah_aturan,
            "Kategori Rekomendasi": kategori_rekomendasi(
                jumlah_pelanggan
            )
        })

    hasil = pd.DataFrame(daftar_hasil)

    if not hasil.empty:
        hasil = hasil.sort_values(
            by="Jumlah Transaksi",
            ascending=False
        ).reset_index(drop=True)

    return hasil


# =========================================================
# FUNGSI FORMAT ANGKA
# =========================================================
def format_persen(nilai):
    if pd.isna(nilai):
        return "-"

    return f"{nilai:.2%}"


def format_angka(nilai):
    if pd.isna(nilai):
        return "-"

    return f"{nilai:.4f}"


def format_waktu(nilai):
    return f"{nilai:.6f} detik"


# =========================================================
# FUNGSI MEMBUAT DATA RINGKASAN
# =========================================================
def buat_ringkasan_data(df_raw, df_clean, basket_matrix):
    return {
        "Jumlah Data Awal": len(df_raw),
        "Jumlah Data Bersih": len(df_clean),
        "Jumlah Transaksi": basket_matrix.shape[0],
        "Jumlah UK Asli": basket_matrix.shape[1],
        "Jumlah Pelanggan": (
            df_clean["CUSTOMER_NORMALIZED"].dropna().nunique()
        ),
        "Jumlah Baris Dikeluarkan": (
            len(df_raw) - len(df_clean)
        )
    }


# =========================================================
# MEMUAT DATASET
# =========================================================
st.sidebar.title("📊 Dashboard PT CKL")
st.sidebar.caption("Analisis Pola Pembelian dengan Algoritma Apriori")

file_upload = st.sidebar.file_uploader(
    "Unggah Data Penjualan 2025",
    type=["xlsx"]
)

try:
    if file_upload is not None:
        df_raw, df_clean, basket_matrix = load_and_prepare_data(
            file_upload.getvalue()
        )
    else:
        df_raw, df_clean, basket_matrix = load_and_prepare_data()

except Exception as e:
    st.error(f"Data belum dapat dimuat: {e}")
    st.info(
        "Pastikan file Excel menggunakan nama sheet "
        "'Rekap 2025' dan memiliki struktur kolom yang sesuai."
    )
    st.stop()


# =========================================================
# MENJALANKAN APRIORI DENGAN PARAMETER TERPILIH
# =========================================================
with st.spinner("Sedang memproses algoritma Apriori..."):
    frequent_itemsets, rules = jalankan_apriori(
        basket_matrix,
        MIN_SUPPORT_FINAL,
        MIN_CONFIDENCE_FINAL
    )

tabel_itemsets = buat_tabel_itemsets(frequent_itemsets)
tabel_aturan = buat_tabel_aturan(rules)

tabel_kandidat = buat_tabel_kandidat(
    df_clean,
    basket_matrix,
    rules
)

ringkasan = buat_ringkasan_data(
    df_raw,
    df_clean,
    basket_matrix
)


# =========================================================
# NAVIGASI DASHBOARD
# =========================================================
menu = st.sidebar.radio(
    "Menu Dashboard",
    [
        "Beranda",
        "Karakteristik Data",
        "Hasil Apriori",
        "Kandidat Paket Penjualan",
        "Pengujian Parameter",
        "Grafik dan Filter Penjualan"
    ]
)

st.sidebar.divider()
st.sidebar.caption("PT Cakrawala Kemasindo Lestari")
st.sidebar.caption("Data Penjualan Tahun 2025")


# =========================================================
# HALAMAN 1: BERANDA
# =========================================================
if menu == "Beranda":

    st.title("Dashboard Analisis Pola Pembelian")
    st.subheader(NAMA_PERUSAHAAN)

    st.write(
        """
        Dashboard ini digunakan untuk menampilkan hasil analisis
        pola pembelian produk berdasarkan data transaksi penjualan
        tahun 2025 menggunakan algoritma Apriori.
        
        Hasil analisis diharapkan dapat membantu perusahaan
        mengidentifikasi kombinasi produk yang berpotensi
        dikembangkan menjadi paket penjualan.
        """
    )

    st.divider()

    st.subheader("Ringkasan Data Penelitian")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Data Penjualan Bersih",
        f"{ringkasan['Jumlah Data Bersih']:,}"
    )

    col2.metric(
        "Jumlah Transaksi",
        f"{ringkasan['Jumlah Transaksi']:,}"
    )

    col3.metric(
        "Jenis UK Asli",
        f"{ringkasan['Jumlah UK Asli']:,}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Jumlah Pelanggan",
        f"{ringkasan['Jumlah Pelanggan']:,}"
    )

    col5.metric(
        "Frequent Itemsets",
        f"{len(frequent_itemsets):,}"
    )

    col6.metric(
        "Aturan Asosiasi",
        f"{len(rules):,}"
    )

    st.divider()

    st.subheader("Parameter Apriori Terpilih")

    col1, col2 = st.columns(2)

    col1.metric(
        "Minimum Support",
        f"{MIN_SUPPORT_FINAL:.3f}"
    )

    col2.metric(
        "Minimum Confidence",
        f"{MIN_CONFIDENCE_FINAL:.2f}"
    )

    st.info(
        "Konfigurasi kedua dipilih sebagai konfigurasi yang "
        "memberikan keseimbangan antara jumlah frequent itemsets "
        "dan jumlah aturan asosiasi yang dihasilkan."
    )

    st.divider()

    st.subheader("Alur Analisis")

    st.markdown(
        """
        1. Memuat data penjualan tahun 2025.
        2. Melakukan preprocessing data.
        3. Membentuk matriks transaksi.
        4. Menerapkan algoritma Apriori.
        5. Membentuk aturan asosiasi.
        6. Mengidentifikasi kandidat paket penjualan.
        """
    )


# =========================================================
# HALAMAN 2: KARAKTERISTIK DATA
# =========================================================
elif menu == "Karakteristik Data":

    st.title("Karakteristik Data Penjualan")

    st.write(
        "Halaman ini menampilkan kondisi dataset setelah "
        "proses preprocessing."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Data Awal",
        f"{ringkasan['Jumlah Data Awal']:,}"
    )

    col2.metric(
        "Data Bersih",
        f"{ringkasan['Jumlah Data Bersih']:,}"
    )

    col3.metric(
        "Data Dikeluarkan",
        f"{ringkasan['Jumlah Baris Dikeluarkan']:,}"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Transaksi Unik",
        f"{basket_matrix.shape[0]:,}"
    )

    col2.metric(
        "UK Asli",
        f"{basket_matrix.shape[1]:,}"
    )

    col3.metric(
        "Pelanggan Unik",
        f"{ringkasan['Jumlah Pelanggan']:,}"
    )

    st.divider()

    st.subheader("Informasi Dataset")

    st.write("Kolom yang digunakan dalam penelitian:")

    st.dataframe(
        df_clean[
            [
                "TANGGAL",
                "CUSTOMER",
                "ID TRANSAKSI",
                "ITEM",
                "QTY",
                "HARGA",
                "TOTAL"
            ]
        ].head(20),
        use_container_width=True
    )

    st.caption(
        "Tabel menampilkan 20 baris pertama dari data bersih."
    )

    st.divider()

    st.subheader("Daftar UK Asli")

    daftar_item = pd.DataFrame({
        "UK Asli": sorted(basket_matrix.columns.tolist())
    })

    st.dataframe(
        daftar_item,
        use_container_width=True,
        height=350
    )

    st.divider()

    st.subheader("Distribusi Jumlah Jenis Produk per Transaksi")

    jumlah_item_transaksi = (
        basket_matrix.sum(axis=1)
    )

    frekuensi_item = (
        jumlah_item_transaksi
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(
        frekuensi_item.index.astype(str),
        frekuensi_item.values
    )

    ax.set_xlabel("Jumlah Jenis UK dalam Transaksi")
    ax.set_ylabel("Jumlah Transaksi")
    ax.set_title("Distribusi Jenis Produk per Transaksi")

    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# HALAMAN 3: HASIL APRIORI
# =========================================================
elif menu == "Hasil Apriori":

    st.title("Hasil Analisis Algoritma Apriori")

    st.write(
        f"""
        Proses Apriori menggunakan minimum support
        **{MIN_SUPPORT_FINAL:.3f}** dan minimum confidence
        **{MIN_CONFIDENCE_FINAL:.2f}**.
        """
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Frequent Itemsets",
        f"{len(frequent_itemsets):,}"
    )

    col2.metric(
        "Aturan Asosiasi",
        f"{len(rules):,}"
    )

    st.divider()

    st.subheader("Frequent Itemsets")

    if tabel_itemsets.empty:
        st.warning("Tidak ditemukan frequent itemsets.")
    else:
        st.dataframe(
            tabel_itemsets,
            use_container_width=True,
            height=400
        )

        csv_itemsets = tabel_itemsets.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Unduh Frequent Itemsets (CSV)",
            data=csv_itemsets,
            file_name="frequent_itemsets_apriori.csv",
            mime="text/csv"
        )

    st.divider()

    st.subheader("Aturan Asosiasi")

    if tabel_aturan.empty:
        st.warning("Tidak ditemukan aturan asosiasi.")
    else:
        st.dataframe(
            tabel_aturan,
            use_container_width=True,
            height=400
        )

        csv_rules = tabel_aturan.to_csv(
            index=False
        ).encode("utf-8-sig")

        st.download_button(
            "Unduh Aturan Asosiasi (CSV)",
            data=csv_rules,
            file_name="aturan_asosiasi_apriori.csv",
            mime="text/csv"
        )

    st.caption(
        "Support menunjukkan proporsi transaksi yang memuat "
        "kombinasi produk. Confidence menunjukkan tingkat "
        "kemunculan konsekuen ketika anteseden terjadi. "
        "Lift lebih dari 1 menunjukkan asosiasi positif."
    )


# =========================================================
# HALAMAN 4: KANDIDAT PAKET PENJUALAN
# =========================================================
elif menu == "Kandidat Paket Penjualan":

    st.title("Kandidat Paket Penjualan")

    st.write(
        """
        Halaman ini menampilkan lima kombinasi produk yang
        diidentifikasi dari hasil aturan asosiasi Apriori.
        Kombinasi tersebut merupakan kandidat yang dapat
        dipertimbangkan perusahaan sebagai paket penjualan.
        """
    )

    if tabel_kandidat.empty:
        st.warning(
            "Kandidat paket belum dapat ditampilkan. "
            "Periksa kembali kesesuaian UK produk dalam dataset."
        )

    else:
        st.dataframe(
            tabel_kandidat,
            use_container_width=True,
            height=350
        )

        st.divider()

        st.subheader("Detail Kandidat Paket")

        pilihan_kandidat = st.selectbox(
            "Pilih kandidat paket",
            tabel_kandidat["Kandidat"].tolist()
        )

        detail = tabel_kandidat[
            tabel_kandidat["Kandidat"] == pilihan_kandidat
        ].iloc[0]

        st.markdown(
            f"### {detail['Kandidat']}"
        )

        st.write(
            f"**Kombinasi Produk:** {detail['Kombinasi Produk']}"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Jumlah Transaksi",
            f"{int(detail['Jumlah Transaksi']):,}"
        )

        col2.metric(
            "Jumlah Pelanggan",
            f"{int(detail['Jumlah Pelanggan']):,}"
        )

        col3.metric(
            "Support",
            format_persen(detail["Support"])
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Confidence",
            (
                detail["Confidence"]
                if isinstance(detail["Confidence"], str)
                else format_persen(detail["Confidence"])
            )
        )

        col2.metric(
            "Lift",
            format_angka(detail["Lift"])
        )

        col3.metric(
            "Jumlah Aturan",
            f"{int(detail['Jumlah Aturan']):,}"
        )

        st.write(
            f"**Kategori:** {detail['Kategori Rekomendasi']}"
        )

        st.info(
            "Kandidat paket merupakan rekomendasi berdasarkan "
            "pola transaksi historis, bukan paket yang sudah "
            "diterapkan atau terbukti meningkatkan penjualan."
        )


# =========================================================
# HALAMAN 5: PENGUJIAN PARAMETER
# =========================================================
elif menu == "Pengujian Parameter":

    st.title("Pengujian Parameter Algoritma Apriori")

    st.write(
        """
        Pengujian dilakukan menggunakan lima konfigurasi
        minimum support dan minimum confidence. Setiap konfigurasi
        digunakan untuk mengetahui perubahan jumlah frequent
        itemsets dan aturan asosiasi yang dihasilkan.
        """
    )

    hasil_pengujian = []

    progress = st.progress(0)
    status = st.empty()

    for i, konfigurasi in enumerate(KONFIGURASI_PENGUJIAN):

        support = konfigurasi["Minimum Support"]
        confidence = konfigurasi["Minimum Confidence"]

        status.text(
            f"Memproses pengujian {i + 1} dari "
            f"{len(KONFIGURASI_PENGUJIAN)}..."
        )

        itemsets_uji, rules_uji = jalankan_apriori(
            basket_matrix,
            support,
            confidence
        )

        hasil_pengujian.append({
            "Pengujian": konfigurasi["Pengujian"],
            "Minimum Support": support,
            "Minimum Confidence": confidence,
            "Frequent Itemsets": len(itemsets_uji),
            "Aturan Asosiasi": len(rules_uji)
        })

        progress.progress(
            (i + 1) / len(KONFIGURASI_PENGUJIAN)
        )

    status.success("Seluruh pengujian selesai.")

    df_pengujian = pd.DataFrame(hasil_pengujian)

    st.divider()

    st.subheader("Tabel Hasil Pengujian")

    tabel_tampil = df_pengujian.copy()

    tabel_tampil["Minimum Support"] = (
        tabel_tampil["Minimum Support"].map(
            lambda x: f"{x:.3f}"
        )
    )

    tabel_tampil["Minimum Confidence"] = (
        tabel_tampil["Minimum Confidence"].map(
            lambda x: f"{x:.2f}"
        )
    )

    st.dataframe(
        tabel_tampil,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Grafik Jumlah Frequent Itemsets")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(
        df_pengujian["Pengujian"].astype(str),
        df_pengujian["Frequent Itemsets"]
    )

    ax.set_xlabel("Pengujian")
    ax.set_ylabel("Jumlah Frequent Itemsets")
    ax.set_title("Perbandingan Frequent Itemsets")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Grafik Jumlah Aturan Asosiasi")

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.bar(
        df_pengujian["Pengujian"].astype(str),
        df_pengujian["Aturan Asosiasi"]
    )

    ax.set_xlabel("Pengujian")
    ax.set_ylabel("Jumlah Aturan Asosiasi")
    ax.set_title("Perbandingan Aturan Asosiasi")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Konfigurasi Parameter Terpilih")

    st.metric(
        "Konfigurasi Terpilih",
        "Pengujian 2"
    )

    st.write(
        """
        Konfigurasi kedua menggunakan minimum support 0,010
        dan minimum confidence 0,20. Konfigurasi ini menghasilkan
        45 frequent itemsets dan 20 aturan asosiasi, sehingga
        dipilih sebagai konfigurasi yang memberikan keseimbangan
        antara jumlah pola yang ditemukan dan aturan yang dihasilkan.
        """
    )

    st.caption(
        "Catatan: nilai parameter support dan confidence "
        "dinaikkan secara bersamaan pada setiap pengujian. "
        "Dengan demikian, perubahan hasil merupakan pengaruh "
        "gabungan kedua parameter."
    )


# =========================================================
# HALAMAN 6: GRAFIK DAN FILTER PENJUALAN
# =========================================================
elif menu == "Grafik dan Filter Penjualan":

    st.title("Grafik dan Filter Data Penjualan")

    st.write(
        """
        Halaman ini digunakan untuk melihat data penjualan
        berdasarkan periode, pelanggan, dan jenis produk.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # FILTER PELANGGAN
    # -----------------------------------------------------
    st.subheader("Filter Data")

    daftar_pelanggan = sorted(
        df_clean["CUSTOMER"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    pilihan_pelanggan = st.multiselect(
        "Pilih pelanggan",
        options=daftar_pelanggan,
        placeholder="Semua pelanggan"
    )

    daftar_produk = sorted(
        df_clean["ITEM"].dropna().unique().tolist()
    )

    pilihan_produk = st.multiselect(
        "Pilih UK produk",
        options=daftar_produk,
        placeholder="Semua produk"
    )

    data_filter = df_clean.copy()

    if pilihan_pelanggan:
        data_filter = data_filter[
            data_filter["CUSTOMER"].astype(str).isin(
                pilihan_pelanggan
            )
        ]

    if pilihan_produk:
        data_filter = data_filter[
            data_filter["ITEM"].isin(pilihan_produk)
        ]

    st.divider()

    st.subheader("Ringkasan Data Hasil Filter")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Jumlah Baris",
        f"{len(data_filter):,}"
    )

    col2.metric(
        "Jumlah Transaksi",
        f"{data_filter['ID TRANSAKSI'].nunique():,}"
    )

    col3.metric(
        "Jumlah Pelanggan",
        f"{data_filter['CUSTOMER_NORMALIZED'].nunique():,}"
    )

    st.divider()

    # -----------------------------------------------------
    # GRAFIK TREN PENJUALAN BULANAN
    # -----------------------------------------------------
    st.subheader("Tren Penjualan Bulanan")

    data_tanggal = data_filter.dropna(
        subset=["TANGGAL"]
    ).copy()

    if not data_tanggal.empty:

        data_tanggal["Bulan"] = (
            data_tanggal["TANGGAL"]
            .dt.to_period("M")
            .astype(str)
        )

        penjualan_bulanan = (
            data_tanggal.groupby("Bulan")["TOTAL"]
            .sum()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(11, 4))

        ax.plot(
            penjualan_bulanan["Bulan"],
            penjualan_bulanan["TOTAL"],
            marker="o"
        )

        ax.set_xlabel("Bulan")
        ax.set_ylabel("Total Penjualan")
        ax.set_title("Tren Total Penjualan per Bulan")

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning(
            "Tidak ada data tanggal yang dapat ditampilkan."
        )

    st.divider()

    # -----------------------------------------------------
    # GRAFIK PRODUK BERDASARKAN TOTAL QTY
    # -----------------------------------------------------
    st.subheader("Produk Berdasarkan Jumlah Penjualan")

    if not data_filter.empty:

        penjualan_produk = (
            data_filter.groupby("ITEM")["QTY"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .sort_values()
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.barh(
            penjualan_produk.index,
            penjualan_produk.values
        )

        ax.set_xlabel("Total QTY")
        ax.set_ylabel("UK Asli")
        ax.set_title("10 Produk dengan Total QTY Tertinggi")

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning(
            "Tidak ada data yang sesuai dengan filter."
        )

    st.divider()

    # -----------------------------------------------------
    # TABEL DATA HASIL FILTER
    # -----------------------------------------------------
    st.subheader("Data Penjualan Hasil Filter")

    kolom_tampil = [
        "TANGGAL",
        "CUSTOMER",
        "ID TRANSAKSI",
        "ITEM",
        "QTY",
        "HARGA",
        "TOTAL"
    ]

    st.dataframe(
        data_filter[kolom_tampil],
        use_container_width=True,
        height=400
    )

    csv_penjualan = data_filter[
        kolom_tampil
    ].to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "Unduh Data Hasil Filter (CSV)",
        data=csv_penjualan,
        file_name="data_penjualan_hasil_filter.csv",
        mime="text/csv"
    )


# =========================================================
# BAGIAN AKHIR DASHBOARD
# =========================================================
st.sidebar.divider()

st.sidebar.caption(
    "Dashboard Penelitian Skripsi"
)

st.sidebar.caption(
    "Analisis Market Basket menggunakan Apriori"
)

st.sidebar.caption(
    "PT Cakrawala Kemasindo Lestari | 2025"
)
