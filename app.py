# =========================================================
# MENU 3: PENGUJIAN 5 ITERASI & PERBANDINGAN KINERJA
# =========================================================
elif menu_pilihan == "⚖️ Pengujian 5 Iterasi":
    st.title("⚖️ Hasil Pengujian 5 Iterasi Komparatif")
    st.markdown("Berikut adalah tabel rincian hasil pengujian komparatif 5 iterasi berdasarkan eksperimen Google Colab.")
    st.markdown("---")

    min_supp = st.session_state.min_supp
    min_conf = st.session_state.min_conf

    # Data waktu iterasi disesuaikan dengan hasil Colab terbaru Anda
    df_iter = pd.DataFrame([
        {'Pengujian': 1, 'Waktu Apriori (detik)': 0.017728, 'Waktu FP-Growth (detik)': 0.117712, 'Frequent Itemsets Apriori': 126.0, 'Frequent Itemsets FP-Growth': 126.0, 'Association Rules Apriori': 86.0, 'Association Rules FP-Growth': 86.0},
        {'Pengujian': 2, 'Waktu Apriori (detik)': 0.018514, 'Waktu FP-Growth (detik)': 0.175241, 'Frequent Itemsets Apriori': 126.0, 'Frequent Itemsets FP-Growth': 126.0, 'Association Rules Apriori': 86.0, 'Association Rules FP-Growth': 86.0},
        {'Pengujian': 3, 'Waktu Apriori (detik)': 0.019864, 'Waktu FP-Growth (detik)': 0.147157, 'Frequent Itemsets Apriori': 126.0, 'Frequent Itemsets FP-Growth': 126.0, 'Association Rules Apriori': 86.0, 'Association Rules FP-Growth': 86.0},
        {'Pengujian': 4, 'Waktu Apriori (detik)': 0.015161, 'Waktu FP-Growth (detik)': 0.094222, 'Frequent Itemsets Apriori': 126.0, 'Frequent Itemsets FP-Growth': 126.0, 'Association Rules Apriori': 86.0, 'Association Rules FP-Growth': 86.0},
        {'Pengujian': 5, 'Waktu Apriori (detik)': 0.011560, 'Waktu FP-Growth (detik)': 0.107394, 'Frequent Itemsets Apriori': 126.0, 'Frequent Itemsets FP-Growth': 126.0, 'Association Rules Apriori': 86.0, 'Association Rules FP-Growth': 86.0}
    ])

    summary_df = pd.DataFrame([
        {
            'Pengujian': 'Rata-rata',
            'Waktu Apriori (detik)': 0.016565,
            'Waktu FP-Growth (detik)': 0.128345,
            'Frequent Itemsets Apriori': 126.0,
            'Frequent Itemsets FP-Growth': 126.0,
            'Association Rules Apriori': 86.0,
            'Association Rules FP-Growth': 86.0
        },
        {
            'Pengujian': 'Standar Deviasi',
            'Waktu Apriori (detik)': 0.003281,
            'Waktu FP-Growth (detik)': 0.032671,
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
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f"{yval:.4f}s", ha='center', va='bottom', fontsize=7, fontweight='bold')

        for bar in bars2:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f"{yval:.4f}s", ha='center', va='bottom', fontsize=7, fontweight='bold')

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
