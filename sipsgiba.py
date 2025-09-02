import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import plotly.express as px
import xlsxwriter
from io import BytesIO
import math

# Konfigurasi halaman
st.set_page_config(
    page_title="SIPSGIBA",
    layout="wide"
)

# Fungsi koneksi database
def get_connection():
    try:
        connection = mysql.connector.connect(
            host="sql12.freesqldatabase.com",
            user="sql12796965",
            password="74L83WK8ZW",
            database="sql12796965",
            port=3306
        )
        return connection
    except Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

# Fungsi login
def login_user(username, password):
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        query = "SELECT * FROM user WHERE username = %s AND password = %s"
        cursor.execute(query, (username, password))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result
    return None

# Inisialisasi session state
if "is_logged_in" not in st.session_state:
    st.session_state.is_logged_in = False
if "df" not in st.session_state:
    st.session_state.df = None
if "df_clustered" not in st.session_state:
    st.session_state.df_clustered = None
if "selected_columns" not in st.session_state:
    st.session_state.selected_columns = None
if "num_clusters" not in st.session_state:
    st.session_state.num_clusters = 3
if "selected_data" not in st.session_state:
    st.session_state.selected_data = None
if "menu" not in st.session_state:
    st.session_state.menu = "menu"
if "df_normalized" not in st.session_state:
    st.session_state.df_normalized = None

def show_data(df):
    st.dataframe(df)

# Fungsi logout
def logout():
    st.session_state.is_logged_in = False
    st.session_state.user = None
    st.session_state.menu = "Beranda"

# Tampilkan hanya jika belum login
if not st.session_state.is_logged_in:
    st.markdown("<h2 style='text-align:center;'>SIPSGIBA</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;'>Sistem Informasi Pemetaan Status Gizi Balita</h3>", unsafe_allow_html=True)

    left, center, right = st.columns([3, 7, 3])
    with center:
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("Username", placeholder="Masukkan Username Anda")
            password = st.text_input("Password", type="password", placeholder="Masukkan Password Anda")
            login_btn = st.form_submit_button("Login")

    if login_btn:
        if not username or not password:
            left, center, right = st.columns([3, 9, 3])
            with center:
                st.warning("Username dan Password tidak boleh kosong!")
        else:
            user = login_user(username, password)
            if user:
                st.session_state.is_logged_in = True
                st.session_state.user = username
                st.session_state.menu = "Beranda"
                st.rerun()  # Refresh halaman untuk menampilkan menu beranda
            else:
                left, center, right = st.columns([3, 9, 3])
            with center:
                st.error("Username atau password salah!")
else:
    # Sidebar: Navigasi (hanya tampil jika sudah login)
    with st.sidebar:
        col1, col2 = st.columns([1, 10])
    with col1:
        st.markdown("### ")  # Ukuran besar, sejajar
    with col2:
        left, center, right = st.columns([10, 9, 10])
        st.write("<h1 style='text-align:center;'>SIPSGIBA</h1>", unsafe_allow_html=True)
        st.markdown("-----")
        menu = option_menu(
        menu_title=None,
        options=["Beranda", "Unggah File", "Perhitungan Clustering", "Diagram Hasil Clustering", "Evaluasi Cluster"],
        icons=["house", "cloud-upload", "calculator", "bar-chart", "check2-circle"],
        default_index=["Beranda", "Unggah File", "Perhitungan Clustering", "Diagram Hasil Clustering", "Evaluasi Cluster"].index(st.session_state.menu),
        orientation="vertical",
        styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": "#000000", "font-size": "14px"},
        "nav-link": {"font-size": "14px", "color": "#262730", "text-align": "left", "margin": "0", "padding": "10px 4px", "font-family": "arial", "background-color": "transparent", "border-radius": "6px"},
        "nav-link-selected": {"background-color": "#FFFFFF", "color": "#000000", "font-weight": "600", "border": "1px solid #CCCCCC", "box-shadow": "0 1px 3px rgba(0, 0, 0, 0.1)", "border-radius": "6px", "padding": "10px 4px"},
    }
)
        st.markdown("-----")
        logout_col1, logout_col2, logout_col3 = st.columns([2, 3, 2])
    with logout_col2:
        if st.button("Logout"):
            logout()
            st.rerun()  # Refresh halaman untuk menampilkan form login

    # Konten Halaman Berdasarkan Menu
    if menu == "Beranda":
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.write("#### Selamat Datang di SIPSGIBA👋")
        st.write("""
        SIPSGIBA adalah singkatan dari Sistem Informasi Pemetaan Status Gizi Balita.
                 
        SIPSGIBA merupakan sebuah sistem informasi berbasis digital yang dirancang untuk mengelola, memantau, dan menganalisis data status gizi balita di suatu wilayah.  
       
        Tujuan utama dari sistem ini adalah membantu tenaga kesehatan dan pemerintah daerah dalam pengambilan keputusan yang lebih tepat sasaran terkait intervensi gizi, penyuluhan kesehatan, serta perencanaan program pencegahan dan penanggulangan masalah gizi.
        """)
    elif menu == "Unggah File":
        st.markdown("### Unggah File CSV")
        st.markdown("*Pastikan data yang diunggah lengkap, terutama pada kolom numerik seperti: Usia (bulan), Berat, Tinggi, ZS BB/U, ZS TB/U, dan ZS BB/TB.*")
    
        uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                df = df.dropna().drop_duplicates().reset_index(drop=True)
                st.session_state.df = df
                st.success("File berhasil diupload!")
                
                st.markdown("### Data Awal")
                with st.expander("Berikut ini adalah data dari hasil file yang telah diunggah", expanded=True):
                    show_data(df)
                    
                all_cols = df.columns.tolist()
                exclude_cols = ["no", "nomor", "id"]
                all_cols = [col for col in all_cols if col.lower() not in exclude_cols]

                if not all_cols:
                    st.error("File CSV tidak mengandung kolom numerik.")
                else:
                    # Normalisasi data menggunakan decimal scaling
                    cols_to_normalize = ['Usia (Bulan)', 'Berat', 'Tinggi']
                    cols_to_normalize = [col for col in cols_to_normalize if col in df.columns]
                    df_normalized = df.copy()
                    for col in cols_to_normalize:
                        if col.lower() in ["no", "nomor"]: 
                            continue
                        max_value = df_normalized[col].max()
                        if max_value != 0:  # Hindari pembagian dengan nol
                            df_normalized[col] = (df_normalized[col] / max_value).round(2)

                        st.session_state.df_normalized = df_normalized
                        st.session_state.selected_columns = all_cols
                    
                    st.markdown("### Data Normalisasi")
                    with st.expander("Berikut ini adalah data dari hasil normalisasi (proses mengubah nilai-nilai data ke skala yang sama)", expanded=True):
                        show_data(st.session_state.df_normalized)
                    
                    
                    st.markdown("### Konfigurasi Clustering")
                    selected_columns = st.multiselect(
                            "Pilih kolom numerik untuk clustering",
                            all_cols,
                            placeholder="Silahkan pilih variabel untuk melanjutkan",
                            key="cols_selector"
                        )
                    st.session_state.selected_columns = selected_columns

                    st.markdown("### Pilih Cluster")    
                    num_clusters = st.slider(
                            "Jumlah cluster (k)",
                            1, 10, 1,
                            key="cluster_slider"
                        )
                    st.session_state.num_clusters = num_clusters

                    # Jumlah data
                    jumlah_data = len(df_normalized)
                    
                    # Hitung titik centroid otomatis berdasarkan rumus (n Data)/(Cluster_i + 1)
                    centroid_positions = []
                    for i in range(num_clusters):
                        posisi = int(jumlah_data / (i + 2))  # Karena rumus: n / (cluster_id + 1)
                        centroid_positions.append(posisi)

                    st.markdown("### Titik Awal Centroid")
                    st.info(f"Jumlah data: {df_normalized.shape[0]}")
                    st.info("Menentukan Titik Awal Centroid: **(n Data) / (n Cluster + 1)**")

                    for i, posisi in enumerate(centroid_positions):
                        st.markdown(f"- **C{i+1}** = {jumlah_data} / ({i+1}+1) = {jumlah_data // (i+2)} → Titik centroid diambil dari data ke-**{posisi}**" )

                    if selected_columns:
                        st.markdown("### Masukkan Titik Awal Centroid")
                        sample_df = df_normalized[selected_columns].copy()
                        st.info("Mohon kurangi 1 saat memilih baris data sebagai titik centroid (karena indeks dimulai dari 0).")
                        
                        centroid_cols = st.columns(num_clusters)
                        selected_data = []
                        
                        for i in range(num_clusters):
                            with centroid_cols[i]:
                                st.markdown(f"### Centroid {i+1}")
                                
                                row_idx = st.selectbox(
                                    f"Pilih baris untuk centroid awal {i+1}",
                                    options=sample_df.index,
                                    format_func=lambda x: f"Baris {x}",
                                    key=f"centroid_{i}"
                                )

                                selected_values = df_normalized.loc[row_idx, selected_columns].values
                                st.write(dict(zip(selected_columns, selected_values)))
                                selected_data.append(selected_values)
                        
                        st.session_state.selected_data = selected_data
                        st.success("Konfigurasi disimpan!")

            except Exception as e:
                st.error(f"Error: {str(e)}")
        elif st.session_state.df is not None:
            show_data(st.session_state.df)
            st.info("File sebelumnya masih tersedia. Upload file baru jika ingin mengganti.")
        else:
            st.warning("Silahkan upload file CSV terlebih dahulu")
        
    elif menu == "Perhitungan Clustering":
        st.markdown("### Proses Clustering dari Data Normalisasi")
        
        if st.session_state.df_normalized is not None:
            with st.expander("Klik tombol 'Jalankan Clustering' untuk mendapatkan hasil Clustering)", expanded=True):
                show_data(st.session_state.df_normalized)
            
        try:
            if st.button("Jalankan Clustering"):
                df = st.session_state.df_normalized
                selected_columns = st.session_state.selected_columns
                selected_data = st.session_state.selected_data
                num_clusters = st.session_state.num_clusters

                # Konversi ke array
                X = df[selected_columns].values
                initial_centroids = np.array(selected_data)

                kmeans = KMeans(
                    n_clusters=num_clusters,
                    init=initial_centroids,
                    random_state=42,
                    n_init=1
                )
                clusters = kmeans.fit_predict(X)

                df_clustered = df.copy()
                df_clustered['Cluster'] = clusters + 1
                st.session_state.df_clustered = df_clustered

                st.success("Clustering berhasil!")

                st.subheader("Hasil Clustering")
                with st.expander("Berikut ini adalah data dari hasil clustering", expanded=True):
                    st.dataframe(df_clustered.sort_values("Cluster"))

                st.subheader("Titik Akhir Centroid")
                with st.expander("Berikut ini adalah data dari titik akhir centroid", expanded=True):
                    cluster_stats = df_clustered.groupby("Cluster")[selected_columns].mean().round(2)
                    st.write(cluster_stats)

                if len(selected_columns) == 2:
                    st.subheader("Visualisasi Cluster")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', edgecolor='k', alpha=0.7)
                    if "Nama" in df.columns:
                        for i, txt in enumerate(df["Nama"]):
                            ax.annotate(txt, (X[i, 0], X[i, 1]), fontsize=8, alpha=0.7)
                            ax.scatter(initial_centroids[:, 0], initial_centroids[:, 1], c='red', s=200, marker='*', label='Centroid Awal', edgecolor='k')
                            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='blue', s=200, marker='X', label='Centroid Akhir', edgecolor='k')
                            ax.set_xlabel(selected_columns[0])
                            ax.set_ylabel(selected_columns[1])
                            ax.set_title("Visualisasi K-Means Clustering")
                            ax.legend()
                            ax.grid(True)
                            st.pyplot(fig)

        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")
        else:
            if not (
            st.session_state.df_normalized is not None and
            st.session_state.selected_columns and
            st.session_state.selected_data
    ):
                st.warning("Silahkan unggah file, konfigurasi clustering, dan pilih centroid terlebih dahulu di menu **Unggah File**.")
                st.markdown("<hr style='margin-top:50px; margin-bottom:10px;'>",unsafe_allow_html=True)
            
    elif menu == "Diagram Hasil Clustering":
        st.markdown("### Hasil Clustering dengan K-Means dan PCA")
    
        try:
            if st.session_state.df_clustered is not None:
                show_data(st.session_state.df_clustered)
                
                df_clustered = st.session_state.df_clustered
                selected_columns = st.session_state.selected_columns
                X = df_clustered[selected_columns].values

                # Add a download button for the clustered DataFrame as Excel
                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    df_clustered.to_excel(writer, sheet_name='Clustered Data', index=False)
                excel_buffer.seek(0)

                st.download_button(
                    label="Unduh Data Excel",
                    data=excel_buffer,
                    file_name="data_clustered.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)

                # Gabungkan PCA dan cluster ke dataframe
                df_pca = pd.DataFrame(X_pca, columns=['Dim1', 'Dim2'], index=df_clustered.index)
                df_pca = pd.concat([df_pca, df_clustered[['Cluster', 'Nama', 'Posyandu']]], axis=1)
                df_pca['Cluster'] = df_clustered['Cluster']
                df_pca['Cluster'] = df_pca['Cluster'].astype(str)
                df_pca['Posyandu'] = df_clustered['Posyandu'] if 'Posyandu' in df_clustered.columns else "Tidak diketahui"
                df_pca['Nama'] = df_clustered['Nama'] if 'Nama' in df_clustered.columns else "Tidak diketahui"
                df_pca["Nama_Posyandu"] = df_pca["Nama"].astype(str) + "<br>" + df_pca["Posyandu"].astype(str)
                
                # Visualisasi utama menggunakan plotly
                custom_color_map = {'1': '#43AA8B', '2': '#F9C74F'}
                custom_symbol_map = {'1': 'circle', '2': 'circle'}
                fig = px.scatter(
                    df_pca, x="Dim1", y="Dim2",
                    color="Cluster", symbol="Cluster",
                    hover_name="Nama_Posyandu",
                    title="Visualisasi Clustering menggunakan PCA",
                    color_discrete_map=custom_color_map,
                    symbol_map=custom_symbol_map
                )
                fig.update_traces(marker=dict(size=12, line=dict(width=0.5, color='black')))
                hovertemplate='%{customdata[0]}<br>%{customdata[1]}<extra></extra>'
                fig.update_layout(
                    xaxis_title=f"Dim1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                    yaxis_title=f"Dim2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                    legend_title_text='Cluster',
                    width=800, height=600
                )
                st.plotly_chart(fig, use_container_width=True)

                df_c1 = df_pca[df_pca['Cluster'] == '1']
                df_c2 = df_pca[df_pca['Cluster'] == '2']

                # Filter Posyandu
                list_posyandu = df_pca['Posyandu'].unique().tolist()
                st.markdown("### Pilih Posyandu")
                selected_posyandu = st.multiselect(
                    "Pilih Posyandu yang ingin ditampilkan",
                    placeholder="Silahkan pilih posyandu",
                    options=list_posyandu,
                    default=[]
                )

                if selected_posyandu:
                    df_pca_filtered = df_pca[df_pca['Posyandu'].isin(selected_posyandu)]
                else:
                    df_pca_filtered = df_pca.copy()

                # Scatter plot cluster 1
                st.subheader("Visualisasi Cluster 1 (C1)")
                df_c1_filtered = df_pca_filtered[df_pca_filtered['Cluster'] == '1']
                fig_c1 = px.scatter(
                    df_c1_filtered, x='Dim1', y='Dim2',
                    hover_name='Nama_Posyandu', title='Cluster 1 (C1) PCA Scatter Plot',
                    color_discrete_sequence=['#43AA8B']
                )
                fig_c1.update_traces(marker=dict(size=12, line=dict(width=0.5, color='black')))
                fig_c1.update_layout(
                    xaxis_title=f"Dim1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                    yaxis_title=f"Dim2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                    width=700, height=500
                )
                st.plotly_chart(fig_c1, use_container_width=True)

                # Scatter plot cluster 2
                st.subheader("Visualisasi Cluster 2 (C2)")
                df_c2_filtered = df_pca_filtered[df_pca_filtered['Cluster'] == '2']
                fig_c2 = px.scatter(
                    df_c2_filtered, x='Dim1', y='Dim2',
                    hover_name='Nama_Posyandu', title='Cluster 2 (C2) PCA Scatter Plot',
                    color_discrete_sequence=['#F9C74F']
                )
                fig_c2.update_traces(marker=dict(size=12, line=dict(width=0.5, color='black')))
                fig_c2.update_layout(
                    xaxis_title=f"Dim1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
                    yaxis_title=f"Dim2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
                    width=700, height=500
                )
                st.plotly_chart(fig_c2, use_container_width=True)

                cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
                cluster_percent = (cluster_counts / len(df_clustered) * 100).round(2)

                distribusi_df = pd.DataFrame({
                    "Cluster": cluster_counts.index,
                    "Jumlah Data": cluster_counts.values,
                    "Persentase (%)": cluster_percent.values
                })

                #Diagram Pie
                fig_pie = px.pie(
                    distribusi_df,
                    names="Cluster",
                    values="Jumlah Data",
                    title="Persentase Cluster per Data(%)",
                    hole=0.3,
                    color="Cluster"
                )
                fig_pie.update_layout(
                    legend_title="Cluster"
                )

                fig_pie.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")
        else:
            if not (
            st.session_state.df_normalized is not None and
            st.session_state.selected_columns and
            st.session_state.selected_data
    ):
                st.warning("Silahkan lakukan clustering terlebih dahulu di menu **Hasil Perhitungan Clustering**")
                st.markdown("<hr style='margin-top:50px; margin-bottom:10px;'>", unsafe_allow_html=True)

    elif menu == "Evaluasi Cluster":
        st.markdown("### Evaluasi Cluster")

        try:
            if st.session_state.df_normalized is not None:
                if st.session_state.selected_data:
                    num_clusters = st.session_state.num_clusters
                    selected_columns = st.session_state.selected_columns

                    st.info(f"Jumlah cluster: {num_clusters}")
                    st.info(f"Kolom yang digunakan: {', '.join(selected_columns)}")

                st.subheader("Data Normalisasi")
                with st.expander("Berikut ini adalah data dari hasil normalisasi (proses mengubah nilai-nilai data ke skala yang sama)", expanded=True):
                    st.dataframe(st.session_state.df_normalized)
        
                if st.session_state.selected_data:
                    st.subheader("Titik Awal Centroid")

                    centroid_df = pd.DataFrame(
                        st.session_state.selected_data,
                        columns=selected_columns
                    )
                    centroid_df.index = [f"C{i+1}" for i in range(len(centroid_df))]
                    with st.expander("Berikut ini adalah titik awal centroid hasil clustering sebelumnya", expanded=True):
                        st.dataframe(centroid_df)

                st.subheader("Tabel Jarak Data ke Tiap Centroid (Iterasi 1)")

                df_norm = st.session_state.df_normalized
                selected_columns = st.session_state.selected_columns
                centroids = st.session_state.selected_data
                jarak_dict = {}

                for i, c in enumerate(centroids):
                    jarak_list = []
                    for idx, row in df_norm[selected_columns].iterrows():
                        distance = math.sqrt(sum((row[col] - c[j])**2 for j, col in enumerate(selected_columns)))
                        jarak_list.append(round(distance, 2))
                        jarak_dict[f'C{i+1}'] = jarak_list

                df_jarak = pd.DataFrame(jarak_dict)
                df_jarak.index = [f'Data {i+1}' for i in df_jarak.index]
                with st.expander("Berikut ini adalah tabel jarak data ke tiap centroid (Iterasi 1)", expanded=True):
                    st.dataframe(df_jarak)

                if st.session_state.df_clustered is not None:
                    st.subheader("Titik Akhir Centroid")

                    # Ambil centroid akhir dari hasil clustering sebelumnya
                    df_clustered = st.session_state.df_clustered
                    selected_columns = st.session_state.selected_columns
                    centroid_final = df_clustered.groupby("Cluster")[selected_columns].mean().round(2)
                    centroid_final.index = [f"C{i}" for i in centroid_final.index]  # Cluster 1 → C1
                    with st.expander("Berikut ini adalah titik akhir centroid hasil clustering sebelumnya", expanded=True):
                         st.dataframe(centroid_final)

                    # Hitung Jarak Data ke Tiap Centroid (Iterasi 10)
                    st.subheader("Tabel Jarak Data ke Tiap Centroid (Iterasi 10)")

                    df_norm = st.session_state.df_normalized
                    jarak_dict = {}

                    for i, c in enumerate(centroid_final.values):
                         jarak_list = []
                         for idx, row in df_norm[selected_columns].iterrows():
                              distance = math.sqrt(sum((row[col] - c[j])**2 for j, col in enumerate(selected_columns)))
                              jarak_list.append(round(distance, 2))
                         jarak_dict[f'C{i+1}'] = jarak_list

                    df_jarak_final = pd.DataFrame(jarak_dict)
                    df_jarak_final.index = [f'Data {i+1}' for i in df_jarak_final.index]
                    with st.expander("Berikut ini adalah tabel jarak data ke tiap centroid (Iterasi 10)", expanded=True):
                         st.dataframe(df_jarak_final)
                if st.session_state.df_clustered is not None and st.session_state.selected_columns is not None:
                    df_clustered = st.session_state.df_clustered
                    selected_columns = st.session_state.selected_columns
                    X = df_clustered[selected_columns].values
                    labels = df_clustered['Cluster'].values

            # Cek minimal 2 cluster
            if len(set(labels)) > 1:
                # Rata-rata
                silhouette_avg = silhouette_score(X, labels)
                st.subheader("Silhouette Coefficient")
                penjelasan = "Cluster terbentuk dengan sangat baik dan jelas terpisah."
                if silhouette_avg > 0.70:
                    struktur_avg = "Strong Structure"
                elif silhouette_avg > 0.50:
                    struktur_avg = "Medium Structure"
                    penjelasan = "Cluster cukup baik, namun ada sedikit tumpang tindih."
                elif silhouette_avg > 0.25:
                    struktur_avg = "Weak Structure"
                    penjelasan = "Struktur Lemah."
                else:
                    struktur_avg = "No Structure"
                    penjelasan = "Tidak ada struktur cluster yang jelas."
                
                st.info(
                    f"**Kategori Struktur Cluster:** {struktur_avg}  \n"
                    f"**Nilai Silhouette Coefficient:** {silhouette_avg:.3f} - {penjelasan}"
                )
                # Per data
                sample_silhouette_values = silhouette_samples(X, labels)
                df_silhouette = pd.DataFrame({
                    "Cluster": labels,
                    "Silhouette Coefficient": np.round(sample_silhouette_values, 2)
                })

                if st.session_state.df is not None:
                    df_awal = st.session_state.df.reset_index(drop=True)
                    df_silhouette_full = pd.concat([df_awal, df_silhouette], axis=1)
                    with st.expander("Berikut ini adalah nilai Silhouette Coefficient ", expanded=True):
                        st.dataframe(df_silhouette_full)

        except Exception as e:
            st.error(f"Terjadi error: {str(e)}")
        else:
            if not (
            st.session_state.df_normalized is not None and
            st.session_state.selected_columns and
            st.session_state.selected_data
    ):
                
                st.warning("Silahkan unggah file terlebih dahulu di menu **Unggah File**.")
                st.markdown("<hr style='margin-top:50px; margin-bottom:10px;'>", unsafe_allow_html=True)

    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown("---")     
    st.markdown(
    "<p style='text-align:center; font-size: 14px;'>© 2025 Puskesmas Tanah Sareal</p>",
    unsafe_allow_html=True
)


