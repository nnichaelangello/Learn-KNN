# Materi: Panduan Definitif K-Nearest Neighbors (KNN) untuk Analisis Data: Teori, Pra-Pemrosesan, dan Implementasi Optimal

## 1. Pengenalan K-Nearest Neighbors (KNN)
K-Nearest Neighbors (KNN) adalah algoritma machine learning berbasis instance yang digunakan untuk klasifikasi dan regresi. KNN bekerja dengan mencari K titik data terdekat (tetangga) dari sebuah titik baru dalam ruang fitur berdasarkan metrik jarak (default: Euclidean), lalu membuat prediksi berdasarkan mayoritas kelas tetangga tersebut (klasifikasi) atau rata-rata nilai tetangga (regresi). Algoritma ini bersifat non-parametrik, tidak mengasumsikan distribusi data tertentu, dan termasuk dalam kategori "lazy learning" karena tidak membangun model selama fase pelatihan.

### 1.1 Cara Kerja KNN
- **Langkah 1**: Simpan semua data pelatihan (X: fitur, y: target) tanpa transformasi.
- **Langkah 2**: Untuk setiap titik data baru, hitung jarak ke semua titik pelatihan menggunakan metrik (misalnya Euclidean: $$\\sqrt{\sum (x_i - y_i)^2}\$$).
- **Langkah 3**: Pilih K tetangga terdekat berdasarkan jarak terkecil.
- **Langkah 4**: Klasifikasi: ambil mayoritas kelas; Regresi: ambil rata-rata nilai.
- **Contoh**: Jika K=5, dan tetangga terdekat adalah [0, 0, 1, 1, 1], prediksi klasifikasi adalah 1 (mayoritas).

### 1.2 Karakteristik Utama KNN
- **Lazy Learning**: Tidak ada pelatihan; komputasi dilakukan saat prediksi, sehingga efisien di awal tapi lambat pada data besar (kompleksitas $$\(O(n \cdot m)\)$$ untuk n sampel, m fitur).
- **Sensitivitas terhadap Jarak**: Skala fitur yang tidak seragam (misalnya, fitur 1: 0-1000, fitur 2: 0-1) menyebabkan fitur berskala besar mendominasi jarak.
- **Sensitivitas terhadap Outlier**: Outlier memperbesar jarak, menggeser tetangga yang dipilih.
- **Curse of Dimensionality**: Performa menurun di dimensi tinggi karena jarak menjadi kurang bermakna (semua titik tampak "jauh").
- **Hyperparameter**:
  - **K**: Jumlah tetangga (misalnya, K=5). K kecil risiko overfitting; K besar risiko underfitting.
  - **Metrik Jarak**: Euclidean, Manhattan $$(\sum |x_i - y_i|\)$$, Minkowski $$(\left(\sum |x_i - y_i|^p\right)^{1/p}\)$$.
  - **Bobot**: Uniform (sama untuk semua tetangga) atau Distance (invers jarak: $$\(1/d\)$$.

### 1.3 Kelebihan dan Kekurangan
- **Kelebihan**:
  - Sederhana, intuitif, dan mudah diimplementasikan.
  - Fleksibel untuk data non-linier dan distribusi arbitrer.
  - Efektif pada dataset kecil hingga menengah dengan pola lokal jelas.
- **Kekurangan**:
  - Lambat pada prediksi (kompleksitas $$\(O(n)\$$).
  - Rentan terhadap noise, outlier, dan dimensi tinggi.
  - Memerlukan pra-pemrosesan intensif untuk hasil optimal.

### 1.4 Mengapa KNN Penting?
- Cocok untuk analisis berbasis pola lokal (misalnya, klasifikasi gambar sederhana, rekomendasi berbasis jarak).
- Menjadi benchmark untuk algoritma berbasis jarak lainnya.
- Menuntut pemahaman mendalam tentang pra-pemrosesan data, menjadikannya alat pembelajaran ideal.

---

## 2. Hubungan KNN dengan Distribusi Data
KNN tidak bergantung pada asumsi distribusi, tetapi karakteristik data memengaruhi perhitungan jarak dan performa.

### 2.1 Distribusi Normal
- **Pengaruh**: KNN tidak peduli apakah data normal (Gaussian) karena hanya menghitung jarak antar titik, bukan parameter distribusi (mean, varians).
- **Implikasi**: Data skewed atau multimodal tetap bisa diproses, tetapi skala fitur harus diseragamkan untuk mencegah dominasi fitur tertentu.
- **Contoh**: Jika fitur 1 ~ N(0, 1) dan fitur 2 ~ N(100, 50), fitur 2 akan mendominasi Euclidean distance tanpa standarisasi.

### 2.2 Skewness
- **Pengaruh**: Skewness ekstrem (misalnya, > 2 atau < -2) menyebabkan ekor distribusi panjang, memperbesar jarak pada fitur tersebut dan mendistorsi tetangga terdekat.
- **Implikasi**: Transformasi (log, Box-Cox) diperlukan untuk menekan ekor dan menyeimbangkan distribusi.
- **Contoh**: Data pendapatan (skewness = 3.5) memiliki nilai ekstrem (misalnya, 1 juta), membuat jarak ke tetangga bias ke arah outlier tanpa transformasi.

### 2.3 Outlier
- **Pengaruh**: Outlier meningkatkan jarak secara tidak proporsional, menggeser tetangga terdekat ke arah data normal dan salah prediksi.
- **Implikasi**: Penghapusan atau transformasi outlier wajib untuk menjaga integritas jarak.
- **Contoh**: Jika 99% data di [0, 100] tapi ada outlier 10.000, jarak ke outlier akan mendominasi perhitungan.

### 2.4 Ketidakseimbangan Kelas
- **Pengaruh**: Kelas mayoritas mendominasi prediksi karena tetangga terdekat lebih mungkin dari kelas tersebut.
- **Implikasi**: Teknik seperti oversampling (SMOTE) atau bobot kelas diperlukan.
- **Contoh**: Dataset 90% kelas 0, 10% kelas 1; tanpa penanganan, prediksi bias ke kelas 0.

---

## 3. Exploratory Data Analysis (EDA) untuk KNN
EDA adalah fondasi untuk pra-pemrosesan KNN yang efektif. Berikut metode lengkap, dengan alasan, visualisasi, dan metrik:

### 3.1 Skala Fitur
- **Metode**: Boxplot, statistik deskriptif (`mean`, `std`, `min`, `max`, `quartiles`).
- **Cara Kerja**: Boxplot tunjukkan median, IQR, dan outlier; statistik beri rentang numerik.
- **Alasan**: KNN sensitif terhadap skala; fitur dengan rentang besar (misalnya, 0-10.000) mendominasi fitur kecil (0-1).
- **Contoh**: Feature1: [0, 20.000], Feature2: [0, 500]; tanpa standarisasi, Feature1 akan menguasai jarak.
- **Output**: Rentang, variabilitas, dan potensi kebutuhan normalisasi.

### 3.2 Distribusi dan Skewness
- **Metode**: Histogram dengan KDE, perhitungan skewness (`skew()`).
- **Cara Kerja**: Histogram visualisasi distribusi; skewness hitung kemiringan (positif > 0, negatif < 0).
- **Alasan**: Skewness ekstrem distorsi jarak; nilai > 2 atau < -2 butuh transformasi.
- **Contoh**: Feature1 skewness = 3.2 (ekor kanan panjang), butuh log transformasi.
- **Output**: Grafik distribusi, nilai skewness, keputusan transformasi.

### 3.3 Outlier
- **Metode**: IQR, scatter plot (pairwise fitur), Z-Score.
- **Cara Kerja**: IQR hitung batas [Q1 - 1.5*IQR, Q3 + 1.5*IQR]; scatter plot tunjukkan outlier multi-dimensi.
- **Alasan**: Outlier mengganggu tetangga terdekat; identifikasi awal pandu penanganan.
- **Contoh**: Feature4: 90% di [40, 60], tapi ada 1.500; IQR tandai sebagai outlier.
- **Output**: Jumlah outlier, lokasi, strategi penanganan.

### 3.4 Korelasi Antar Fitur
- **Metode**: Heatmap korelasi (`corr()`), koefisien Pearson/Spearman.
- **Cara Kerja**: Heatmap visualisasi korelasi antar fitur (nilai -1 hingga 1).
- **Alasan**: Korelasi tinggi (misalnya, > 0.8) tunjukkan redundansi; reduksi dimensi bisa diperlukan.
- **Contoh**: Feature1 dan Feature2 berkorelasi 0.9; PCA bisa kurangi dimensi.
- **Output**: Matriks korelasi, keputusan reduksi dimensi.

### 3.5 Missing Values
- **Metode**: Persentase (`isna().mean()`), heatmap missing values.
- **Cara Kerja**: Persentase hitung proporsi NaN; heatmap tunjukkan pola hilang.
- **Alasan**: KNN tidak proses NaN; imputasi wajib untuk kelengkapan data.
- **Contoh**: Feature2: 25% hilang, Feature4: 20%; butuh KNN Imputation.
- **Output**: Proporsi dan pola missing values, strategi imputasi.

### 3.6 Distribusi Kelas Target
- **Metode**: Countplot, persentase (`value_counts(normalize=True)`).
- **Cara Kerja**: Countplot visualisasi jumlah per kelas; persentase hitung proporsi.
- **Alasan**: Ketidakseimbangan (misalnya, 90:10) bias prediksi; butuh penanganan.
- **Contoh**: Target 0: 90%, Target 1: 10%; SMOTE diperlukan.
- **Output**: Distribusi kelas, keputusan oversampling.

### 3.7 Hubungan Antar Fitur
- **Metode**: Pairplot, scatter plot dengan hue target.
- **Cara Kerja**: Pairplot tunjukkan hubungan pairwise dan distribusi diagonal.
- **Alasan**: Deteksi overlap kelas, noise, dan kompleksitas klasifikasi.
- **Contoh**: Feature1 vs Feature2 overlap tinggi; klasifikasi sulit tanpa pra-pemrosesan.
- **Output**: Visualisasi hubungan, wawasan noise dan separabilitas.

### 3.8 Metode Paling Cocok untuk KNN
- **Rekomendasi**: Gunakan **semua metode di atas** secara berurutan untuk gambaran lengkap.
- **Alasan**: KNN membutuhkan data bersih, seragam, dan seimbang; EDA holistik pandu pra-pemrosesan optimal.

---

## 4. Metode Pra-Pemrosesan Data untuk KNN
Pra-pemrosesan menentukan keberhasilan KNN. Berikut metode lengkap untuk setiap aspek, dengan analisis mendalam, contoh, dan rekomendasi terbaik.

### 4.1 Penanganan Outlier
Outlier distorsi jarak KNN, wajib ditangani:
1. **IQR (Interquartile Range)**:
   - **Cara Kerja**: Hitung Q1, Q3, IQR = Q3 - Q1; hapus data di luar [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
   - **Kelebihan**: Sederhana, non-parametrik, cepat, cocok untuk fitur univariat.
   - **Kekurangan**: Tidak deteksi outlier multi-dimensi, buang data (risiko kehilangan informasi).
   - **Contoh**: Feature4: Q1=45, Q3=55, IQR=10, batas [30, 70]; 1.500 dihapus.
   - **Cocok untuk KNN**: Ya, untuk dataset sederhana dengan outlier univariat.
2. **Isolation Forest**:
   - **Cara Kerja**: Bangun pohon keputusan acak; titik dengan jalur pendek dianggap outlier (parameter: `contamination`).
   - **Kelebihan**: Tangani outlier multi-dimensi, robust pada data kompleks, tidak asumsi distribusi.
   - **Kekurangan**: Kompleks, `contamination` subjektif (misalnya, 0.1-0.2), lebih untuk anomali.
   - **Contoh**: Feature1, Feature2, Feature4 bersama tandai 15% data sebagai outlier.
   - **Cocok untuk KNN**: Ya, terbaik untuk dataset "sangat hancur" dengan noise multi-dimensi.
3. **Winsorizing**:
   - **Cara Kerja**: Ganti outlier dengan batas percentile (misalnya, 5% bawah, 95% atas).
   - **Kelebihan**: Pertahankan ukuran dataset, kurangi dampak outlier.
   - **Kekurangan**: Ubah distribusi, kurang agresif pada outlier ekstrem.
   - **Contoh**: Feature1: 20.000 jadi 1.000 (95th percentile).
   - **Cocok untuk KNN**: Kurang, karena outlier tetap ada meski dikurangi.
4. **Z-Score**:
   - **Cara Kerja**: Hitung `(x - mean) / std_dev`; hapus jika |Z| > 3.
   - **Kelebihan**: Berbasis statistik, mudah dihitung.
   - **Kekurangan**: Asumsi simetri, sensitif pada skewness ekstrem.
   - **Contoh**: Feature2: mean=500, std=1000, Z=9.5 untuk 10.000; dihapus.
   - **Cocok untuk KNN**: Tidak, asumsi distribusi tidak selaras dengan KNN.
5. **Log Transformasi**:
   - **Cara Kerja**: Terapkan `log(x+1)` untuk tekan ekor distribusi.
   - **Kelebihan**: Kurangi outlier dan skewness sekaligus, sederhana.
   - **Kekurangan**: Hanya untuk data positif, tidak hapus outlier.
   - **Contoh**: Feature1: 20.000 jadi ~9.9 (log scale).
   - **Cocok untuk KNN**: Ya, pelengkap untuk skewness > 2.
6. **DBSCAN**:
   - **Cara Kerja**: Kluster berdasarkan densitas; titik di luar kluster jadi outlier (parameter: `eps`, `min_samples`).
   - **Kelebihan**: Tangani outlier multi-dimensi berdasarkan kepadatan.
   - **Kekurangan**: Kompleks, sensitif parameter, tidak skalabel.
   - **Contoh**: Feature1-Feature2 tandai 5% data sebagai noise.
   - **Cocok untuk KNN**: Tidak, terlalu berat untuk pra-pemrosesan.
7. **LOF (Local Outlier Factor)**:
   - **Cara Kerja**: Bandingkan densitas lokal titik dengan tetangganya.
   - **Kelebihan**: Deteksi outlier lokal multi-dimensi.
   - **Kekurangan**: Lambat, butuh tuning (k tetangga).
   - **Cocok untuk KNN**: Tidak, lebih untuk analisis anomali.
- **Metode Paling Cocok untuk KNN**:
  - **Isolation Forest**: Terbaik untuk dataset kompleks ("sangat hancur") karena tangani outlier multi-dimensi dengan robust.
  - **IQR**: Terbaik untuk dataset sederhana (univariat) karena cepat dan non-parametrik.
  - **Log Transformasi**: Pelengkap wajib jika skewness > 2, kombinasikan dengan Isolation Forest/IQR.

### 4.2 Normalisasi atau Standarisasi
Skala fitur harus seragam untuk KNN:
1. **Standarisasi (Z-Score)**:
   - **Cara Kerja**: `(x - mean) / std_dev`, jadi mean=0, std_dev=1.
   - **Kelebihan**: Seragamkan skala, kurangi dampak skewness ringan, umum digunakan.
   - **Kekurangan**: Sensitif outlier ekstrem (mean/std dipengaruhi).
   - **Contoh**: Feature1: [0, 20.000] jadi [-0.5, 2.5].
   - **Cocok untuk KNN**: Ya, standar untuk data biasa setelah outlier ditangani.
2. **RobustScaler**:
   - **Cara Kerja**: `(x - median) / IQR`, gunakan median dan IQR.
   - **Kelebihan**: Tahan outlier, cocok untuk data noisy/"sangat hancur".
   - **Kekurangan**: Kurang sensitif pada variabilitas normal dibandingkan Z-Score.
   - **Contoh**: Feature2: [0, 10.000] jadi [-0.2, 1.8].
   - **Cocok untuk KNN**: Ya, terbaik untuk data dengan outlier sisa.
3. **Min-Max Scaling**:
   - **Cara Kerja**: `(x - min) / (max - min)`, jadi [0, 1].
   - **Kelebihan**: Pertahankan distribusi, sederhana, cocok untuk data terbatas.
   - **Kekurangan**: Sensitif outlier (min/max terpengaruh).
   - **Contoh**: Feature4: [40, 1.500] jadi [0, 1].
   - **Cocok untuk KNN**: Ya, hanya jika outlier sudah dihapus.
4. **L2 Normalization**:
   - **Cara Kerja**: Bagi fitur dengan norma L2 $$(\sqrt{\sum x_i^2}\)$$, jadi vektor unit.
   - **Kelebihan**: Cocok untuk cosine similarity, kurangi dimensi efektif.
   - **Kekurangan**: Kurang relevan untuk Euclidean/Manhattan KNN.
   - **Contoh**: [3, 4] jadi [0.6, 0.8].
   - **Cocok untuk KNN**: Tidak, lebih untuk model lain.
5. **Quantile Transformation**:
   - **Cara Kerja**: Ubah data ke distribusi uniform/normal via ranking.
   - **Kelebihan**: Kurangi skewness drastis, seragamkan distribusi.
   - **Kekurangan**: Ubah struktur data, tidak perlu untuk KNN.
   - **Contoh**: Feature1 jadi distribusi uniform [0, 1].
   - **Cocok untuk KNN**: Tidak, terlalu agresif.
6. **MaxAbs Scaling**:
   - **Cara Kerja**: `(x / max(|x|))`, jadi [-1, 1].
   - **Kelebihan**: Pertahankan sparsity, sederhana.
   - **Kekurangan**: Sensitif outlier, kurang umum.
   - **Cocok untuk KNN**: Tidak, kurang robust.
- **Metode Paling Cocok untuk KNN**:
  - **RobustScaler**: Terbaik untuk data noisy/"sangat hancur" karena tahan outlier.
  - **Standarisasi (Z-Score)**: Terbaik untuk data biasa setelah outlier minim.
  - **Min-Max Scaling**: Alternatif jika outlier sudah bersih dan distribusi ingin dipertahankan.

### 4.3 Penanganan Missing Values
KNN tidak proses NaN:
1. **KNN Imputation**:
   - **Cara Kerja**: Isi NaN dengan rata-rata K tetangga terdekat berdasarkan fitur lain.
   - **Kelebihan**: Konsisten dengan KNN, akurat pada data spasial, pertahankan struktur jarak.
   - **Kekurangan**: Lambat pada data besar $$(O(n^2)\)$$, sensitif parameter K.
   - **Contoh**: Feature2 NaN diisi dengan rata-rata 5 tetangga.
   - **Cocok untuk KNN**: Ya, metode terbaik karena selaras dengan jarak.
2. **Median Imputation**:
   - **Cara Kerja**: Isi NaN dengan median per fitur.
   - **Kelebihan**: Cepat, robust pada skewness/outlier, sederhana.
   - **Kekurangan**: Kurang presisi, abaikan hubungan antar fitur.
   - **Contoh**: Feature4: median=50 untuk 20% NaN.
   - **Cocok untuk KNN**: Ya, efisien untuk data kecil/miring.
3. **Mean Imputation**:
   - **Cara Kerja**: Isi NaN dengan mean per fitur.
   - **Kelebihan**: Cepat, sederhana.
   - **Kekurangan**: Sensitif outlier, distorsi distribusi miring.
   - **Contoh**: Feature2: mean=500, dipengaruhi outlier 10.000.
   - **Cocok untuk KNN**: Tidak, kurang akurat.
4. **Iterative Imputer**:
   - **Cara Kerja**: Model prediktif (misalnya regresi) iteratif untuk imputasi.
   - **Kelebihan**: Akurat pada data terstruktur, tangkap hubungan fitur.
   - **Kekurangan**: Asumsi linearitas, kompleks, lambat.
   - **Contoh**: Feature2 diprediksi dari Feature1 dan Feature4.
   - **Cocok untuk KNN**: Tidak, terlalu berat dan tidak selaras.
5. **Random Imputation**:
   - **Cara Kerja**: Isi NaN dengan sampel acak dari distribusi fitur.
   - **Kelebihan**: Pertahankan variabilitas, cepat.
   - **Kekurangan**: Tambah noise, kurang akurat.
   - **Contoh**: Feature4 NaN jadi nilai acak [40, 1500].
   - **Cocok untuk KNN**: Tidak, distorsi jarak.
6. **Mode Imputation**:
   - **Cara Kerja**: Isi NaN dengan modus (khusus kategorikal).
   - **Kelebihan**: Sederhana, cocok untuk data diskrit.
   - **Kekurangan**: Hanya untuk kategorikal, kurang relevan untuk numerik.
   - **Cocok untuk KNN**: Tidak, terbatas pada fitur non-numerik.
- **Metode Paling Cocok untuk KNN**:
  - **KNN Imputation**: Terbaik untuk akurasi dan konsistensi dengan prinsip jarak KNN.
  - **Median Imputation**: Alternatif cepat untuk data kecil atau skewness tinggi.

### 4.4 Transformasi untuk Skewness
Skewness ekstrem (> 2 atau < -2) distorsi jarak:
1. **Logaritma**:
   - **Cara Kerja**: `log(x+1)` untuk data positif (tambah 1 hindari log(0)).
   - **Kelebihan**: Sederhana, tekan skewness/outlier, cepat.
   - **Kekurangan**: Hanya untuk data positif, tidak hapus outlier.
   - **Contoh**: Feature1: skewness 3.2 jadi ~0.5 setelah log.
   - **Cocok untuk KNN**: Ya, metode terbaik untuk skewness ekstrem.
2. **Box-Cox**:
   - **Cara Kerja**: Transformasi parametrik $$(\frac{x^\lambda - 1}{\lambda}\)$$ untuk data positif.
   - **Kelebihan**: Fleksibel untuk berbagai skewness, optimalisasi lambda.
   - **Kekurangan**: Kompleks, butuh data positif, lambat.
   - **Contoh**: Feature2: skewness 2.8 jadi ~0.3.
   - **Cocok untuk KNN**: Ya, alternatif presisi tinggi.
3. **Akar Kuadrat**:
   - **Cara Kerja**: `sqrt(x)` untuk data non-negatif.
   - **Kelebihan**: Sederhana, kurangi skewness ringan.
   - **Kekurangan**: Kurang efektif pada skewness > 2.
   - **Contoh**: Feature1: skewness 1.5 jadi 0.8.
   - **Cocok untuk KNN**: Ya, untuk skewness ringan.
4. **Yeo-Johnson**:
   - **Cara Kerja**: Transformasi untuk data positif/negatif via parameter.
   - **Kelebihan**: Fleksibel, tangani negatif.
   - **Kekurangan**: Kompleks, jarang diperlukan untuk KNN.
   - **Contoh**: Feature negatif jadi skewness ~0.
   - **Cocok untuk KNN**: Tidak, terlalu rumit.
5. **Power Transform**:
   - **Cara Kerja**: Transformasi eksponensial kustom (misalnya, \(x^p\)).
   - **Kelebihan**: Kustomisasi tinggi.
   - **Kekurangan**: Butuh tuning manual, tidak standar.
   - **Cocok untuk KNN**: Tidak, kurang efisien.
6. **Rank Transformation**:
   - **Cara Kerja**: Ubah data ke peringkat (ordinal).
   - **Kelebihan**: Kurangi skewness tanpa asumsi.
   - **Kekurangan**: Hilang informasi numerik.
   - **Cocok untuk KNN**: Tidak, distorsi jarak.
- **Metode Paling Cocok untuk KNN**:
  - **Logaritma**: Terbaik untuk skewness ekstrem karena sederhana dan efektif.
  - **Box-Cox**: Alternatif jika skewness kompleks dan presisi tinggi dibutuhkan.

### 4.5 Reduksi Dimensi
Dimensi tinggi kurangi akurasi KNN (*curse of dimensionality*):
1. **PCA (Principal Component Analysis)**:
   - **Cara Kerja**: Ubah fitur ke komponen utama berdasarkan varians maksimum.
   - **Kelebihan**: Efisien, hapus korelasi, skalabel, linier.
   - **Kekurangan**: Kehilangan interpretasi, asumsi linieritas.
   - **Contoh**: 10 fitur jadi 5 komponen (95% varians).
   - **Cocok untuk KNN**: Ya, metode terbaik untuk efisiensi.
2. **t-SNE**:
   - **Cara Kerja**: Reduksi non-linier untuk visualisasi berdasarkan kemiripan lokal.
   - **Kelebihan**: Tangkap struktur lokal, visualisasi bagus.
   - **Kekurangan**: Lambat $$(O(n^2)\)$$, hanya untuk visualisasi.
   - **Contoh**: 10 fitur jadi 2D plot.
   - **Cocok untuk KNN**: Tidak, tidak untuk prediksi.
3. **UMAP (Uniform Manifold Approximation)**:
   - **Cara Kerja**: Reduksi non-linier berbasis topologi.
   - **Kelebihan**: Cepat, tangkap struktur kompleks, skalabel.
   - **Kekurangan**: Sensitif parameter, kurang stabil.
   - **Contoh**: 10 fitur jadi 3D dengan struktur lokal.
   - **Cocok untuk KNN**: Ya, alternatif canggih tapi kurang umum.
4. **Seleksi Fitur (Feature Selection)**:
   - **Cara Kerja**: Pilih fitur penting (chi-square, mutual info, F-test).
   - **Kelebihan**: Pertahankan interpretasi, sederhana.
   - **Kekurangan**: Subjektif, risiko hilang informasi.
   - **Contoh**: Pilih 3 dari 10 fitur berdasarkan F-score.
   - **Cocok untuk KNN**: Ya, untuk dataset kecil/fitur jelas.
5. **Autoencoder**:
   - **Cara Kerja**: Neural network kompresi data ke dimensi rendah.
   - **Kelebihan**: Tangkap pola non-linier, fleksibel.
   - **Kekurangan**: Kompleks, butuh data besar, tuning berat.
   - **Contoh**: 10 fitur jadi 4 via hidden layer.
   - **Cocok untuk KNN**: Tidak, terlalu berat.
6. **LDA (Linear Discriminant Analysis)**:
   - **Cara Kerja**: Reduksi dimensi berbasis kelas (supervised).
   - **Kelebihan**: Optimalkan separabilitas kelas.
   - **Kekurangan**: Asumsi linieritas, butuh label.
   - **Cocok untuk KNN**: Tidak, lebih untuk klasifikasi linear.
- **Metode Paling Cocok untuk KNN**:
  - **PCA**: Terbaik untuk efisiensi, stabilitas, dan penghapusan korelasi (gunakan 95-99% varians).
  - **Seleksi Fitur**: Alternatif untuk dataset kecil dengan fitur interpretatif.

### 4.6 Encoding Kategorikal
KNN butuh data numerik:
1. **One-Hot Encoding**:
   - **Cara Kerja**: Ubah kategori jadi kolom biner (0/1).
   - **Kelebihan**: Hindari ordinalitas, cocok untuk jarak KNN.
   - **Kekurangan**: Tambah dimensi (misalnya, 5 kategori jadi 5 kolom).
   - **Contoh**: Feature3: [A, B, C] jadi [1,0,0], [0,1,0], [0,0,1].
   - **Cocok untuk KNN**: Ya, metode terbaik.
2. **Label Encoding**:
   - **Cara Kerja**: Beri angka unik (misalnya, A=0, B=1, C=2).
   - **Kelebihan**: Hemat dimensi, sederhana.
   - **Kekurangan**: Asumsi ordinalitas, distorsi jarak (2 lebih "jauh" dari 0).
   - **Contoh**: Feature3: [A, B, C] jadi [0, 1, 2].
   - **Cocok untuk KNN**: Tidak, salah urutkan kategori.
3. **Target Encoding**:
   - **Cara Kerja**: Ganti kategori dengan rata-rata target (misalnya, A=0.3, B=0.7).
   - **Kelebihan**: Kurangi dimensi, gunakan info target.
   - **Kekurangan**: Risiko overfitting, sensitif data kecil.
   - **Contoh**: Feature3: A (mean target=0.3), B (0.7).
   - **Cocok untuk KNN**: Tidak, terlalu spesifik.
4. **Binary Encoding**:
   - **Cara Kerja**: Ubah kategori ke kode biner (misalnya, A=00, B=01, C=10).
   - **Kelebihan**: Hemat dimensi dibandingkan One-Hot.
   - **Kekurangan**: Masih ada ordinalitas implisit.
   - **Contoh**: Feature3: [A, B, C] jadi [0,0], [0,1], [1,0].
   - **Cocok untuk KNN**: Tidak, kurang ideal untuk jarak.
5. **Frequency Encoding**:
   - **Cara Kerja**: Ganti kategori dengan frekuensi kemunculan.
   - **Kelebihan**: Sederhana, kurangi dimensi.
   - **Kekurangan**: Hilang makna kategorikal, distorsi jarak.
   - **Cocok untuk KNN**: Tidak, tidak relevan.
- **Metode Paling Cocok untuk KNN**:
  - **One-Hot Encoding**: Terbaik karena hindari ordinalitas dan sesuai metrik jarak (tambahan dimensi diatasi PCA jika perlu).

### 4.7 Penanganan Ketidakseimbangan Kelas
Ketidakseimbangan bias prediksi KNN ke kelas mayoritas:
1. **SMOTE (Synthetic Minority Oversampling Technique)**:
   - **Cara Kerja**: Buat data sintetis untuk kelas minoritas via interpolasi tetangga.
   - **Kelebihan**: Seimbangkan kelas, pertahankan struktur lokal.
   - **Kekurangan**: Tambah ukuran data, risiko noise sintetis.
   - **Contoh**: Kelas 1 (10%) jadi 50% via SMOTE.
   - **Cocok untuk KNN**: Ya, metode terbaik untuk ketidakseimbangan ekstrem.
2. **Random Oversampling**:
   - **Cara Kerja**: Duplikat data minoritas secara acak.
   - **Kelebihan**: Sederhana, cepat, tidak ubah struktur.
   - **Kekurangan**: Risiko overfitting, tambah redundansi.
   - **Contoh**: Kelas 1: 100 jadi 900 via duplikasi.
   - **Cocok untuk KNN**: Ya, tapi kurang akurat dibandingkan SMOTE.
3. **Random Undersampling**:
   - **Cara Kerja**: Kurangi data mayoritas secara acak.
   - **Kelebihan**: Cepat, kurangi ukuran data.
   - **Kekurangan**: Hilang informasi, risiko underfitting.
   - **Contoh**: Kelas 0: 900 jadi 100.
   - **Cocok untuk KNN**: Tidak, kehilangan pola krusial.
4. **Class Weighting**:
   - **Cara Kerja**: Beri bobot lebih pada kelas minoritas di KNN (`weights='distance'` atau custom).
   - **Kelebihan**: Tidak ubah data, sederhana, langsung diimplementasikan.
   - **Kekurangan**: Kurang agresif pada ketidakseimbangan ekstrem.
   - **Contoh**: Kelas 1 bobot 9x lebih besar dari kelas 0.
   - **Cocok untuk KNN**: Ya, untuk ketidakseimbangan ringan.
5. **ADASYN (Adaptive Synthetic Sampling)**:
   - **Cara Kerja**: SMOTE adaptif, fokus pada area sulit (minoritas dekat mayoritas).
   - **Kelebihan**: Tingkatkan akurasi di batas kelas.
   - **Kekurangan**: Kompleks, risiko noise lebih tinggi.
   - **Contoh**: Kelas 1 lebih banyak di area overlap.
   - **Cocok untuk KNN**: Ya, alternatif canggih untuk SMOTE.
6. **Tomek Links**:
   - **Cara Kerja**: Hapus data mayoritas di batas kelas untuk separabilitas.
   - **Kelebihan**: Tingkatkan pemisahan kelas.
   - **Kekurangan**: Kurangi data, kurang agresif.
   - **Cocok untuk KNN**: Tidak, terlalu ringan untuk ketidakseimbangan besar.
- **Metode Paling Cocok untuk KNN**:
  - **SMOTE**: Terbaik untuk ketidakseimbangan ekstrem (misalnya, 90:10) karena seimbangkan kelas dengan struktur lokal.
  - **Class Weighting**: Terbaik untuk ketidakseimbangan ringan (misalnya, 70:30) karena sederhana dan langsung.

---

## 5. Ensemble dengan KNN
Ensemble tingkatkan stabilitas dan akurasi KNN:
1. **Bagging (Bootstrap Aggregating)**:
   - **Cara Kerja**: Latih beberapa KNN pada subset data acak (bootstrap), agregasi prediksi via mayoritas.
   - **Kelebihan**: Kurangi varians, tingkatkan stabilitas, robust pada noise.
   - **Kekurangan**: Komputasi berat $$(O(n \cdot m \cdot t)\$$ untuk t estimator).
   - **Contoh**: 10 KNN, masing-masing pada 80% data acak.
   - **Cocok untuk KNN**: Ya, metode terbaik untuk stabilitas.
2. **Weighted KNN**:
   - **Cara Kerja**: Bobot tetangga berdasarkan invers jarak (`1/d`) atau custom.
   - **Kelebihan**: Tingkatkan akurasi dengan prioritas tetangga dekat, sederhana.
   - **Kekurangan**: Tidak kurangi varians seperti Bagging.
   - **Contoh**: Tetangga jarak 0.1 bobot 10, jarak 1 bobot 1.
   - **Cocok untuk KNN**: Ya, efektif untuk akurasi sederhana.
3. **Stacking**:
   - **Cara Kerja**: Gabung KNN dengan model lain (misalnya, Random Forest, SVM), prediksi akhir via meta-model (misalnya, logistic regression).
   - **Kelebihan**: Kombinasi kekuatan model, tingkatkan akurasi.
   - **Kekurangan**: Kompleks, butuh tuning, risiko overfitting.
   - **Contoh**: KNN + RF → Logistic Regression untuk prediksi akhir.
   - **Cocok untuk KNN**: Ya, untuk performa maksimal pada data kompleks.
4. **Clustering + KNN**:
   - **Cara Kerja**: Kluster data (misalnya, K-Means), lalu KNN per kluster.
   - **Kelebihan**: Tingkatkan efisiensi $$(O(k \cdot n/k)\)$$ vs $$\(O(n)\)$$, fokus lokal.
   - **Kekurangan**: Risiko kluster salah, tambah langkah.
   - **Contoh**: 5 kluster, KNN hanya cari tetangga dalam kluster.
   - **Cocok untuk KNN**: Ya, untuk dataset sangat besar.
5. **Random Subspace KNN**:
   - **Cara Kerja**: Latih KNN pada subset fitur acak, agregasi prediksi.
   - **Kelebihan**: Kurangi dimensi efektif, tingkatkan stabilitas.
   - **Kekurangan**: Risiko hilang fitur penting, kurang umum.
   - **Contoh**: 3 dari 10 fitur acak per model.
   - **Cocok untuk KNN**: Ya, untuk data berdimensi tinggi.
6. **AdaBoost dengan KNN**:
   - **Cara Kerja**: Bobot ulang data sulit, iterasi KNN.
   - **Kelebihan**: Fokus pada kesalahan, tingkatkan akurasi.
   - **Kekurangan**: KNN tidak cocok untuk boosting (lazy learner).
   - **Cocok untuk KNN**: Tidak, kurang praktis.
- **Metode Paling Cocok untuk KNN**:
  - **Bagging**: Terbaik untuk stabilitas dan performa umum pada dataset besar/noisy.
  - **Weighted KNN**: Terbaik untuk akurasi sederhana pada dataset kecil.

---

## 6. Evaluasi Performa KNN
Evaluasi holistik wajib untuk validasi KNN:
1. **Akurasi**: Proporsi prediksi benar $$(\frac{TP + TN}{TP + TN + FP + FN}\)$$.
   - **Kapan Digunakan**: Kelas seimbang.
   - **Contoh**: 80% prediksi benar.
2. **ROC-AUC**: Area di bawah kurva ROC (True Positive Rate vs False Positive Rate).
   - **Kapan Digunakan**: Ketidakseimbangan kelas, diskriminasi antar kelas.
   - **Contoh**: AUC = 0.85 (baik).
3. **Cross-Validation**: Akurasi rata-rata pada K-fold (misalnya, 5-fold).
   - **Kapan Digunakan**: Stabilitas model.
   - **Contoh**: Mean CV = 0.78 ± 0.05.
4. **Classification Report**: Precision $$(\frac{TP}{TP + FP}\)$$, Recall $$(\frac{TP}{TP + FN}\)$$, F1-score $$(2 \cdot \frac{precision \cdot recall}{precision + recall}\)$$.
   - **Kapan Digunakan**: Performa per kelas.
   - **Contoh**: Kelas 1: Precision=0.75, Recall=0.80, F1=0.77.
5. **Confusion Matrix**: Tabel TP, TN, FP, FN.
   - **Kapan Digunakan**: Detail kesalahan prediksi.
   - **Contoh**: [[800, 100], [50, 150]].
6. **Learning Curve**: Akurasi training vs validation berdasarkan ukuran data.
   - **Kapan Digunakan**: Deteksi overfitting/underfitting.
   - **Contoh**: Training=0.95, Validation=0.75 (overfitting).
7. **Precision-Recall Curve**: Fokus pada kelas minoritas.
   - **Kapan Digunakan**: Ketidakseimbangan ekstrem.
   - **Contoh**: PR-AUC = 0.70.
- **Metode Paling Cocok untuk KNN**: Gunakan **semua metrik di atas** untuk evaluasi lengkap; fokus **ROC-AUC** dan **F1-score** pada ketidakseimbangan.

---

## 7. Kesimpulan dan Rekomendasi untuk KNN
KNN adalah algoritma berbasis jarak yang kuat namun sensitif terhadap kualitas data. Berikut panduan definitif untuk dataset kompleks (misalnya, "sangat hancur" dengan 10.000 baris, outlier, skewness, missing values, ketidakseimbangan):

### 7.1 EDA
- **Rekomendasi**: Lakukan semua langkah:
  - Skala (boxplot, statistik).
  - Distribusi (histogram, skewness).
  - Outlier (IQR, scatter).
  - Korelasi (heatmap).
  - Missing values (persentase, heatmap).
  - Kelas (countplot).
  - Hubungan (pairplot).
- **Alasan**: Berikan wawasan holistik untuk pra-pemrosesan optimal.

### 7.2 Pra-Pemrosesan
- **Penanganan Outlier**:
  - **Terbaik**: **Isolation Forest** (contamination=0.1-0.2) untuk data kompleks/multi-dimensi; kombinasikan dengan **Log Transformasi** jika skewness > 2.
  - **Alternatif**: **IQR** untuk data sederhana/univariat.
- **Standarisasi**:
  - **Terbaik**: **RobustScaler** untuk data noisy/"sangat hancur" dengan outlier sisa.
  - **Alternatif**: **Z-Score** jika outlier minim.
- **Missing Values**:
  - **Terbaik**: **KNN Imputation** (K=5) untuk akurasi dan konsistensi jarak.
  - **Alternatif**: **Median Imputation** untuk efisiensi.
- **Skewness**:
  - **Terbaik**: **Logaritma** untuk skewness ekstrem (> 2).
  - **Alternatif**: **Box-Cox** untuk presisi tinggi.
- **Reduksi Dimensi**:
  - **Terbaik**: **PCA** (95-99% varians) untuk efisiensi dan stabilitas.
  - **Alternatif**: **Seleksi Fitur** untuk data kecil.
- **Encoding Kategorikal**:
  - **Terbaik**: **One-Hot Encoding** untuk semua kasus.
- **Ketidakseimbangan**:
  - **Terbaik**: **SMOTE** untuk ketidakseimbangan ekstrem (misalnya, 90:10).
  - **Alternatif**: **Class Weighting** untuk ketidakseimbangan ringan.

### 7.3 Ensemble
- **Terbaik**: **Bagging** untuk stabilitas dan performa pada dataset besar/noisy (n_estimators=50-100).
- **Alternatif**: **Weighted KNN** untuk akurasi sederhana pada data kecil.

### 7.4 Evaluasi
- Gunakan **ROC-AUC**, **F1-score**, **CV**, **Confusion Matrix**, dan **Learning Curve** untuk validasi holistik.

### 7.5 Implementasi pada Dataset "Sangat Hancur"
- **Langkah**: EDA → Isolation Forest + Log → KNN Imputation → RobustScaler → PCA (99%) → One-Hot → SMOTE → Bagging KNN → Evaluasi lengkap.
- **Alasan**: Tangani outlier multi-dimensi, missing values, skewness, skala, dimensi, ketidakseimbangan, dan stabilitas secara optimal.
