# Struktur Materi: Memahami Alur Belajar Neural Network

Dokumen ini menjelaskan konsep teoretis, arsitektur, dan alur belajar (learning flow) dari Neural Network (Jaringan Saraf Tiruan). Pemahaman materi ini menjadi fondasi sebelum melakukan implementasi praktis di dalam Jupyter Notebook.

---

## 1. Pendahuluan ke Neural Network
Neural Network adalah model komputasi yang terinspirasi oleh struktur dan fungsi jaringan saraf biologis manusia. Model ini terdiri dari unit-unit pemrosesan informasi dasar yang disebut **neuron** (atau *node*), yang saling terhubung melalui hubungan berbobot (**weights**).

Secara umum, arsitektur Neural Network (Multilayer Perceptron / MLP) terdiri atas tiga bagian utama:
1. **Input Layer (Lapisan Masukan)**: Menerima data mentah dari luar setelah dikonversi menjadi representasi numerik (fitur).
2. **Hidden Layer (Lapisan Tersembunyi)**: Mengekstrak pola non-linear dari fitur input secara bertingkat.
3. **Output Layer (Lapisan Keluaran)**: Menghasilkan prediksi akhir berdasarkan informasi yang telah diproses oleh lapisan tersembunyi.

---

## 2. Representasi Input (Input Layer)
Sebelum data teks (seperti ulasan aplikasi atau komentar) dapat diproses oleh Neural Network, data tersebut harus dikonversi menjadi angka. Proses ini disebut **Ekstraksi Fitur Teks** (Text Feature Extraction).

Salah satu metode paling sederhana dan efektif untuk skala kecil adalah **Bag-of-Words (BoW)** atau **Binerisasi Kata (Binary Vectorization)**:
* Kita menentukan daftar kosa kata unik (vocabulary) yang relevan dengan kasus.
* Setiap ulasan direpresentasikan sebagai vektor biner $X = [x_1, x_2, ..., x_n]$, di mana $x_i = 1$ jika kata ke-$i$ dalam kosa kata muncul pada ulasan tersebut, dan $x_i = 0$ jika tidak muncul.

Vektor $X$ ini bertindak sebagai **Input Layer** yang disimbolkan sebagai neuron masukan.

---

## 3. Propagasi Maju (Forward Propagation)
Propagasi Maju adalah proses mengalirkan informasi dari input layer menuju hidden layer hingga menghasilkan nilai prediksi di output layer. Untuk setiap neuron di hidden layer dan output layer, terjadi dua langkah perhitungan dasar:

### A. Kombinasi Linear (Weighted Sum)
Setiap input dikalikan dengan bobot (*weight*) masing-masing, kemudian dijumlahkan dan ditambah dengan nilai bias.
$$z = \sum_{i=1}^{n} (x_i \cdot w_i) + b$$
Secara matriks untuk suatu lapisan:
$$Z = X \cdot W + b$$
* $W$ adalah matriks bobot yang mengontrol kekuatan pengaruh input terhadap neuron berikutnya.
* $b$ adalah bias yang memungkinkan pergeseran fungsi aktivasi ke kiri atau ke kanan untuk membantu kecocokan data (*fitting*).

### B. Fungsi Aktivasi (Activation Function)
Hasil penjumlahan linear $z$ dilewatkan ke fungsi non-linear agar Neural Network dapat mempelajari hubungan non-linear yang kompleks. Beberapa fungsi aktivasi populer:
1. **Sigmoid**: Memetakan nilai ke rentang $[0, 1]$, sangat cocok untuk probabilitas klasifikasi biner.
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
2. **ReLU (Rectified Linear Unit)**: Memetakan nilai negatif menjadi 0 dan membiarkan nilai positif tetap.
   $$f(z) = \max(0, z)$$

---

## 4. Lapisan Keluaran (Output Layer)
Pada kasus klasifikasi biner (dua kelas target, misalnya Positif vs Negatif), output layer biasanya terdiri dari satu neuron tunggal yang menggunakan fungsi aktivasi **Sigmoid**.
* Nilai keluaran $\hat{y}$ (prediksi) berkisar antara $0$ hingga $1$.
* Interpretasi: Jika $\hat{y} \geq 0.5$, data diklasifikasikan ke kelas positif ($1$). Jika $\hat{y} < 0.5$, data diklasifikasikan ke kelas negatif ($0$).

---

## 5. Pengukuran Kesalahan (Error/Loss Function)
Untuk mengetahui seberapa jauh tebakan model dari label asli ($y$), kita menggunakan fungsi kerugian (*Loss Function*).
* **Mean Squared Error (MSE)** (sering digunakan untuk kemudahan visualisasi perhitungan):
  $$E = \frac{1}{2} (y - \hat{y})^2$$
* **Binary Cross-Entropy (BCE)** (standar industri untuk klasifikasi biner):
  $$L = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]$$

Tujuan proses pelatihan model adalah meminimalkan nilai kesalahan ini menuju sedekat mungkin dengan nol.

---

## 6. Propagasi Mundur (Backpropagation)
Backpropagation adalah metode untuk menghitung gradien (kemiringan) dari fungsi kesalahan terhadap setiap parameter bobot dan bias dalam jaringan. Gradien ini memberitahu kita seberapa besar perubahan pada bobot/bias mempengaruhi total kesalahan.

Proses ini didasarkan pada **Aturan Rantai (Chain Rule)** dari kalkulus. Contoh untuk bobot output ($w_{j}^{(2)}$) dengan loss MSE:
$$\frac{\partial E}{\partial w_{j}^{(2)}} = \frac{\partial E}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{(2)}} \cdot \frac{\partial z^{(2)}}{\partial w_{j}^{(2)}}$$

Di mana:
1. **Turunan Kesalahan**: $\frac{\partial E}{\partial \hat{y}} = - (y - \hat{y})$
2. **Turunan Aktivasi Sigmoid**: $\frac{\partial \hat{y}}{\partial z^{(2)}} = \hat{y}(1 - \hat{y})$
3. **Turunan Kombinasi Linear**: $\frac{\partial z^{(2)}}{\partial w_{j}^{(2)}} = a_j^{(1)}$ (output dari hidden neuron $j$)

Sehingga delta output dapat dirumuskan sebagai:
$$\delta^{(2)} = -(y - \hat{y}) \cdot \hat{y}(1 - \hat{y})$$
Gradien bobot output:
$$\frac{\partial E}{\partial w_{j}^{(2)}} = a_j^{(1)} \cdot \delta^{(2)}$$

Langkah serupa diulangi untuk menghitung gradien pada hidden layer dengan merambatkan nilai $\delta^{(2)}$ ke belakang menggunakan bobot penghubungnya.

---

## 7. Pembaruan Bobot (Weight Update via Gradient Descent)
Setelah gradien untuk setiap bobot ($\frac{\partial E}{\partial W}$) dan bias ($\frac{\partial E}{\partial b}$) dihitung, kita memperbarui nilainya dengan arah yang berlawanan dari gradien untuk menurunkan nilai loss.
$$W_{\text{baru}} = W_{\text{lama}} - \eta \cdot \frac{\partial E}{\partial W}$$
$$b_{\text{baru}} = b_{\text{lama}} - \eta \cdot \frac{\partial E}{\partial b}$$
Di mana $\eta$ (eta) adalah **Learning Rate** (tingkat pembelajaran) yang mengontrol seberapa besar langkah penyesuaian yang diambil pada setiap iterasi pembelajaran.

---

## 8. Mengapa Neural Network Lebih Cocok?
Model linier sederhana (seperti Naive Bayes tanpa interaksi kata atau Logistic Regression linier) mengasumsikan kontribusi setiap fitur (kata) terhadap kelas target bersifat independen dan linier. Namun, dalam bahasa manusia (teks ulasan/komentar):
1. **Interaksi Kata**: Arti dari sebuah ulasan sering kali ditentukan oleh kombinasi kata, bukan hanya kata individual. Contohnya, "tidak buruk" memiliki makna positif, meskipun mengandung kata negatif "tidak" dan kata bernada kurang baik "buruk".
2. **Ketidaklinieran**: Hubungan antara fitur teks dan sentimen bersifat non-linear dan memiliki pola yang sangat kompleks. Neural Network dengan lapisan tersembunyi (*hidden layer*) dan fungsi aktivasi non-linear mampu memetakan interaksi antar-kata yang rumit ini ke dalam ruang representasi berdimensi tinggi untuk klasifikasi yang jauh lebih akurat.
