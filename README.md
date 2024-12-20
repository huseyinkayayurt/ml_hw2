# Yapay Nöron Ağları (ANN) ve Destek Vektör Makineleri (SVM) ile Sınıflandırma

## Proje Açıklaması

Bu proje, yapay nöron ağları (ANN) ve destek vektör makineleri (SVM) kullanılarak iki sınıflı bir veri kümesinde
sınıflandırma probleminin çözümüne odaklanmaktadır. Proje, scikit-learn ve numpy gibi popüler Python kütüphanelerini
kullanarak eğitim, doğrulama ve test veri setleri üzerinde performans analizi yapar. Her iki model için de çeşitli
öğrenme yöntemleri ve hiperparametreler test edilmiştir.

## Özellikler

- **Veri Kümesi:** Scikit-learn'ün `make_moons` fonksiyonu kullanılarak oluşturulan 400 örnek.
- **Model Türleri:**
    - Yapay Nöron Ağları (ANN)
    - Destek Vektör Makineleri (SVM)
- **Öğrenme Yöntemleri:**
    - ANN: Stochastic Gradient Descent (SGD), Batch Gradient Descent, Mini-Batch Gradient Descent
    - SVM: Linear, Polynomial, Gaussian RBF Kernel
- **Performans Ölçütleri:**
    - Accuracy (Doğruluk)
    - Precision (Hassasiyet)
    - Recall (Duyarlılık)
    - F1-Score
    - Confusion Matrix (Karışıklık Matrisi)

## Kullanılan Teknolojiler

- Python 3.9
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn

## Proje Yapısı

```
project_root/
├── data_set/
│   ├── generator.py          # Veri kümesi oluşturma
│   ├── splitter.py           # Veri seti ayırma
│   ├── visualize.py          # Veri kümesi görselleştirme
├── ann/
│   ├── model.py              # ANN modeli
│   ├── train.py              # Eğitim fonksiyonları
│   ├── visualize.py          # Grafik ve karar sınırı görselleştirme
│   ├── metrics.py            # Metrik hesaplama ve görselleştirme
├── svm/
│   ├── model.py              # SVM eğitim ve değerlendirme
├── results/                  # Sonuç görselleri ve raporlar
├── main.py                   # Ana çalıştırma dosyası
├── requirements.txt          # Gereksinim dosyası
└── README.md                 # Proje açıklaması
```

## Kurulum

1. Projeyi klonlayın:
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2. Sanal ortam oluşturun ve aktif edin:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # Windows için: .venv\Scripts\activate
    ```

3. Gereksinimleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

## Çalıştırma

1. Ana dosyayı çalıştırın:
    ```bash
    python main.py
    ```

2. Sonuçlar `results/` ve `results_svm/` dizinlerinde görseller ve metrikler olarak kaydedilir.

## Kullanım Senaryoları

1. Veri kümesi oluşturulup eğitim, doğrulama ve test olarak ayrılır.
2. ANN modelleri (1, 2 ve 3 gizli katmanlı) için SGD, Batch GD ve Mini-Batch GD yöntemleriyle eğitim yapılır.
3. SVM modelleri için Linear, Polynomial ve Gaussian RBF kernel türleri üzerinde hiperparametre optimizasyonu
   gerçekleştirilir.
4. Her model için:
    - Loss grafikleri çizilir.
    - Karar sınırları görselleştirilir.
    - Performans metrikleri hesaplanır ve raporlanır.

## Örnek Sonuçlar

- **En Başarılı ANN Modeli:** 3 Gizli Katmanlı ANN (Mini-Batch Gradient Descent)
    - Accuracy: 96.25%
    - F1-Score: 0.9577

- **En Başarılı SVM Modeli:** Gaussian RBF Kernel
    - Accuracy: 96.25%
    - F1-Score: 0.9589
