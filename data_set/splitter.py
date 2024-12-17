import numpy as np
from sklearn.utils import shuffle


def split_data(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    """
    Veri kümesini eğitim, doğrulama ve test setlerine ayırır.

    Parametreler:
    X (array): Özellikler (girdi verileri).
    y (array): Etiketler (çıkış verileri).
    train_ratio (float): Eğitim seti oranı.
    val_ratio (float): Doğrulama seti oranı.
    test_ratio (float): Test seti oranı.
    random_state (int): Rastgelelik için başlangıç değeri.

    Geri Dönüş:
    tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    # Veriyi rastgele karıştır
    X, y = shuffle(X, y, random_state=random_state)

    # Veri seti uzunluğu
    total_samples = len(X)

    # Veri setinin bölünme noktalarını hesapla
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)

    # Eğitim, doğrulama ve test setlerini oluştur
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"Veri kümesi bölündü:\n- Eğitim: {len(X_train)} örnek\n- Doğrulama: {len(X_val)} örnek\n- Test: {len(X_test)} örnek")

    return X_train, y_train, X_val, y_val, X_test, y_test
