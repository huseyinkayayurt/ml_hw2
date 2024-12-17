import matplotlib.pyplot as plt
import numpy as np


def plot_loss(training_loss, validation_loss, title="Loss Değişimi", save_path=None):
    """
    Eğitim ve doğrulama loss değerlerini çizdirir.

    Parametreler:
    - training_loss (list): Eğitim loss değerleri.
    - validation_loss (list): Doğrulama loss değerleri.
    - title (str): Grafik başlığı.
    - save_path (str): Grafiği kaydetme yolu. None ise gösterilir.
    """
    epochs = len(training_loss)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), training_loss, label="Eğitim Loss")
    plt.plot(range(1, epochs + 1), validation_loss, label="Doğrulama Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path)
        print(f"Grafik kaydedildi: {save_path}")
    else:
        plt.show()


def plot_decision_boundary(model, X, y, title="Karar Sınırı", save_path=None):
    """
    Modelin karar sınırını görselleştirir.

    Parametreler:
    - model (ANN): Eğitilmiş Yapay Sinir Ağı modeli.
    - X (array): Veri kümesi (girişler).
    - y (array): Etiketler.
    - title (str): Grafik başlığı.
    - save_path (str): Grafiği kaydetme yolu. None ise gösterilir.
    """
    h = 0.1  # Ağ grid aralığı
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Tüm grid noktaları için tahmin yap
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.forward(grid_points)
    Z = np.round(Z).reshape(xx.shape)

    # Grafik çizimi
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap="coolwarm")
    plt.title(title)
    plt.xlabel("Özellik 1")
    plt.ylabel("Özellik 2")

    if save_path:
        plt.savefig(save_path)
        print(f"Karar sınırı grafiği kaydedildi: {save_path}")
    else:
        plt.show()
