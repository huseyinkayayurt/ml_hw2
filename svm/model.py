from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(y_true, y_pred):
    """
    Performans metriklerini hesaplar.
    Args:
        y_true (array): Gerçek etiketler.
        y_pred (array): Tahmin edilen etiketler.
    Returns:
        dict: Accuracy, precision, recall, f1-score değerlerini içerir.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics


def train_evaluate_svm_with_params(kernel, C, gamma, X_train, y_train, X_val, y_val):
    """
    SVM modeli eğitir ve doğrulama setinde metrikleri hesaplar.
    Args:
        kernel (str): Kernel türü ('linear', 'poly', 'rbf').
        C (float): Regularization parametresi.
        gamma (float): Kernel parametresi (poly ve rbf için).
        X_train, y_train: Eğitim verileri.
        X_val, y_val: Doğrulama verileri.
    Returns:
        dict: Doğrulama metrikleri ve model.
    """
    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, y_val_pred),
        "precision": precision_score(y_val, y_val_pred, zero_division=0),
        "recall": recall_score(y_val, y_val_pred, zero_division=0),
        "f1": f1_score(y_val, y_val_pred, zero_division=0)
    }
    return metrics, model


def plot_decision_boundary_svm(model, X, y, title, save_path):
    """
    SVM karar sınırını çizdirir.
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    print(f"Karar sınırı kaydedildi: {save_path}")


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """
    Confusion matrix'i görselleştirir.
    Args:
        y_true (array): Gerçek etiketler.
        y_pred (array): Tahmin edilen etiketler.
        title (str): Grafik başlığı.
        save_path (str): Kaydedilecek dosya yolu.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.title(title)
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix kaydedildi: {save_path}")
