import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def confusion_matrix(y_true, y_pred):
    """
    Confusion matrix hesaplama.
    Args:
        y_true: Gerçek etiketler.
        y_pred: Tahmin edilen etiketler.
    Returns:
        tn, fp, fn, tp: Confusion matrix elemanları.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positive
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negative
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positive
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negative

    return tn, fp, fn, tp


def plot_confusion_matrix(y_true, y_pred, set_name, model_name):
    """
    Confusion matrix'i görselleştirir ve kaydeder.
    Args:
        y_true: Gerçek etiketler.
        y_pred: Tahmin edilen etiketler.
        set_name: Veri seti adı (Eğitim, Doğrulama, Test).
        model_name: Model adı (1 Layer, 2 Layers, vb.)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)
    matrix = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negatif", "Pozitif"], yticklabels=["Negatif", "Pozitif"])
    plt.title(f"Confusion Matrix - {set_name} ({model_name})")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()

    filename = f"results/confusion_matrix_{set_name.lower()}_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Confusion Matrix kaydedildi: {filename}")


def calculate_metrics(y_true, y_pred):
    """
    Accuracy, Precision, Recall ve F1-Score hesaplama.
    Args:
        y_true: Gerçek etiketler.
        y_pred: Tahmin edilen etiketler.
    Returns:
        accuracy, precision, recall, f1_score
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

    return accuracy, precision, recall, f1_score


def print_metrics(set_name, y_true, y_pred, model_name):
    """
    Metrikleri hesaplar, ekrana yazdırır ve confusion matrix'i kaydeder.
    Args:
        set_name: Veri seti adı (Eğitim, Doğrulama, Test).
        y_true: Gerçek etiketler.
        y_pred: Tahmin edilen etiketler.
        model_name: Model adı.
    """

    accuracy, precision, recall, f1_score = calculate_metrics(y_true, y_pred)
    print("-" * 30)
    print(f"{set_name} Seti Sonuçları ({model_name}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")

    # Confusion Matrix görselleştirme
    plot_confusion_matrix(y_true, y_pred, set_name, model_name)

    return accuracy, precision, recall, f1_score


def plot_metrics_bar_text(metrics, model_name):
    """
    Eğitim, doğrulama ve test setleri için metrik değerlerini bar grafiği ile görselleştirir.
    Args:
        metrics: Eğitim, doğrulama ve test setleri için metrikler.
                 [(eğitim_metricleri), (doğrulama_metricleri), (test_metricleri)]
        model_name: Model adı.
    """
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    set_names = ["Eğitim", "Doğrulama", "Test"]

    metrics_array = np.array(metrics)
    x = np.arange(len(metric_names))  # Metriklerin x ekseni pozisyonları

    bar_width = 0.2
    offsets = [-bar_width, 0, bar_width]  # Eğitim, doğrulama, test için ofsetler

    plt.figure(figsize=(10, 6))
    for i, set_name in enumerate(set_names):
        bars = plt.bar(x + offsets[i], metrics_array[i], bar_width, label=set_name)

        # Barların üzerine metrik değerlerini yazdır
        for bar, value in zip(bars, metrics_array[i]):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{value:.2f}', ha='center', va='bottom', fontsize=10)

    plt.title(f"Metrik Değerleri - {model_name}")
    plt.xlabel("Metrikler")
    plt.ylabel("Değer")
    plt.xticks(x, metric_names)
    plt.ylim(0, 1.1)
    plt.legend(title="Veri Seti")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    filename = f"results/metrics_bar_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Metrik bar grafiği kaydedildi: {filename}")
