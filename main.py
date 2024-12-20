import time
import os
import itertools

from ann.metrics import print_metrics, plot_metrics_bar_text
from data_set.generator import MoonDataGenerator
from data_set.splitter import split_data
from data_set.visualize import save_plot
from ann.model import ANN
from ann.train import train_sgd, train_batch_gd, train_mini_batch_gd, evaluate_model
from ann.visualize import plot_loss, plot_decision_boundary
from svm.model import train_evaluate_svm_with_params, plot_decision_boundary_svm, compute_metrics, plot_confusion_matrix


def train_and_evaluate(model_name, model, train_func, X_train, y_train, X_val, y_val, X_test, y_test, train_params, results_dir):
    """
    Modeli eğitir, değerlendirir ve sonuçları görselleştirir.

    Args:
        model_name (str): Modelin adı.
        model (ANN): ANN modeli.
        train_func (function): Eğitim fonksiyonu (train_sgd, train_batch_gd, vb.).
        X_train, y_train: Eğitim verileri.
        X_val, y_val: Doğrulama verileri.
        X_test, y_test: Test verileri.
        train_params (dict): Eğitim parametreleri (epochs, batch_size, vb.).
        results_dir (str): Sonuçların kaydedileceği dizin.

    Returns:
        None
    """
    print(f"\n{model_name} - Eğitim Başlıyor:")

    # Modeli eğit
    train_loss, val_loss = train_func(model, X_train, y_train, X_val, y_val, **train_params)

    # Tahminler
    y_train_pred = evaluate_model(model, X_train, y_train)
    y_val_pred = evaluate_model(model, X_val, y_val)
    y_test_pred = evaluate_model(model, X_test, y_test)

    # Metrikleri yazdır ve görselleştir
    train_metrics = print_metrics("Eğitim", y_train, y_train_pred, model_name)
    val_metrics = print_metrics("Doğrulama", y_val, y_val_pred, model_name)
    test_metrics = print_metrics("Test", y_test, y_test_pred, model_name)

    plot_metrics_bar_text([train_metrics, val_metrics, test_metrics], model_name)
    plot_loss(train_loss, val_loss, f"{model_name} Loss Grafiği", save_path=f"{results_dir}/{model_name.lower()}_loss.png")
    plot_decision_boundary(model, X_train, y_train, f"{model_name} Karar Sınırı", save_path=f"{results_dir}/{model_name.lower()}_boundary.png")


def evaluate_best_model(best_model, X_train, y_train, X_val, y_val, X_test, y_test, folder):
    """
    En iyi modeli eğitim, doğrulama ve test veri setlerinde değerlendirir.
    Args:
        best_model: En iyi SVM modeli.
        X_train, y_train: Eğitim veri seti.
        X_val, y_val: Doğrulama veri seti.
        X_test, y_test: Test veri seti.
    """
    # Eğitim, doğrulama ve test tahminleri
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    # Performans metrikleri
    train_metrics = compute_metrics(y_train, y_train_pred)
    val_metrics = compute_metrics(y_val, y_val_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    # Confusion matrix görselleştirme
    plot_confusion_matrix(y_train, y_train_pred, "Eğitim Seti - Confusion Matrix", f"{folder}/confusion_matrix_train.png")
    plot_confusion_matrix(y_val, y_val_pred, "Doğrulama Seti - Confusion Matrix", f"{folder}/confusion_matrix_val.png")
    plot_confusion_matrix(y_test, y_test_pred, "Test Seti - Confusion Matrix", f"{folder}/confusion_matrix_test.png")

    # Performans metriklerini konsola yazdır
    print("\nPerformans Metrikleri:")
    print(f"Eğitim Seti: {train_metrics}")
    print(f"Doğrulama Seti: {val_metrics}")
    print(f"Test Seti: {test_metrics}")


def print_best_model_info(best_model, best_params):
    """
    En iyi modelin kernel ve parametre bilgilerini yazdırır.
    Args:
        best_model: En iyi SVM modeli.
        best_params: En iyi modelin parametreleri.
    """
    print("\nEn İyi Model Bilgileri:")
    print(f"Kernel Türü: {best_params.get('kernel', 'Bilinmiyor')}")
    print(f"C (Regularization): {best_params.get('C', 'Bilinmiyor')}")
    print(f"Gamma: {best_params.get('gamma', 'Bilinmiyor')}")
    print(f"Degree (Polinom Kernel için): {best_params.get('degree', 'Bilinmiyor')}")

    # Modelin özeti
    print("\nModel Özeti:")
    print(best_model)


def main():
    # Veri kümesi oluşturucu sınıfını başlat
    generator = MoonDataGenerator(n_samples=400, noise=0.2, random_state=42)

    # Veri kümesini oluştur
    data, labels = generator.generate_data()

    # Grafiği dosyaya kaydet
    save_plot(data, labels, filename="data_set/make_moons_dataset.png")

    # Veri kümesini ayır
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(data, labels)

    save_plot(X_train, y_train, filename="data_set/make_moons_dataset_train.png")
    save_plot(X_val, y_val, filename="data_set/make_moons_dataset_val.png")
    save_plot(X_test, y_test, filename="data_set/make_moons_dataset_test.png")

    # Klasörleri oluştur
    results_dir = "results"
    results_svm_dir = "results_svm"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(results_svm_dir, exist_ok=True)

    # Model parametreleri ve eğitim fonksiyonları
    models = [
        {"name": "1 Gizli Katman - SGD", "hidden_layers": [6], "learning_rate": 0.1, "train_func": train_sgd, "train_params": {"epochs": 1000}},
        {"name": "1 Gizli Katman - Batch", "hidden_layers": [6], "learning_rate": 0.1, "train_func": train_batch_gd,
         "train_params": {"epochs": 1500}},
        {"name": "1 Gizli Katman - Mini-Batch", "hidden_layers": [6], "learning_rate": 0.1, "train_func": train_mini_batch_gd,
         "train_params": {"batch_size": 16, "epochs": 200}},
        {"name": "2 Gizli Katman - SGD", "hidden_layers": [10, 6], "learning_rate": 0.1, "train_func": train_sgd, "train_params": {"epochs": 1800}},
        {"name": "2 Gizli Katman - Batch", "hidden_layers": [10, 6], "learning_rate": 1.0, "train_func": train_batch_gd,
         "train_params": {"epochs": 2500}},
        {"name": "2 Gizli Katman - Mini-Batch", "hidden_layers": [10, 6], "learning_rate": 0.1, "train_func": train_mini_batch_gd,
         "train_params": {"batch_size": 16, "epochs": 1000}},
        {"name": "3 Gizli Katman - SGD", "hidden_layers": [14, 10, 6], "learning_rate": 0.1, "train_func": train_sgd,
         "train_params": {"epochs": 1300}},
        {"name": "3 Gizli Katman - Batch", "hidden_layers": [14, 10, 6], "learning_rate": 5.0, "train_func": train_batch_gd,
         "train_params": {"epochs": 2000}},
        {"name": "3 Gizli Katman - Mini-Batch", "hidden_layers": [14, 10, 6], "learning_rate": 0.2, "train_func": train_mini_batch_gd,
         "train_params": {"batch_size": 8, "epochs": 1000}},
    ]

    # Modelleri eğit ve değerlendir
    for model_config in models:
        model = ANN(input_dim=2, hidden_layers=model_config["hidden_layers"], output_dim=1, learning_rate=model_config["learning_rate"])
        train_and_evaluate(model_config["name"], model, model_config["train_func"], X_train, y_train, X_val, y_val, X_test, y_test,
                           model_config["train_params"], results_dir)

    ###### SVM #####
    # Parametre Grid'i
    kernels = ['linear', 'poly', 'rbf']
    C_values = [0.1, 1, 10]
    gamma_values = [0.01, 0.1, 1]  # Sadece poly ve rbf için

    best_params = {}
    best_model = None
    metrics = None
    y_train = y_train.ravel()
    y_val = y_val.ravel()

    for kernel in kernels:
        best_score = -1
        best_model = None
        print(f"\nSVM - {kernel.upper()} kernel denemeleri:")

        # Parametre kombinasyonlarını oluştur
        if kernel in ['linear']:
            param_grid = itertools.product(C_values, [None])  # Linear için gamma yok
        else:
            param_grid = itertools.product(C_values, gamma_values)

        for C, gamma in param_grid:
            print(f"Denenen parametreler -> C: {C}, Gamma: {gamma}")

            # Modeli eğit ve doğrulama metriklerini hesapla
            gamma_param = 'scale' if gamma is None else gamma
            metrics, model = train_evaluate_svm_with_params(kernel, C, gamma_param, X_train, y_train, X_val, y_val)

            print(
                f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1']:.4f}")

            # En iyi sonucu kontrol et
            if metrics['f1'] > best_score:
                best_score = metrics['f1']
                best_params[kernel] = {"C": C, "gamma": gamma, "metrics": metrics}
                best_params = {  # En iyi parametreleri doldur
                    "kernel": kernel,
                    "C": C,
                    "gamma": gamma if gamma is not None else 'scale',
                    "degree": 'Bilinmiyor' if kernel != 'poly' else 'Polinom Derecesi'  # Polinom kernel yoksa bilinmiyor
                }
                best_model = model

        # En iyi modeli ve karar sınırını kaydet
        print(f"\nEn iyi {kernel.upper()} kernel parametreleri: C: {best_params['C']}, Gamma: {best_params['gamma']}")
        print(f"Metrics: {metrics}")
        plot_decision_boundary_svm(best_model, X_train, y_train, f"SVM {kernel.upper()} En İyi Karar Sınırı",
                                   f"{results_svm_dir}/svm_best_boundary_{kernel}.png")

        # Tüm kernel'ler için en iyi sonuçları yazdır
    print("\nEn İyi SVM Sonuçları:")
    print_best_model_info(best_model, best_params)

    # En iyi modeli yazdır
    print_best_model_info(best_model, best_params)
    # En iyi modeli değerlendirin
    print("\nEn iyi model ile değerlendirme başlıyor...")
    evaluate_best_model(best_model, X_train, y_train, X_val, y_val, X_test, y_test, results_svm_dir)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
