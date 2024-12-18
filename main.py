import time
import os

from ann.metrics import print_metrics, plot_metrics_bar_text
from data_set.generator import MoonDataGenerator
from data_set.splitter import split_data
from data_set.visualize import save_plot
from ann.model import ANN
from ann.train import train_sgd, train_batch_gd, train_mini_batch_gd, evaluate_model
from ann.visualize import plot_loss, plot_decision_boundary


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
    os.makedirs(results_dir, exist_ok=True)

    # Model parametreleri ve eğitim fonksiyonları
    models = [
        {"name": "1 Gizli Katman - SGD", "hidden_layers": [6], "learning_rate": 0.1, "train_func": train_sgd, "train_params": {"epochs": 1000}},
        {"name": "1 Gizli Katman - Batch", "hidden_layers": [8], "learning_rate": 0.1, "train_func": train_batch_gd,
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


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
