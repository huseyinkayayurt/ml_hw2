import time
import os

from data_set.generator import MoonDataGenerator
from data_set.splitter import split_data
from data_set.visualize import save_plot
from ann.model import ANN
from ann.train import train_sgd, train_batch_gd, train_mini_batch_gd
from ann.visualize import plot_loss, plot_decision_boundary


def train_and_evaluate(hidden_layers, X_train, y_train, X_val, y_val, training_method, method_name):
    """
    Verilen gizli katman sayısına ve eğitim metoduna göre ağı eğitir ve sonuçları kaydeder.
    """
    print(f"\n{len(hidden_layers)} Gizli Katmanlı Ağ - {method_name} ile Eğitim:")
    model = ANN(input_dim=2, hidden_layers=hidden_layers, output_dim=1, learning_rate=0.1)

    # Eğitim
    if training_method == "sgd":
        train_loss, val_loss = train_sgd(model, X_train, y_train, X_val, y_val, epochs=1000)
    elif training_method == "batch_gd":
        train_loss, val_loss = train_batch_gd(model, X_train, y_train, X_val, y_val, epochs=1000)
    elif training_method == "mini_batch_gd":
        train_loss, val_loss = train_mini_batch_gd(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=1000)
    else:
        raise ValueError("Geçersiz eğitim yöntemi")

    # Loss grafiği
    plot_loss(train_loss, val_loss, f"{method_name} Loss Grafiği ({len(hidden_layers)} Katman)",
              save_path=f"results/{method_name.lower()}_loss_{len(hidden_layers)}layer.png")

    # Karar sınırı grafiği
    plot_decision_boundary(model, X_train, y_train,
                           f"{method_name} Karar Sınırı ({len(hidden_layers)} Katman)",
                           save_path=f"results/{method_name.lower()}_boundary_{len(hidden_layers)}layer.png")


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
    os.makedirs("results", exist_ok=True)

    # Gizli katman yapıları
    layer_configs = {
        "1 Katman": [8],
        "2 Katman": [16, 8],
        "3 Katman": [32, 16, 8]
    }

    # Eğitim yöntemleri
    training_methods = {
        "SGD": "sgd",
        "Batch GD": "batch_gd",
        "Mini-Batch GD": "mini_batch_gd"
    }

    # Tüm yapı ve eğitim yöntemleri için döngü
    for config_name, hidden_layers in layer_configs.items():
        for method_name, method in training_methods.items():
            train_and_evaluate(hidden_layers, X_train, y_train, X_val, y_val, method, method_name)

    print("\nEğitim tamamlandı. Sonuçlar 'results/' klasörüne kaydedildi.")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
