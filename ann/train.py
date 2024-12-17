import numpy as np
from ann.model import ANN


def train_sgd(model, X_train, y_train, X_val, y_val, epochs=1000):
    """
    Yapay Sinir Ağı modeli için Stochastic Gradient Descent (SGD) ile eğitim yapar.

    Parametreler:
    - model (ANN): Yapay Sinir Ağı modeli.
    - X_train (array): Eğitim verisi.
    - y_train (array): Eğitim etiketleri.
    - X_val (array): Doğrulama verisi.
    - y_val (array): Doğrulama etiketleri.
    - epochs (int): Epoch sayısı.
    """
    training_loss = []
    validation_loss = []

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(X_train)):
            X_sample = X_train[i].reshape(1, -1)
            y_sample = y_train[i].reshape(1, -1)

            # İleri ve geri yayılım
            output = model.forward(X_sample)
            gradients = model.backward(X_sample, y_sample, output)
            model.update_weights(gradients)

            # Kayıp hesapla
            loss = model.binary_cross_entropy(y_sample, output)
            epoch_loss += loss

        avg_epoch_loss = epoch_loss / len(X_train)
        training_loss.append(avg_epoch_loss)

        # Doğrulama verisi ile kayıp hesapla
        val_output = model.forward(X_val)
        val_loss = model.binary_cross_entropy(y_val, val_output)
        validation_loss.append(val_loss)

        # Epoch bilgisi yazdır
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

    return training_loss, validation_loss


def train_batch_gd(model, X_train, y_train, X_val, y_val, epochs=1000):
    """
    Batch Gradient Descent algoritması ile modeli eğitir.
    """
    training_loss = []
    validation_loss = []

    for epoch in range(epochs):
        # Tüm eğitim verisi üzerinde ileri yayılım
        output = model.forward(X_train)

        # Geri yayılım ve ağırlık güncelleme
        gradients = model.backward(X_train, y_train, output)
        model.update_weights(gradients)

        # Kayıp hesapla
        train_loss = model.binary_cross_entropy(y_train, output)
        training_loss.append(train_loss)

        # Doğrulama verisi için kayıp hesapla
        val_output = model.forward(X_val)
        val_loss = model.binary_cross_entropy(y_val, val_output)
        validation_loss.append(val_loss)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    return training_loss, validation_loss


def train_mini_batch_gd(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=1000):
    """
    Mini Batch Gradient Descent algoritması ile modeli eğitir.
    """
    training_loss = []
    validation_loss = []
    m = len(X_train)

    for epoch in range(epochs):
        epoch_loss = 0

        # Veriyi karıştır
        indices = np.arange(m)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for i in range(0, m, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # İleri ve geri yayılım
            output = model.forward(X_batch)
            gradients = model.backward(X_batch, y_batch, output)
            model.update_weights(gradients)

            # Kayıp hesapla
            batch_loss = model.binary_cross_entropy(y_batch, output)
            epoch_loss += batch_loss

        avg_epoch_loss = epoch_loss / (m // batch_size)
        training_loss.append(avg_epoch_loss)

        # Doğrulama verisi için kayıp hesapla
        val_output = model.forward(X_val)
        val_loss = model.binary_cross_entropy(y_val, val_output)
        validation_loss.append(val_loss)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

    return training_loss, validation_loss
