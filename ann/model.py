import numpy as np


class ANN:
    """
    Yapay Sinir Ağı (ANN) Modeli.
    1, 2 veya 3 gizli katmanlı ağların oluşturulmasını sağlar.
    """

    def __init__(self, input_dim, hidden_layers, output_dim=1, learning_rate=0.01):
        """
        Parametreler:
        - input_dim (int): Giriş katmanı boyutu.
        - hidden_layers (list): Her bir gizli katmanın nöron sayısı.
        - output_dim (int): Çıkış katmanı boyutu. Varsayılan: 1.
        - learning_rate (float): Öğrenme oranı.
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.weights = self._initialize_weights()

    def _initialize_weights(self):
        """
        Ağın ağırlıklarını rastgele başlatır.
        """
        layers = [self.input_dim] + self.hidden_layers + [self.output_dim]
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1]) * 0.01
            b = np.zeros((1, layers[i + 1]))
            weights.append((w, b))
        return weights

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid aktivasyon fonksiyonu.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        """
        Binary Cross Entropy loss fonksiyonu.
        """
        epsilon = 1e-9  # Log hesaplamalarında sıfır bölmeyi önlemek için epsilon
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def forward(self, X):
        """
        İleri yayılım (Forward Propagation).
        """
        activations = X
        self.cache = []  # Aktivasyonları ve ağırlıkları kaydeder
        for w, b in self.weights[:-1]:  # Gizli katmanlar
            z = np.dot(activations, w) + b
            activations = self.sigmoid(z)
            self.cache.append((z, activations))
        # Çıkış katmanı
        w, b = self.weights[-1]
        z = np.dot(activations, w) + b
        output = self.sigmoid(z)
        self.cache.append((z, output))
        return output

    def backward(self, X, y_true, output):
        """
        Geri yayılım (Backward Propagation).
        """
        m = X.shape[0]
        gradients = []
        dA = -(y_true / output - (1 - y_true) / (1 - output))

        for i in reversed(range(len(self.weights))):
            z, activation = self.cache[i]
            dZ = dA * activation * (1 - activation)  # Sigmoid türevi
            A_prev = self.cache[i - 1][1] if i > 0 else X
            dW = np.dot(A_prev.T, dZ) / m
            dB = np.sum(dZ, axis=0, keepdims=True) / m
            dA = np.dot(dZ, self.weights[i][0].T)
            gradients.insert(0, (dW, dB))

        return gradients

    def update_weights(self, gradients):
        """
        Ağırlıkları günceller.
        """
        for i, (dW, dB) in enumerate(gradients):
            self.weights[i] = (
                self.weights[i][0] - self.learning_rate * dW,
                self.weights[i][1] - self.learning_rate * dB,
            )
