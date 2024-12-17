from sklearn.datasets import make_moons


class MoonDataGenerator:
    """
    Bu sınıf, sklearn 'make_moons' fonksiyonunu kullanarak 2 sınıflı veri kümesi oluşturur.
    """

    def __init__(self, n_samples=400, noise=0.1, random_state=42):
        """
        Parametreler:
        n_samples (int): Veri kümesindeki örnek sayısı.
        noise (float): Gürültü seviyesi.
        random_state (int): Rastgelelik için başlangıç değeri.
        """
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state
        self.data = None
        self.labels = None

    def generate_data(self):
        """
        make_moons fonksiyonunu kullanarak veri kümesini oluşturur.
        """
        self.data, self.labels = make_moons(n_samples=self.n_samples,
                                            noise=self.noise,
                                            random_state=self.random_state)
        self.labels = self.labels.reshape(-1, 1)
        
        print(f"Veri kümesi {self.n_samples} örnek ile başarıyla oluşturuldu.")
        return self.data, self.labels
