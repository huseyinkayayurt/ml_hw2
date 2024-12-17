import matplotlib.pyplot as plt


def save_plot(data, labels, filename):
    """
    Verilen veri kümesini görselleştirir ve dosyaya kaydeder.

    Parametreler:
    data (array): Veri noktalarının özellikleri.
    labels (array): Veri noktalarının sınıf etiketleri.
    filename (str): Kaydedilecek dosyanın adı.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolor='k')
    plt.title("make_moons ile Oluşturulan 2 Sınıflı Veri Kümesi")
    plt.xlabel("Özellik 1")
    plt.ylabel("Özellik 2")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Veri kümesi görselleştirmesi '{filename}' olarak kaydedildi.")
