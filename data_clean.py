import pandas as pd
import matplotlib as plt

data = pd.read_json("C:/Users/alejo/Desktop/pythonProject7/dataset.json")


def clean(a):
    """
    objetive: Remove rows where any essential column (CustomerID, FirstName, LastName, Email, PhoneNumber) is missing.
    """
    a = a.drop_duplicates(['CustomerID'])

    return a

def test_graphic_double_3(data):
    # data = np.random.randn(100)

    fig, ax = plt.subplots(figsize=(8, 6))

    boxplot = ax.boxplot(data, vert=False)

    y = np.zeros(len(data))
    ax.scatter(data, y, c='red', marker='o')

    ax.set_yticklabels(['Datos'])  # Etiqueta del eje Y
    ax.set_title('Boxplot con Datos Superpuestos')

    plt.show()
data_vers_2 = clean(data)
data_vers_2.info()
