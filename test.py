import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_json("C:/Users/alejo/Desktop/pythonProject7/dataset.json")
print(data)


def fate_ejem(data):
    df = data
    book_df = df.loc[:, ['published_year', 'book_title', 'author_name']]

    book_df.loc[
        book_df['published_year'].isna(), 'published_year'] = 0  # rellenar valores nullos con un numero por defecto
    # book_df['published_year'] = pd.to_numeric(book_df['published_year'], errors="ignore", downcast='integer')
    book_df['published_year'] = book_df['published_year'].astype('int16', errors="ignore")

    book_df = book_df.drop_duplicates(subset=['book_title', 'author_name'])
    print(book_df)

    book_df = book_df.rename(columns={'book_title': 'name_book'})
    print(book_df)
    book_df = book_df.loc[:, ['published_year', 'name_book']]

    # Export a la tabla libros (Load)
    # book_df.to_sql(name='books', con=db_conn, index=False, if_exists='append')


def clean(a):
    a = a.drop_duplicates(['CustomerID'])
    # CustomerID, FirstName, LastName, Email, PhoneNumber
    a = a.dropna(subset=['CustomerID', 'FirstName', 'LastName', 'Email', 'PhoneNumber'])

    return a


def confirm_data(a):
    # CustomerID, FirstName,LastName, Email,PhoneNumber, Age,Gender,Address,PurchaseAmount
    # 'FirstName', 'LastName','Email', 'PhoneNumber', 'Gender','Address'

    a.info()
    cols_cat = ['FirstName', 'LastName', 'Email', 'PhoneNumber', 'Gender', 'Address']

    for col in cols_cat:
        print(f'column: {col}: {a[col].nunique()}, sublevel')


data_2 = clean(data)
confirm_data(data_2)


def outliers_var(a):
    plt.boxplot(a)

    plt.show()


var_2 = data.loc[:, ['Age']]
var_3 = data.loc[:, ['PurchaseAmount']]


# outliers_var(var_2)
def test_graphic_double():
    data = [np.random.normal(0, 1, 100), np.random.normal(0, 2, 100)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    ax1.boxplot(data)
    ax1.set_title('Boxplot')

    x = np.arange(1, len(data) + 1)
    for i, variable in enumerate(data):
        jittered_x = x[i] + 0.2 * np.random.rand(len(variable))
        ax2.scatter(jittered_x, variable, label=f'Variable {i + 1}')

    ax2.set_xticks(x)
    ax2.set_xticklabels(['Variable 1', 'Variable 2'])
    ax2.set_title('Scatter Plot')
    ax2.legend()

    plt.tight_layout()

    plt.show()


# test_graphic_double()
def test_graphic_double_2():
    data = np.random.randn(100)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.boxplot(data, vert=False)

    ax.plot(data, np.zeros_like(data), 'ro', alpha=0.6)

    ax.set_yticklabels(['Datos'])  # Etiqueta del eje Y
    ax.set_title('Boxplot con Datos Superpuestos')
    plt.show()


# test_graphic_double_2()

def test_graphic_double_3(data):
    # data = np.random.randn(100)

    fig, ax = plt.subplots(figsize=(8, 6))

    boxplot = ax.boxplot(data, vert=False)

    y = np.zeros(len(data))
    ax.scatter(data, y, c='red', marker='o')

    ax.set_yticklabels(['Datos'])  # Etiqueta del eje Y
    ax.set_title('Boxplot con Datos Superpuestos')

    plt.show()


test_graphic_double_3(var_3)
