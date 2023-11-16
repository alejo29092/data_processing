import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
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


#test_graphic_double_3(var_3)

def drop_var_outlier(df):
    data.describe()
    mean = df['PurchaseAmount'].mean()
    q1 = df['PurchaseAmount'].quantile(0.25)
    q3 = df['PurchaseAmount'].quantile(0.75)

    umbral= q3-q1

    standard_desviation = df['PurchaseAmount'].std()
    upper_limit = (mean + umbral * standard_desviation)
    #return (upper_limit)
    #values = q3 + umbral * 1.5
    #return (values)

    df_filtrado = df.drop[(df['PurchaseAmount'] >= mean - umbral * standard_desviation) &
                     (df['PurchaseAmount'] <= mean + umbral * standard_desviation)]
    #return df_filtrado


data_ver_3 = drop_var_outlier(data_2)
#print(data_ver_3)
#data_ver_3.info()
#print(drop_var_outlier(data_2))




def valid_email(email):
    """
    exprexion regular used of : https://parzibyte.me/blog/2018/12/04/comprobar-correo-electronico-python/
    """
    expresion_regular = r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`" \
                        r"{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\
                        x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\
                        [(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9]" \
                        r"[0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\" \
                        r"[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"
     #return re.match(expresion_regular, email) is not None
    #for correo in data_ver_3:
     #   if valid_email(data_ver_3[data_ver_3('Email')]) != is not None:
      #      data_ver_3 = data_ver_3.drop[data_ver_3('Email')]


def valid_email_2(email):

    expresion_regular = r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"


    filas_con_correos_invalidos = []

    # Itera a través de las filas del DataFrame
    iterrows = email.iterrows()
    for index, row in iterrows:
        correo = row['Email']  # Supongo que la columna se llama 'Email'
        if not re.match(expresion_regular, correo):
            filas_con_correos_invalidos.append(index)

    # Elimina las filas con correos electrónicos no válidos
    email = email.drop(filas_con_correos_invalidos)

data_ver4= valid_email_2(data_ver_3)
# def table_use():
# df_loans = pd.read_csv(df_rooute_3_loans, delimiter=',', encoding=encoding)
# data_age = data.loc[:, ['Age']]
# print(data_age)

# Lectura del archivo
# df_books = pd.read_csv(df_rooute_1_books, delimiter=',', encoding=encoding)
