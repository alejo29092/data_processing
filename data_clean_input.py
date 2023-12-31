import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

with open('password.txt') as file_object:
    password = file_object.read()
data = pd.read_json("C:/Users/alejo/Desktop/pythonProject7/dataset.json")
db_conn = create_engine(f'mysql+pymysql://root:{password}@localhost/data_purchase_amount')


# db_conn = create_engine(url=f'mysql://root:123456@localhost/librarys')

def clean(a):
    """
    Remove rows where any essential column repeated (CustomerID, FirstName, LastName, Email, PhoneNumber).
    a = data brought from the store.
    """
    a = a.drop_duplicates(['CustomerID'])
    # CustomerID, FirstName, LastName, Email, PhoneNumber
    a = a.dropna(subset=['CustomerID', 'FirstName', 'LastName', 'Email', 'PhoneNumber'])

    return a


def confirm_data(a):
    """
    this function to confirm unique values.

    a = data brought from the store, preprocessed
    """
    # a.info() = to know that data i work with
    cols_cat = ['FirstName', 'LastName', 'Email', 'PhoneNumber', 'Gender', 'Address']

    for col in cols_cat:
        print(f'column: {col}: {a[col].nunique()}, sublevel')


def test_graphic_double_3(a):
    """
    here confirm the outlier data. the case not its, but i also did the respective cleaning exercise.
    a = data brought from the store, preprocessed
    """
    # data = np.random.randn(100)

    fig, ax = plt.subplots(figsize=(8, 6))

    boxplot = ax.boxplot(a, vert=False)

    y = np.zeros(len(a))
    ax.scatter(a, y, c='red', marker='o')

    ax.set_yticklabels(['Datos'])  # Etiqueta del eje Y
    ax.set_title('Boxplot con Datos Superpuestos')

    plt.show()


data_1 = clean(data)
# confirm_data(data_2)


data_vers_2 = clean(data)


# data_vers_2.info()


# test_graphic_double_3(var_3)


def drop_var_outlier(df, x):
    """
    here I removed the outliers, but since there wasn't one had to enlarge the function
    df = data brought from the store, preprocessed
    x = columns locate the outlier data
    """
    try:
        data_out = df.copy()
        q1 = df[x].quantile(0.25)
        q3 = df[x].quantile(0.75)
        threshold = q3 - q1

        df_filtrado = df[(df[x] <= q1 - threshold * 1.5) & (df[x] >= q3 + threshold * 1.5)]

        data_out = data_out.drop(df_filtrado.index)
        if data_out.empty:
            return df.copy()

        return data_out
    except ValueError:
        print('no outliers were found')
        return df.copy()


var_2 = ['Age']
var_3 = ['PurchaseAmount']

data_ver_3 = drop_var_outlier(data_vers_2, var_2)
data_ver_3 = drop_var_outlier(data_ver_3, var_3)


# data_ver_3.info()
def valid_email_2(a):
    """
    this function it for valid email with characters valid
    a = data brought from the store, preprocessed
    """
    regex_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

    rows_with_invalid_emails = []

    # Iterate through the rows of the DataFrame
    for index, row in a.iterrows():
        email = row['Email']
        if not re.fullmatch(regex_pattern, email):
            rows_with_invalid_emails.append(index)

    a = a.drop(rows_with_invalid_emails)
    return a


data_ver_4 = valid_email_2(data_ver_3)


# print(data_ver_4)
# data_ver_4.info()


def data_clean_put(colm, table, a, db_con):
    """
    here input the data for MYSQL in the secondary tables
    colm = columns put in the table.
    table = name of the table where it will be placed
    a = data brought from the store, preprocessed
    db_con = route the MYSQL
    """
    a = a.loc[:, colm]

    a.to_sql(name=table, con=db_con, index=False, if_exists='append')


var1 = ['PurchaseAmount']
table_1 = 'purchase_amount'
# data_clean_put(var1, table_1, data_ver_4, db_conn)

var2 = ['Age']
table_2 = 'age'
# data_clean_put(var2, table_2, data_ver_4, db_conn)

var3 = ['LastName', 'FirstName']
table_3 = 'name_lastname'


# data_clean_put(var3, table_3, data_ver_4, db_conn)

def table_complete(table, a, db_con):
    """
    table = name of the table where it will be placed
    a = data brought from the store, preprocessed
    db_con = route the MYSQL
    """
    encoding = 'latin-1'
    route_name = ("C:/Users/alejo/Desktop/pythonProject7/name_lastname.csv")
    data_name = pd.read_csv(route_name, delimiter=';', encoding=encoding)
    route_age = ("C:/Users/alejo/Desktop/pythonProject7/age.csv")
    data_age = pd.read_csv(route_age, delimiter=';', encoding=encoding)
    route_gender = ("C:/Users/alejo/Desktop/pythonProject7/gender.csv")
    data_gender = pd.read_csv(route_gender, delimiter=';', encoding=encoding)
    route_purchaseamount = ("C:/Users/alejo/Desktop/pythonProject7/purchaseamount.csv")
    data_purchaseamount = pd.read_csv(route_purchaseamount, delimiter=';', encoding=encoding)
    data_marge_name = pd.merge(a, data_name, on=('FirstName', 'LastName'), how='inner')
    data_marge_name = data_marge_name.rename(columns={'id': 'name_id'})
    data_marge_age = pd.merge(data_marge_name, data_age, on='Age', how='inner')
    data_marge_age = data_marge_age.rename(columns={'id': 'age_id'})
    data_marge_gender = pd.merge(data_marge_age, data_gender, on='Gender', how='inner')
    data_marge_gender = data_marge_gender.rename(columns={'id': 'gender_id'})
    data_marge_purchaseamount = pd.merge(data_marge_gender, data_purchaseamount, on='PurchaseAmount', how='inner')
    data_marge_purchaseamount = data_marge_purchaseamount.rename(columns={'id': 'purchaseamount_id'})
    data_finalized = data_marge_purchaseamount.loc[:, ['CustomerID', 'purchaseamount_id', 'Email', 'name_id',
                                                       'gender_id', 'PhoneNumber', 'age_id', 'Address']]

    data_finalized['PhoneNumber'] = data_finalized['PhoneNumber'].str.replace(' ', '')
    data_finalized['PhoneNumber'] = data_finalized['PhoneNumber'].str.replace('-', ' ')

    data_finalized = data_finalized.drop_duplicates('CustomerID')

    # data_finalized.to_sql(name=table, con=db_con, index=False, if_exists='append')
    return data_finalized


table_4 = 'main_table'
table_5 = table_complete(table_4, data_ver_4, db_conn)
print(table_5)
table_5.info()
