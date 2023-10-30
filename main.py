import pandas as pd
import numpy as np
data = pd.read_json("C:/Users/alejo/Desktop/pythonProject7/data/dataset.json")
print(data)
def fate_ejem():
    df= np.array.random(10)
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
    #book_df.to_sql(name='books', con=db_conn, index=False, if_exists='append')

