import pandas as pd
import numpy as np
from icecream.icecream import ic

def get_dataframe(file_name : str, column_names : list, sep : str, encoding : str = 'latin_1', low_memory=False) -> pd.DataFrame:
    df = pd.read_csv(file_name, sep=sep, header=None, names = column_names, encoding = 'latin_1', low_memory=False)
    return df

def drop(df):
    df.drop(0, axis=0, inplace=True)
    return df

def merge_df(df1, df2):
    return pd.merge(df1, df2[['ISBN', 'Book-Title']], on = 'ISBN')

def preprocessing_df_1(df):
    df = df.astype({'Book-Rating':'int'})
    df = df.sort_values('Book-Rating', ascending = False)
    df = df.reset_index(drop=True)

    user_ids = df['User-ID'].unique().tolist()

    user2idx_dict = {x: i for i, x in enumerate(user_ids)}
    idx2user_dict= {i: x for i, x in enumerate(user_ids)}

    new_ISBN = df['ISBN'].unique().tolist()

    ISBN2idx_dict = {x: i for i, x in enumerate(new_ISBN)}
    idx2ISBN_dict= {i: x for i, x in enumerate(new_ISBN)}

    df['user_idx'] = df['User-ID'].map(user2idx_dict)
    df['ISBN_idx'] = df['ISBN'].map(ISBN2idx_dict)

    df = df.sort_values(['user_idx', 'ISBN_idx'])
    df = df.reset_index(drop = True)

    df = add_normal_data((df))

    return df

def preprocessing_df_2(df):
    df['Location'] = df['Location'].str.split(',')
    # 결측치만 있는 데이터와 나이 있는 데이터 나눠주기
    df = df[df['Age'].isnull()]

    return df

def preprocessing_df_3(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    df['Age'] = df['Age'].astype(int)

    return df

def add_normal_data(df):
    min_rating = df['Book-Rating'].min()
    max_rating = df['Book-Rating'].max()
    rating_range = max_rating - min_rating
    df['rating_minmax'] = df['Book-Rating'].apply(lambda x : (x - min_rating)/rating_range)
    y_mm = df['Book-Rating'].apply(lambda x : (x - min_rating)/rating_range)

    import scipy.stats as ss
    y_z = ss.zscore(df['Book-Rating'])
    df['rating_zscore'] = y_z

    return df

def calc_data_num(df):
    num_users = df['user_idx'].nunique()
    num_ISBN = df['ISBN_idx'].nunique()

    return num_users, num_ISBN

def preprocessing_pipeline(file_info):

    book_ratings_df = get_dataframe(file_info[0]['f_name'], file_info[0]['col_names'], sep = file_info[0]['sep'])
    books_df = get_dataframe(file_info[1]['f_name'], file_info[1]['col_names'], sep = file_info[1]['sep'])
    book_ratings_df = drop(book_ratings_df)
    books_df = drop(books_df)
    merged_df = merge_df(book_ratings_df, books_df)
    prep_df = preprocessing_df_1(merged_df)
    return prep_df, books_df

def preprocessing_pipeline2(file_info):
    users_df = get_dataframe(file_info[0]['f_name'], file_info[0]['col_names'], sep = file_info[0]['sep'])
    users_df = drop(users_df)
    users_df_na = preprocessing_df_2(users_df)
    users_df_not_na = preprocessing_df_3(users_df)
    return users_df, users_df_na, users_df_not_na

if __name__ == '__main__':
    rating_cols = ['User-ID', 'ISBN', 'Book-Rating']
    book_cols = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
    user_cols = ['User-ID', 'Location', 'Age']

    book_ratings_df = get_dataframe('/Users/seunghoonchoi/Downloads/SKKU KDT 2기/프로젝트/5차/BX-CSV-Dump/BX-Book-Ratings.csv', rating_cols, sep = ';')
    books_df = get_dataframe('/Users/seunghoonchoi/Downloads/SKKU KDT 2기/프로젝트/5차/BX-CSV-Dump/BX-Books.csv', book_cols, sep = ';')
    users_df = get_dataframe('/Users/seunghoonchoi/Downloads/SKKU KDT 2기/프로젝트/5차/BX-CSV-Dump/BX-Users.csv', user_cols, sep = ';')
    
    book_ratings_df = drop(book_ratings_df)
    books_df = drop(books_df)
    users_df = drop(users_df)

    merged_df = merge_df(book_ratings_df, books_df)
    prep_df = preprocessing_df_1(merged_df)
    num_users, num_movies = calc_data_num(prep_df)
    users_df_na = preprocessing_df_2(users_df)
    users_df_not_na = preprocessing_df_3(users_df)

    print(prep_df)
    print(users_df_na)
    print(users_df_not_na)
    ic(num_users, num_movies)