import numpy as np
from scipy.sparse import coo_matrix
from scipy import stats
from scipy.sparse import linalg
import data_prep_using_function as prep

def df_to_sparse_matrix(df1, num_users, num_ISBN):
    return coo_matrix((df1['Book-Rating'].to_list(),
                                (df1['user_idx'].to_list(),
                                    df1['ISBN_idx'].to_list())
                                ), 
                                shape = (num_users, num_ISBN),
                                dtype = np.float32,
                                )

def train_model(sp_data, latent_factor = 100):
    U, S, V = linalg.svds(sp_data, k=latent_factor)
    model = {'U':U, 'S':S, 'V':V}
    return model

def run_model(df1, model, some_user_id):
    some_user_idx = df1[df1['User-ID'] == str(some_user_id)]['user_idx'].iloc[0]
    pred = model['U'][some_user_idx] @ np.diag(model['S']) @ model['V']
    return pred

def SVD_recommend(df1, df2, pred, some_user_id, top_k=5):
    sorted_index = np.argsort(pred)[::-1]

    re_book_df = df1[['ISBN_idx', 'ISBN']]
    re_book_df = re_book_df.drop_duplicates()

    red_books = df1[df1['User-ID'] == some_user_id]['ISBN'].tolist()

    result = []
    for i in sorted_index:
        bid = re_book_df[re_book_df['ISBN_idx'] == i]['ISBN'].values[0]
        if bid not in red_books:
            result.append(bid)
            if len(result) >= top_k:
                break   
    
    return df2[df2['ISBN'].isin(result)][['ISBN', 'Book-Title']]

if __name__ == '__main__':

    some_user_id = main.some_user_id
    file_info = [
                {
                    'f_name' : '/Users/seunghoonchoi/Downloads/SKKU KDT 2기/프로젝트/5차/BX-CSV-Dump/BX-Book-Ratings.csv',
                    'sep' : ';',
                    'col_names' : ['User-ID', 'ISBN', 'Book-Rating']
                },
                {
                    'f_name' : '/Users/seunghoonchoi/Downloads/SKKU KDT 2기/프로젝트/5차/BX-CSV-Dump/BX-Books.csv',
                    'sep' : ';',
                    'col_names' : ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
                },
            ]
    df1, df2 = prep.preprocessing_pipeline(file_info)
    num_users, num_ISBN = prep.calc_data_num(df1)
    rating_sparse_arr = df_to_sparse_matrix(df1, num_users=num_users, num_ISBN=num_ISBN)
    svd_model = train_model(rating_sparse_arr)
    pred = run_model(svd_model, some_user_id)
    top_books = SVD_recommend(df1, df2, pred, top_k = 5)
    print(top_books)