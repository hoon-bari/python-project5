import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
import data_prep_using_function as prep

def df_to_sparse_matrix(df1, num_users, num_ISBN):
    return coo_matrix((df1['Book-Rating'].to_list(),
                                (df1['user_idx'].to_list(),
                                    df1['ISBN_idx'].to_list())
                                ), 
                                shape = (num_users, num_ISBN),
                                dtype = np.float32,
                                )

def train_model(sp_data, k = 40, metric = 'cosine'):
    knn_model = NearestNeighbors(n_neighbors = k, metric = metric)
    knn_model.fit(sp_data)
    return knn_model

def get_single_sp_mat(df1, num_ISBN, some_user_id):
    single_user_df = df1[df1['User-ID'] == str(some_user_id)]
    user_list = np.zeros((1, num_ISBN))

    for ISBN_idx, rating in zip(single_user_df['ISBN_idx'].to_list(), 
                                single_user_df['Book-Rating'].to_list()):
        user_list[0, ISBN_idx] = rating

    return user_list

def run_model(knn_model, sp_user, k = 40):
    distances, indices = knn_model.kneighbors(sp_user, n_neighbors = k)

    return distances, indices

def KNN_recommend(df1, df2, indices, some_user_id, top_k = 5):
    freq_score_df = df1[df1['user_idx'].isin(indices[0][1:])].groupby('ISBN').agg('count')
    freq_score_df = freq_score_df.sort_values('Book-Rating',ascending = False)
    top_freq_books_list = freq_score_df.index.tolist()

    red_books = df1[df1['User-ID'] == some_user_id]['ISBN'].tolist()

    book_result = []
    for bid in top_freq_books_list:
        if bid not in red_books:
            book_result.append(bid)
            if len(book_result) >= top_k:
                break

    return df2[df2['ISBN'].isin(book_result)][['ISBN', 'Book-Title']]

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
    knn_model = train_model(rating_sparse_arr)

    sp_user = get_single_sp_mat(df1, some_user_id)
    _, indices = run_model(knn_model, sp_user)
    top_books = KNN_recommend(df1, df2, indices, some_user_id, top_k = 5)
    print(top_books)