import pandas as pd
import numpy as np
import data_prep_using_function as prep
import user_based_recommend as ub
import contents_based_recommend as cb 
import KNN_recommend as knn
import SVD_recommend as svd

if __name__ == '__main__':

    some_user_id = input('유저 ID를 입력하세요 : ')
    try:
        # 전처리 
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
        file_info2 = [
                    {
                        'f_name' : '/Users/seunghoonchoi/Downloads/SKKU KDT 2기/프로젝트/5차/BX-CSV-Dump/BX-Users.csv',
                        'sep' : ';',
                        'col_names' : ['User-ID', 'Location', 'Age']
                    }
                ]      
        df1, df2 = prep.preprocessing_pipeline(file_info)
        df3, df4, df5 = prep.preprocessing_pipeline2(file_info2)

        # 유저 베이스
        recom1 = ub.recommend1_state(df1, df2, df3, some_user_id)
        recom2 = ub.recommend2_age(df1, df2, df3, df4, df5, some_user_id)
        user_recommend = ub.user_based_recommend(recom1, recom2)

        # 컨텐츠 베이스
        made_list = cb.making_list(df1, df2, some_user_id)
        random_ISBN, random_title = cb.use_random(df2)
        cos_matrix = cb.making_matrix(random_title)
        cont_based_recommend = cb.contents_based_recommend(made_list, random_ISBN, cos_matrix, df2)

        # KNN
        num_users, num_ISBN = prep.calc_data_num(df1)
        rating_sparse_arr1 = knn.df_to_sparse_matrix(df1, num_users=num_users, num_ISBN=num_ISBN)
        knn_model = knn.train_model(rating_sparse_arr1)

        sp_user = knn.get_single_sp_mat(df1, num_ISBN, some_user_id)
        _, indices = knn.run_model(knn_model, sp_user)
        knn_top_books = knn.KNN_recommend(df1, df2, indices, some_user_id, top_k = 5)

        # SVD
        rating_sparse_arr2 = svd.df_to_sparse_matrix(df1, num_users=num_users, num_ISBN=num_ISBN)
        svd_model = svd.train_model(rating_sparse_arr2)
        pred = svd.run_model(df1, svd_model, some_user_id)
        svd_top_books = svd.SVD_recommend(df1, df2, pred, some_user_id, top_k = 5)
        
        # 결과
        total_df = pd.concat([user_recommend, cont_based_recommend, knn_top_books, svd_top_books], axis=0).drop_duplicates()[:10]

        print(f'{some_user_id}번 유저님께서 관심가질만한 10권의 책을 추천합니다.')
        print(total_df)
    except:
        print(f'{some_user_id}번 유저님께서 읽은 책이 없어 추천을 할 수 없습니다.')