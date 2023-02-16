from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import data_prep_using_function as prep
import main
import pandas as pd
import random

def making_list(df1, df2, some_user_id):
    user_red_book = df1[df1['User-ID'] == str(some_user_id)]
    user_red_book_df = df2[df2['ISBN'].isin(user_red_book['ISBN'])]
    user_red_book_df_ISBN_list = list(user_red_book_df['ISBN'])
    
    return user_red_book_df_ISBN_list

# 전체 유사도 계산하려고 하니까 로컬에서도 터지네... 흐음... 50000개만 한다.
def use_random(df2):
    random_ISBN = random.sample(list(df2['ISBN']), 50000)
    random_title = df2[df2['ISBN'].isin(random_ISBN)]['Book-Title']
    return random_ISBN, random_title

def making_matrix(random_title):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(random_title)
    cos_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return cos_sim_matrix

def contents_based_recommend(user_red_book_df_ISBN_list, random_ISBN, cos_sim_matrix, df2):
    cos_sim_df = pd.DataFrame(cos_sim_matrix, index = random_ISBN, columns = random_ISBN)
    contents_title_df = pd.DataFrame(cos_sim_df[~cos_sim_df.columns.isin(user_red_book_df_ISBN_list)].iloc[:,0])
    contents_title_df = contents_title_df.sort_values(contents_title_df.columns[0], ascending=False)
    contents_title_df_top2 = contents_title_df[1:3]
    contents_title_df_top2 = contents_title_df_top2.reset_index()
    top2_books_content_based = df2[df2['ISBN'].isin(contents_title_df_top2['index'])][['ISBN', 'Book-Title']]

    return top2_books_content_based

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
    made_list = making_list(df1, df2, some_user_id)
    random_ISBN, random_title = use_random(df2)
    cos_matrix = making_matrix(random_title)
    cont_based_recommend = contents_based_recommend(made_list, random_ISBN, cos_matrix, df2)
    print(cont_based_recommend)
