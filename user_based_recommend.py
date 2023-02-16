import pandas as pd
import data_prep_using_function as prep

def recommend1_state(df1, df2, df3, some_user_id):
    some_user_location = df3[df3['User-ID'] == str(some_user_id)]['Location'].values
    some_user_state = some_user_location[0][1]
    same_state_user_index = []

    for row in df3.itertuples():
        if some_user_state in row.Location:
            same_state_user_index.append(row.Index)

    same_state_user_df = df3[df3.index.isin(same_state_user_index)]
    same_state_user_book_rating_df = df1[df1['User-ID'].isin(same_state_user_df['User-ID'])]
    same_state_user_book_rating_df1 = same_state_user_book_rating_df.groupby('ISBN')[['Book-Rating']].mean()
    same_state_user_book_rating_df2 = same_state_user_book_rating_df.groupby('ISBN')[['Book-Rating']].count()
    same_state_user_book_rating_df2.rename(columns={'Book-Rating':'count'}, inplace=True)
    same_state_user_book_rating_df_merge = pd.concat([same_state_user_book_rating_df1, same_state_user_book_rating_df2], axis=1)
    same_state_user_book_rating_df_merge = same_state_user_book_rating_df_merge.sort_values(['Book-Rating',	'count'], ascending=False)
    same_state_user_book_rating_df_merge = same_state_user_book_rating_df_merge.reset_index()
    same_state_user_book = df2[df2['ISBN'] == same_state_user_book_rating_df_merge.loc[0]['ISBN']][['ISBN', 'Book-Title']]

    return same_state_user_book

def recommend2_age(df1, df2, df3, df4, df5, some_user_id):
    some_user_age = df3[df3['User-ID'] == str(some_user_id)]['Age'].values
    if pd.isnull(some_user_age):
        same_age_user_df = df4
    else:
        same_age_user_index = []
        for row in df5.itertuples():
            if ((int(some_user_age)-1) - (int(some_user_age)-1)%10)  <= int(row.Age) < ((int(some_user_age)+ 9)//10 * 10):
                same_age_user_index.append(row.Index)
        same_age_user_df = df5[df5.index.isin(same_age_user_index)]

    same_age_user_book_rating_df = df1[df1['User-ID'].isin(same_age_user_df['User-ID'])]
    same_age_user_book_rating_df1 = same_age_user_book_rating_df.groupby('ISBN')[['Book-Rating']].mean()
    same_age_user_book_rating_df2 = same_age_user_book_rating_df.groupby('ISBN')[['Book-Rating']].count()
    same_age_user_book_rating_df2.rename(columns={'Book-Rating':'count'}, inplace=True)
    same_age_user_book_rating_df_merge = pd.concat([same_age_user_book_rating_df1, same_age_user_book_rating_df2], axis=1)
    same_age_user_book_rating_df_merge = same_age_user_book_rating_df_merge.sort_values(['Book-Rating',	'count'], ascending=False)
    same_age_user_book_rating_df_merge = same_age_user_book_rating_df_merge.reset_index()
    same_age_user_book = df2[df2['ISBN'] == same_age_user_book_rating_df_merge.loc[0]['ISBN']][['ISBN', 'Book-Title']]

    return same_age_user_book

def user_based_recommend(same_state_user_book, same_age_user_book):
    user_based_recommend_book = pd.concat([same_state_user_book, same_age_user_book], axis=0)

    return user_based_recommend_book


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
    file_info2 = [
                {
                    'f_name' : '/Users/seunghoonchoi/Downloads/SKKU KDT 2기/프로젝트/5차/BX-CSV-Dump/BX-Users.csv',
                    'sep' : ';',
                    'col_names' : ['User-ID', 'Location', 'Age']
                }
            ]
    df1, df2 = prep.preprocessing_pipeline(file_info)
    df3, df4, df5 = prep.preprocessing_pipeline2(file_info2)
    recom1 = recommend1_state(df1, df2, df3, some_user_id)
    recom2 = recommend2_age(df1, df2, df3, df4, df5, some_user_id)
    user_recommend = user_based_recommend(recom1, recom2)
    print(user_recommend)