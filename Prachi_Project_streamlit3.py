import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import movieposters as mp
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

## importing files from local

# ratings=pd.read_csv(r'ml-latest-small/ratings.csv')
# movies=pd.read_csv(r'ml-latest-small/movies.csv')



## importing files in github

ratings=pd.read_csv(r'./ratings.csv')
movies=pd.read_csv(r'./movies.csv')

# popularity

# function definition
def pop_rec(genre, year, n_output):

    scaler = MinMaxScaler()

    rating_df = pd.DataFrame(ratings.groupby('movieId')['rating'].mean()) # group movies and get their avarage rating
    rating_df['rating_count'] = ratings.groupby('movieId')['rating'].count() # get rating count of each movie(how many times each movie was rated)
    scaled_df = pd.DataFrame(scaler.fit_transform(rating_df), index=rating_df.index, columns=rating_df.columns) # scale the ratings and rating counts
      #scaled_df
    scaled_df["hybrid"] = scaled_df['rating'] + scaled_df['rating_count'] # add up rating and rating count for each mivie
    sort_rate = pd.DataFrame(scaled_df["hybrid"].sort_values(ascending=False))
    pattern = '\((\d{4})\)'
    recommend1 = sort_rate.merge(rating_df.merge(movies, how='left', left_index=True, right_on="movieId"), how='left', left_index=True, right_index=True)
    recommend1['year'] = movies.title.str.extract(pattern, expand=False)
    recommend2 = recommend1.dropna()
    recommender3 = recommend2.loc[recommend2.genres.str.contains(genre)] 
    recommender3['year'] = recommender3['year'].astype(int)
    recommender4 = recommender3.loc[(recommender3["year"] >= year-5) & (recommender3["year"] <= year+5)]
    return pd.DataFrame(recommender4['title']).head(n_output)


### Genres list

genres = movies['genres'].str.split("|", expand=True)
genre_list = pd.unique(genres[[0, 1, 2, 3, 4, 5, 6]].values.ravel('K'))
fin_genre_list = np.delete(genre_list, np.where(genre_list == '(no genres listed)') | (genre_list == None))




#genrebox
genre_inp = st.selectbox(
    'What genre would you like to watch?',
    (fin_genre_list))

st.write('Genre:', genre_inp)


#yearbox

year_inp = st.slider('Give a year range', 1990, 2020, 2020)
st.write(year_inp)

#number of recommended movies
num_inp = st.slider('Give a number of recommendations(1-20)', 1, 20, 1)
st.write(num_inp)

#st.dataframe(pop_rec(genre_inp, year_inp, num_inp))
 
# importing images ###############################
    
#posters
from PIL import Image

posters_list = []
titles_list = []
for i in pop_rec(genre_inp, year_inp, num_inp).title:
    link = mp.get_poster(title= i)
    #title_postlink = i, link
    posters_list.append(link)
    titles_list.append(i)
#posters_list 

columns = st.columns(len(posters_list))
for a, i, j in zip (columns, posters_list, titles_list):
    with a:
        st.image(i, j, 100)
        


        
        
        
        
        
        
        
    
# Making Recommendations Correlation - Item Based

# defining the pivot matrix
movies_crosstab = pd.pivot_table(data=ratings, values='rating', index='userId', columns='movieId')

# defining the rating
rating = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())
rating['rating_count'] = ratings.groupby('movieId')['rating'].count()

# function definition
def top_n_movie(movie_id,n):
    movie_ratings = movies_crosstab[movie_id]
    similar_to_movie = movies_crosstab.corrwith(movie_ratings)
    corr_movie = pd.DataFrame(similar_to_movie, columns=['PearsonR'])
    corr_movie.dropna(inplace=True)
    movie_corr_summary = corr_movie.join(rating['rating_count'])
    movie_corr_summary.drop(movie_id, inplace=True) # drop the inputed restaurant itself
    top10 = movie_corr_summary[movie_corr_summary['rating_count']>=10].sort_values('PearsonR', ascending=False).head(5)
    top10 = top10.merge(movies, left_index=True, right_on="movieId")
    return top10

# # with Alis help --> mind taking the indexies instead of movie IDs


m_var = movies.sample(axis=0).index[0]
m_title = movies.loc[m_var, "title"]
m_Id = movies.loc[m_var, "movieId"]


###  
# items = pickle.load(open('./Item_movies.sav', 'rb'))

with st.container():
    #st.subheader("Item Based Movie Recommendation System")
    st.subheader(f"If you like :blue[_{m_title}_], You might also like..")
    movie_names= top_n_movie(m_Id, 5)
    # st.write(movie_names)
    

## Movie posters/images for item based

#posters
from PIL import Image

posters_list = []
titles_list = []
for i in top_n_movie(m_Id, 5).title:
    link = mp.get_poster(title= i)
    #title_postlink = i, link
    posters_list.append(link)
    titles_list.append(i)
#posters_list 

columns = st.columns(len(posters_list))
for a, i, j in zip (columns, posters_list, titles_list):
    with a:
        st.image(i, j, 100)
    
#Making Recommendations Correlation - User Based

users_items = pd.pivot_table(data=ratings, values='rating', index='userId', columns='movieId')
users_items.fillna(0, inplace=True)

user_similarities = pd.DataFrame(cosine_similarity(users_items),
                                 columns=users_items.index, 
                                 index=users_items.index)

def weighted_user_rec(user_id, n):
  weights =(user_similarities.query("userId!=@user_id")[user_id] / sum(user_similarities.query("userId!=@user_id")[user_id]) )
  not_seen_movies = users_items.loc[users_items.index!=user_id, users_items.loc[user_id,:]==0]
  weighted_averages = pd.DataFrame(not_seen_movies.T.dot(weights), columns=["predicted_rating"])
  recommendations = weighted_averages.merge(movies, left_index=True, right_on="movieId")
  result=recommendations.sort_values("predicted_rating", ascending=False).head(n)
  return result

with st.form(key='my_form'):
    st.subheader("User Based Movie Recommendation System")
    user_id = st.number_input(label="Hi! I'm your personal recommender. Tell me your userID",min_value=1,step=1)
    submit_button = st.form_submit_button(label='Show')
    if user_id in ratings.userId.values:
        result = weighted_user_rec(user_id, 5)
        st.write("You will probably like these movies:") 
        final =list(result['title'])[0:5]
        # st.write(final)
    else:
        st.write("Sorry, but you have entered wrong UserID. Please enter correct UserID")
        # pass

## Movie posters/images for user based

#posters
from PIL import Image

posters_list = []
titles_list = []
for i in final:
    link = mp.get_poster(title= i)
    #title_postlink = i, link
    posters_list.append(link)
    titles_list.append(i)
#posters_list 

columns = st.columns(len(posters_list))
for a, i, j in zip (columns, posters_list, titles_list):
    with a:
        st.image(i, j, 100)
