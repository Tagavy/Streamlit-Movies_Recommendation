import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity
ratings=pd.read_csv('ratings.csv')
movies=pd.read_csv('movies.csv')
posters=pd.read_csv('posters.csv')

#Making Recommendations Based on Popularity

top_10 = pickle.load(open('./Popular_movies.sav', 'rb'))

st.title("WBSFLIX")
with st.container():
    st.subheader("Top 5 WBSFLIX Popular Movies")
    movie_names=list(top_10['title'][0:5])
    st.write(movie_names)
    
#Making Recommendations Correlation - User Based

items = pickle.load(open('./Item_movies.sav', 'rb'))


with st.container():
    #st.subheader("Item Based Movie Recommendation System")
    st.subheader("Because you loved :blue[_Pocahontas_], You might also like..")
    movie_names=list(items['title'][0:5])
    st.write(movie_names)
    
#function
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

#with st.form(key='my_form'):
    #st.subheader("Because you loved :blue[__Pocahontas__], You might also like..")
    #if movie_id in movies.movieId.values:
        #result = top_n_movie(movie_id,5)
        #final=list(items['title'])[0:5]
        #st.write(final)
    #else:
        #st.write("Hello")
    
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
        final=list(result['title'])[0:5]
        st.write(final)
    else:
        st.write("Sorry, but you have entered wrong UserID. Please enter correct UserID")
    
