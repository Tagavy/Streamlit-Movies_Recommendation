#pip install streamlit --upgrade
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

##Streamlit
#title
st.title(":red[*Movie Recommender*] :movie_camera:")

#Recommenders

url1 = "https://drive.google.com/file/d/1sc_yJw6Ej7hmS36OIQHl1UX6h6Jq32uN/view?usp=share_link"
path1 = 'https://drive.google.com/uc?export=download&id='+url1.split('/')[-2]
movies = pd.read_csv(path1)

url2 = "https://drive.google.com/file/d/1shB74shA6w-rOcHHANDiXkiq4-RO-uyP/view?usp=share_link"
path2 = 'https://drive.google.com/uc?export=download&id='+url2.split('/')[-2]
ratings = pd.read_csv(path2)

url3 = "https://drive.google.com/file/d/1Kwuudazbm-jcRIxGO2g3HipoFFRlU17x/view?usp=share_link"
path3 = 'https://drive.google.com/uc?export=download&id='+url3.split('/')[-2]
links = pd.read_csv(path3)

url4 = "https://drive.google.com/file/d/1hgwga5-UMVYEM3qpQT_3vlLerJ3_F2B9/view?usp=share_link"
path4 = 'https://drive.google.com/uc?export=download&id='+url4.split('/')[-2]
tags = pd.read_csv(path4)

###Genres list

genres = movies['genres'].str.split("|", expand=True)
genre_list = pd.unique(genres[[0, 1, 2, 3, 4, 5, 6]].values.ravel('K'))
fin_genre_list = np.delete(genre_list, np.where(genre_list == '(no genres listed)') | (genre_list == None))


##Popularity Recommender

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



#genrebox
genre_inp = st.selectbox(
    'What genre would you like to watch?',
    (fin_genre_list))

st.write('Genre:', genre_inp)


#yearbox

year_inp = st.slider('Give a year range', 1990, 2020, 2020)
st.write(year_inp)

#number of recommended movies
#num_inp = st.slider('Give a number of recommendations(1-20)', 1, 20, 1)
#st.write(num_inp)

st.dataframe(pop_rec(genre_inp, year_inp, 10))
 
#posters
#from PIL import Image
#image1 = Image.open('https://m.media-amazon.com/images/M/MV5BNGY3NWYwNzctNWU5Yi00ZjljLTgyNDgtZjNhZjRlNjc0ZTU1XkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_QL75_UX380_CR0,0,380,562_.jpg')
#image2 = Image.open('https://m.media-amazon.com/images/M/MV5BMjAyN2EwNzAtZjQ4NS00NTQ3LWE1MzctYmIwOTA4Nzc2ZGZhXkEyXkFqcGdeQXVyNjQ2MjQ5NzM@._V1_QL75_UY562_CR1,0,380,562_.jpg')
#st.image(image1,  caption='Billy Elliot (2000)') #', 'Last Dance (1996)'))



