# Import principal libraries
import pandas as pd
#import numpy as np
import streamlit as st
#import altair as alt
from recommender import recommend, movies_df2, genres_index_df, movies_df3

st.title("Films Recommender")

#Select a gender
selected_gender = st.selectbox("Select gender", genres_index_df.columns)

# Title names selector
title_names_indexes = []
title_names_indexes = genres_index_df[selected_gender].dropna().astype(int)

# Filtrar los índices que están fuera del rango de movies_df2
title_names_indexes = title_names_indexes[title_names_indexes < len(movies_df2)]
title_names = movies_df2['title'].iloc[title_names_indexes].values


#Film Selector
selected_movie = st.selectbox("Select a film", title_names)

#Recommendations
recommendations = recommend(selected_movie)
recommendations_df = pd.DataFrame({
    'Title': movies_df3['title'].iloc[recommendations],
    'Director': movies_df3['crew'].iloc[recommendations],
    'Cast': movies_df3['cast'].iloc[recommendations],
    'Overview': movies_df3['overview'].iloc[recommendations]
})
recommendations_df = recommendations_df.reset_index(drop=True)
recommendations_df.index += 1

if st.button("Recommend"):
   
    st.write(f"Recomendations for {selected_movie}:")
    st.dataframe(recommendations_df)