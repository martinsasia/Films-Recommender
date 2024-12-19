# Import principal libraries
import pandas as pd
#import numpy as np
import streamlit as st
#import altair as alt
from recommender import recommend, movies_df2

st.title("Recomendador de Películas")

# Selector de película
selected_movie = st.selectbox("Selecciona una película:", movies_df2['title'].values)
recommendations = recommend(selected_movie)

if st.button("Recomendar"):
   
    st.write(f"Recomendaciones para {selected_movie}:")
    st.write(recommendations)