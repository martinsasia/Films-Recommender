# Import principal libraries
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

# Load data frames
credits_df = pd.read_csv("https://raw.githubusercontent.com/martinsasia/Films-Recommender/refs/heads/main/Datasets/credits.csv")
movies_df = pd.read_csv("https://raw.githubusercontent.com/martinsasia/Films-Recommender/refs/heads/main/Datasets/movies.csv")
movies_df_raw = movies_df.copy()
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',10)
# @title Preparación del Dataset
movies_df = movies_df_raw

# Merge the movies and credits dataset to work with all the data together
movies_df = pd.merge(movies_df, credits_df, how='inner', left_on='id', right_on= 'movie_id')

# Select the desired columns
movies_df = movies_df[['id', 'title_x', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies_df.dropna(inplace= True)

# Function for extracting only the name field of the jason
def extract_names(genres_string):
    # Convert strings to list of dictionaries
    genres_list = ast.literal_eval(genres_string)
    # Extract the field "name" from each dictionary
    return [genre['name'] for genre in genres_list]

# Apply the function to 'genres'
movies_df['genres'] = movies_df['genres'].apply(extract_names)
movies_df['keywords'] = movies_df['keywords'].apply(extract_names)
movies_df = movies_df.rename(columns = {'title_x':'title'})

# Funtion for extracting the field 'name' of the first tree dictionaries
def extract_first_three_names(cast_string):
    try:
        # Convertir la cadena JSON a una lista de diccionarios
        cast_list = ast.literal_eval(cast_string)
        # Extraer los nombres de los primeros 3 elementos, si existen
        return [member['name'] for member in cast_list[:3] if 'name' in member]
    except (ValueError, SyntaxError):
        # Retornar una lista vacía si ocurre algún error en la conversión
        return []

# Aplicar la función a la columna 'cast'
movies_df['cast'] = movies_df['cast'].apply(extract_first_three_names)

# Function for extracting the name of the director
def extract_director_name(crew_string):
    try:
        # Convert the string into a list of dictionaries
        crew_list = ast.literal_eval(crew_string)
        # Search the dictionary where 'job' is 'Director' and return 'name'
        for crew_member in crew_list:
            if crew_member.get('job') == 'Director':
                return crew_member.get('name')
    except (ValueError, SyntaxError):
        pass  # Manage errors for invalid data
    return None  # If the director isn't found

# Apply the funcion to the column 'crew'
movies_df['crew'] = movies_df['crew'].apply(extract_director_name)
# Convert elements into a list of the element
movies_df['crew'] = movies_df['crew'].apply(lambda x: [x])


# Split the overview field. After could be replace by a tokenization
movies_df['overview'] = movies_df['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Replace spaces with no space in some strings columns
movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(" ", "") for i in x if isinstance(i,str)] if isinstance(x, list) else [])
movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(" ", "") for i in x if isinstance(i,str)] if isinstance(x, list) else [])

# Make a tags column
movies_df['tags'] = movies_df[['overview', 'genres', 'keywords', 'cast', 'crew']].sum(axis= 'columns')

# New dataframe with the only 3 columns of intest
movies_df2 = movies_df[['id', 'title', 'tags']]

# Recover the joined text
movies_df2['tags'] = movies_df2['tags'].apply(lambda x : " ".join(x))

# Turn tag texts into lower case
movies_df2['tags'] = movies_df2['tags'].apply(lambda x : x.lower())


# @title Vectorize tag text, find word stem and meassure similarities
# Vectorize text: It transform tags column in an array that contains the values of a table with all the significative words with each frecuencies 
vectorizer = CountVectorizer(max_features= 5000, stop_words= 'english') 
vectors = vectorizer.fit_transform(movies_df2['tags']).toarray()

# Find the word stem
ps = PorterStemmer()

# Function to do it
def stemmer(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# Apply the function stemmer to the column 'tags'
movies_df2['tags'] = movies_df2['tags'].apply(stemmer)

# Meassure all the similarities of the vectors between them. It appears a diagonal of ones.
similarity = cosine_similarity(vectors)
# Function of recommending
def recommend(movie):
    movie_index = movies_df2[movies_df2['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse= True, key = lambda x : x[1])[1:6]

    recommended_titles = []
    for i in movies_list:
        recommended_titles.append(movies_df2.iloc[i[0]].title)

    return recommended_titles