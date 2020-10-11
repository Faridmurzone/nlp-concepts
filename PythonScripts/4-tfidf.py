# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
comentarios = ["esta es malisima", "esta movie no es malisima", "esta pelicula es buenisima", "malisima", "me gustó", "no me gustó", "no creo que sea una buena opcion", "es buenisima"]

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))

features = tfidf.fit_transform(comentarios)

df = pd.DataFrame(
    features.todense(),
    columns=tfidf.get_feature_names()
)
print(df)