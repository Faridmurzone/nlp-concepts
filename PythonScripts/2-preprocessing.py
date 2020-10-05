#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:04:01 2020

@author: fmurzone
"""
from nltk.tokenize import WhitespaceTokenizer 
tk = WhitespaceTokenizer() 

texto = "¿Cuanto tiempo pasó desde que comí una manzana?"
texto_tokenizado = tk.tokenize(texto)
print(texto_tokenizado) 

from nltk.tokenize import WordPunctTokenizer 
from nltk.tokenize import TreebankWordTokenizer 
texto_tokenizado1 = WordPunctTokenizer().tokenize(texto)
texto_tokenizado2 = TreebankWordTokenizer().tokenize(texto)

print(texto_tokenizado1)
print(texto_tokenizado2)

texto_pablito = "Pablito clavó un clavito cuantos clavitos clava pablito"
pablito_tokenizado = WordPunctTokenizer().tokenize(texto_pablito)

print(pablito_tokenizado)

from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')
stemmed_text = [stemmer.stem(i) for i in pablito_tokenizado]

print(stemmed_text)

from nltk.stem import WordNetLemmatizer
nltk.download('omw')
nltk.download('wordnet')
wnl = WordNetLemmatizer()
lemmatized_text = [wnl.lemmatize(i) for i in pablito_tokenizado]
print(lemmatized_text)

import stanza
stanza.download("es")
nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')
texto_pablito = "Pablito clavó un clavito cuantos clavitos clava pablito"
doc = nlp(texto_pablito)
print(*[f'Palabra: {word.text+" "}\tLemma: {word.lemma}' for sent in doc.sentences for word in sent.words], sep='\n')

