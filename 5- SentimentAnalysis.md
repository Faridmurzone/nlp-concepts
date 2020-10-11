# Clasificador de texto con regresiones logísticas (NLP)

En este artículo veremos como clasificar un texto utilizando un modelo de **Logistic Regression** con el objetivo de hacer un análisis de sentimientos de comentarios.

Si aún no lo hiciste, te recomiendo visitar los artículos de [tokenización](https://medium.com/escueladeinteligenciaartificial/procesamiento-de-texto-para-nlp-1-tokenizaci%C3%B3n-4d533f3f6c9b), [preprocesado](https://medium.com/escueladeinteligenciaartificial/procesamiento-de-lenguaje-natural-stemming-y-lemmas-f5efd90dca8) y [feature extraction con TF-IDF](https://medium.com/escueladeinteligenciaartificial/extracci%C3%B3n-de-features-con-bag-of-words-bow-y-tf-idf-para-nlp-f89d678abc0e) que te brindarán los fundamentos previos al análisis de texto. No son conocimientos excluyentes pero te darán muchas herramientas para hacer el end-to-end.

## ¿Qué es una regresión logística?

Una regresión logística es un algoritmo que intenta predecir el valor positivo o negativo dadas las features (propiedades) que le dame de un modelo determinado. Para esto habitualmente se utiliza una función sigmoide, si estás sólido en matemáticas no te preocupes... lo veremos. 
La función sigmoide es una función matemática que graficada se ve como la siguiente curva:

[Imagen]

Si observamos es una curva que recorre valores de Y (eje vertical) entre 0 y 1. Esto nos permite hacer una clasificación binaria sencilla donde mientras más cercano a 1 es positivo y mientras más cercano a 0 es negativo. Esta curva tiene otra particularidad que la hace muy apta para la clasificación: a medida que nos movemos en X (eje horizontal) la variación es más drástica hacia 1 o 0 en Y. Es decir, tiende a convertirse en 0 o 1 de una manera muy brusca. Y eso está bueno, ya que queremos que tienda a los extremos sin perder los matices de que tan 1 o que tan 0 es.

Como es una función lineal puede manejar los datos vectorizados de un texto (nuestro famoso *sparse vector* de valores entre 0 y 1 de aparición de palabras), es muy rápido de entrenar y fácil de interpretar dado que nos brinda pesos (*weights*) que nos dicen que tan influyente es una palabra tendiendo a 0 o 1.

## 

Veamos cómo hacer esto con un ejemplo real.

...


Esto es realmente impresionante. Con un modelo realmente sencillo ya le hemos enseñado a la computadora a comprender palabras positivas y negativas. Y apenas estamos empezando.

