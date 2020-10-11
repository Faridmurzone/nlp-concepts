
# Extracción de Features con Bag of Words (BOW) para NLP

En este artículo veremos como transformar tokens en features, con ello podremos contar cuantas veces o en qué contexto aparecen determinadas palabras y comenzar a darle a nuestra computadora cosas que puede entender.

## ¿Qué es bag of words?

Supongamos que queremos detectar determinadas palabras en un texto como **buenísima** u **malísima** para realizar un análisis de sentimientos de comentarios de una película en base a la aparición de algunas de estas palabras. Donde **buenísima** nos indicaría que es un comentario positivo y viceversa.

La manera de lograr que la computadora pueda interpretar esas palabras se llama *vectorización del texto* y consiste en convertir las oraciones en un vector con la siguiente forma:


Si observamos se construyó una matriz con cada una de las palabras que aparecen en alguno de los cometarios. Donde para cada fila se completa con un 0 o un 1 según la aparición (1) o ausencia (0) de esa palabra en la frase. A estor vectores de 0's y 1's se los denomina *sparse vectors*.

Como vimos en la imagen a simple vista, por la aparición de la palabra **malísima** hay dos comentarios negativos y por **buenisima** hay uno positivo. Lo he simplificado al extremo para introducir los conceptos, esto se torna un poco más intrincado si tenemos, en lugar de comentarios de 4 o 5 palabras, un millón de reviews de 1000 palabras ¿Ya empiezan a imaginarse esos vectores? Los animo a pensar en el siguiente problema: ¿Y qué pasa con el orden de las palabras?. Exacto, ya no existe orden, sólo aparición o ausencia de cada una de ellas en un comentario. Es por esto que a esta técnica se la denomina *bag of words* o *bolsa de palabras* en español. 


## Cómo resolver el problema de alteración de orden de las palabras

Un primer enfoque para resolver el problema del orden es trabajar con *n-gramas* que nos permiten saber que tan probable es una sentencia conformada por **n  palabras** o qué tan común es en determinados corpus de texto. Los n-gramas pueden estar conformados por una palabra (unigramas), dos palabras (bigramas), tres palabras (trigramas) o cualquier número n de palabras (n-gramas) donde n hace referencia a la cantidad de palabras.

Veamos algunos ejemplos para la frase "Hola, como estás hoy":
```
hola, como, estas, hoy -> unigramas
como estas -> bigrama
como estas hoy -> trigrama
```

El trigrama "como estas hoy" nos aporta una información mucho más completa que "como estás", y a la vez el bigrama nos aporta información más consisa que las palabras sueltas. El problema de esto ya empieza a saltar a la vista. Al conformar un vector de bigramas tenemos todas las posibles combinaciones de cada palabra con la palabra anterior y con la palabra posterior. A medida que aumentamos n la cantidad de combinaciones crece. Pongamos esto ahora en el contexto de un millón de comentarios de 200 palabras. Es demasiado, no escala.

## Cómo resolver el problema de combinaciones que crecen exponencialmente en n-gramas

Seguimos afinando nuestro problema y ahora debido a la capacidad finita y limitada de procesamiento (por lo menos por ahora) tenemos que acotar el problema. Un buen **aproach** para disminuir la cantidad de combinaciones es tomar solamente los determinados n-gramas. ¿Y con cuáles nos quedamos? Uno tendería a decir: con los más frecuentes. Pero no, las palabras más frecuentes se encuentran en muchos, a veces en todos los comentarios. Una palabra que aparece siempre no nos aporta realmente demasiada información para el problema que estamos intentando resolver. Palabras como esta, el, la, y, o, a, en, etc (preposiciones, conjunciones, etc) están siempre y no son precisamente las que le otorgan el sentido a la oración. A estas palabras se las denomina *stop words* y en la mayoría de los preprocesados para extracción de features se opta por eliminarlas.

En el otro extremo tenemos palabras realmente raras que aparecen por ejemplo 1 sola vez en la totalidad de los cometarios. Estas palabras tampoco aportan (en general demasiada información), en la mayoría de los casos corresponden a errores de tipeo o vienen de otro idioma y también es bueno prescindir de ellas para evitar overfiting (sobre ajuste de nuestro modelo). Por ejemplo si hay sólo un comentario que contiene el bigrama "sombriro rojo" y es un comentario positivo, es muy probable que si en un nuevo comentario aparece el mismo bigrama (sombrero mal escrito + la palabra rojo) incline el peso del modelo a pensar que es un comentario positivo, ya que es una palabra que nunca apareció y cuya aparición se dió sólo en el contexto de un comentario positivo.

¿Entonces con qué nos quedamos? Con la amplia avenida del medio, excluyendo las palabras muy frecuentes (stop words) y a las muy poco frecuentes.

## TF-IDF

Ya logramos acotar bastante la cantidad de n-gramas. Sin embargo análisis muy grandes como el que mencionamos (un millón de comentarios de 200 palabras cada uno) nos vamos a encontrar que sigue siendo demasiado. Para esto se han ideado algunos algoritmos como el que mencionaremos en este apartado: *TF-IDF*.

TF proviene del inglés **Term Frecuency**, frecuencia del término. Existen diversas maneras de ver si un término existe en un documento: 
- Mediante una clasificación binaria, el término se encuentra o no se encuentra (0 o 1).
- Mediante un conteo exaustivo de apariciones en el documento.
- Term Frecuency: Contar las apariciones del término en el documento y dividirlo sobre la totalidad de términos en el documento.
- Normalización logaritmica: Tomando el logaritmo del recuento.

IDF proviene del inglés **Inverse Document Frecuency**, frecuencia de documento inversa. Vamos a nomenclar con la letra N a la totalidad de comentarios en nuestro corpus y al corpus con la letra D, que es el conjunto de todos nuestros comentarios. Ahora veamos cuantos comentarios hay en el corpus que tengan un término específico (por ejemplo "buenísimo"), su representación matemática sería así: *|{d ∈ D: t ∈ d}|* donde d es cada documento en el corpus y t el término en el documento.

Con esta simbología ya tenemos la interpretación de IDF:


Que es el logaritmo de N (totalidad de comentarios) sobre comentarios que tengan el término específico en el corpus.


Con todo lo expuesto ya podemos calcular el TF-IDF como el producto de TF y IDF:

Para este cálculo hemos requerido, el término a examinar, el documento y el corpus completo de documentos (en nuestro ejemplo comentarios). Una vez procesado este cálculo obtenemos un peso que es la frecuencia en determinado documento. Queríamos encontrar qué términos eran más o menos frecuentes que otros, así que vamos a la parte divertida y hagamoslo en python. Para ello utilizaremos la librería scikit-learn que hará toda la matemática por nosotros:

```python

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
comentarios = ["esta pelicula es malisima", "esta pelicula no es malisima", "esta pelicula es buenisima", "malisima", "me gustó", "no me gustó", "no creo que sea una buena película", "es buenisima"]

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))

features = tfidf.fit_transform(comentarios)

df = pd.DataFrame(
    features.todense(),
    columns=tfidf.get_feature_names()
)
print(df)

# Output:
   buenisima        es  es buenisima  ...        no  pelicula  pelicula es
0   0.000000  0.321127      0.000000  ...  0.000000  0.366257     0.424441
1   0.000000  0.328779      0.000000  ...  0.374985  0.374985     0.000000
2   0.415002  0.313986      0.415002  ...  0.000000  0.358112     0.415002
3   0.000000  0.000000      0.000000  ...  0.000000  0.000000     0.000000
4   0.000000  0.000000      0.000000  ...  0.000000  0.000000     0.000000
5   0.000000  0.000000      0.000000  ...  0.445928  0.000000     0.000000
6   0.000000  0.000000      0.000000  ...  1.000000  0.000000     0.000000
7   0.623489  0.471725      0.623489  ...  0.000000  0.000000     0.000000

[8 rows x 13 columns]
```

Para explicar brevemente lo que hicimos:
- En las lineas 1 y 2 importamos TfidVectorizer y pandas
- En la linea 3 armamos una lista de comentarios ficticios para analizar.
- En la linea 4 definimos los parámetros para el TfidVectorizer, min_df y max_df definen un humbral mínimo y máximo de frecuencia (threshold). En síntesis lo que hace es eliminar los n-gramas que tienen una frecuencia demasiado baja y ngram_range define que tipo de ngramas vamos a analizar (en nuestro caso puse unigramas y bigramas) así scikit sabe qué tipo de n-gramas tomar para la vectorización.
- Luego el método fit_transform() ejecuta la instrucción
- La última linea sólo arma la tabla en pandas para visualizarla. 



El output mostrado está resumido, ya que para las 6 frases se encontraron un total de 13 columnas. Tengamos también en cuenta que algunos n-gramas fueron filtrados por el threshold de min_df y max_df. También vemos que la librería normalizó los valores por nosotros en un rango entre 0 y 1. Esto facilita a la computadora poder analizar los pesos en forma relativa a los demás. Los invito a explorar los pesos, ya tenemos nuestro TF-IDF. Si bien el ejemplo es un dataset demasiado chico ya podemos empezar a reconocer cómo para cada frase algunos n-gramas empiezan a ser más determinantes que otros. A medida que crece en volumen nuestro dataset será más y más obvio. Ya estamos listos para entrenar un modelo que clasifique automáticamente.