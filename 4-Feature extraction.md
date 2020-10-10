
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

IDF proviene del inglés **Inverse Document Frecuency**, frecuencia de documento inversa. Vamos a nomenclar con la letra N a la totalidad de documentos en nuestro corpus y al corpus con la letra D.