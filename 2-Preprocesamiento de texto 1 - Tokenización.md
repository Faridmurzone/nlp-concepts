
# Procesamiento de texto 1: Tokenización
## ¿Qué es el texto? 
Podemos pensar al texto como una secuencia de caracteres, de palabras, de frases y entidades nombradas, de oraciones, párrafos, etc.
Vamos a empezar por las más básicas. Podemos pensar en un texto como una secuencia de palabras y a la palabra como una secuencia de letras que juntas le dan forma y nos remiten a algun concepto. Varias palabras, junto con algunos símbolos como signos de puntuación, interrogación o exclamación terminan por otorgarle el significado.
Supongamos que tenemos las siguientes secuencias:

- lalaajradeaatengguesosedyartásobremes
- tengosedylajarradeaguaestásobrelamesa
- no tengo la jarra y la sed de agua está sobre mesa
- no tengo sed y la jarra de agua está sobre la mesa
- no, tengo sed y la jarra de agua está sobre la mesa

Como vemos, la misma secuencia de caracteres o de palabras pueden ser totalmente diferentes, inclusive incomprensibles. Y a veces un sólo signo de puntuación puede darle a la oración el significado exactamente opuesto. Estas reglas que a veces hasta los propios humanos omitimos, o que muchas veces comprendemos gracias a contextos o a la forma de comunicarse de una persona. Cómo hacen las máquinas para entender?

Algunos idiomas tienen particularidades como palabras compuestas. Por ejemplo el alemán tiene palabras como rechtsschutzversicherungsgesellschaften que significa "compañia de seguros que provee protección legal". Otros idiomas como el japonés no utilizan espacios. Y sin embargo la mente humana se entrena para comprender cada uno de esos idiomas. ¿Cómo hacemos para entrenar a la computadora de la misma manera?

## Tokenización 
La clave es el procesamiento del lenguaje. En NLP el proceso de convertir nuestras secuencias de caracteres, palabras o párrafos en inputs para la computadora se llama **tokenización**. Se puede pensar al token como la unidad para procesamiento semántico.

Para tokenizar un texto existen varias herramientas. Nosotros utilizaremos NLTK, una librería de python muy popular y potente.
Comencemos tokenizando una frase por palabras, separando por espacios:

```python
from nltk.tokenize import WhitespaceTokenizer 
tk = WhitespaceTokenizer() 
texto = "¿Cuánto tiempo pasó desde que comí una manzana?"
texto_tokenizado = tk.tokenize(texto)
print(texto_tokenizado)
# Output: ['¿Cuánto', 'tiempo', 'pasó', 'desde', 'que', 'comí', 'una', 'manzana?']
```
La salida del texto nos da una lista de palabras. Sin embargo *manzana* y *cuánto* figuran con el signo de pregunta. Y si tuvieramos la palabra manzana mencionada otra vez saldría nuevamente como un token diferente. Lo mismo nos sucedería si aparece una coma o un punto ¿Cómo hacemos para evitarlo?

La opción más popular es utilizar *TreebankWordTokenizer* en lugar de WhitespaceTokenizer. Pero no es la única manera. También podemos preprocesar el texto quitando comas y signos de puntuación y separar por espacios, o bien utilizar opciones como *WordPunctTokenizer* que separa por palabras tomando como separadores todo lo que no sea un caracter alfabetico. En inglés donde se utilizan muchos apóstrofes esto suele ser un problema. En español no tanto, por el contrario al estar *TreebankWordTokenizer* preparado para el inglés puede tener algún efecto negativo, veamos:

```python
from nltk.tokenize import WordPunctTokenizer 
from nltk.tokenize import TreebankWordTokenizer 

texto = "¿Cuánto tiempo pasó desde que comí una manzana?"
texto_tokenizado1 = WordPunctTokenizer().tokenize(texto)
texto_tokenizado2 = TreebankWordTokenizer().tokenize(texto)

print(texto_tokenizado1)
# Output: ['¿', 'Cuanto', 'tiempo', 'pasó', 'desde', 'que', 'comí', 'una', 'manzana', '?']
print(texto_tokenizado2)
# Output: ['¿Cuanto', 'tiempo', 'pasó', 'desde', 'que', 'comí', 'una', 'manzana', '?']
```

Como vemos, a pesar de que la opción de *TreebankWordTokenizer* es la más popular en inglés el signo de apertura de pregunta *¿* fue un problema para ella. Mientras que la opción de *WordPunctTokenizer* no tuvo ningún problema (Los hubiera tenido si en español tuvieramos palabras con apóstrofes).

