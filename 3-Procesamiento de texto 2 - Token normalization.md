
# Procesamiento de texto 2 - Token normalization

Ya vimos anteriormente que podemos *tokenizar* un texto para hacerlo más amigable al input de una máquina. 
Habitualmente nos encontramos con que una palabra puede tener el mismo significado pero se puede representar de varias maneras, en singular, plural, en diferentes tiempos verbales, etc.

Veamos un ejemplo:

```python
from nltk.tokenize import WordPunctTokenizer 

texto_pablito = "Pablito clavó un clavito cuantos clavitos clava pablito"
pablito_tokenizado = WordPunctTokenizer().tokenize(texto_pablito)

print(pablito_tokenizado)
# Output: ['Pablito', 'clavó', 'un', 'clavito', 'cuantos', 'clavitos', 'clava', 'pablito']
```
Podemos notar que algunas palabras tienen el mismo significado o muy parecido y nos gustaría tokenizarlas como la misma. Por ejemplo: clavó y clava, clavito y clavitos y Pablito y pablito. Si bien el sustantivo propio con minúscula corresponde a un error, podemos encontrarnos con cosas de este tipo en un corpus de texto real.

## Stemming y Lemmas

Una de las formas de normalizar nuestros tokens es mediante **stemming** y **lemmatization**.

El *stemming* consiste en quitar y reemplazar sufijos de la raíz de la palabra. La *lemmatización* es un poco más compleja e implica hacer un análisis del vocabulario y su morfología para retornar la forma básica de la palabra (sin conjugar, en singular, etc).

El *stemming* es una forma rápida pero un poco torpe de tomar las raíces. Para eso utilizaremos SnowballStemmer de NLTK. El algoritmo es bastante simple y se puede ver [en este link](http://snowball.tartarus.org/algorithms/spanish/stemmer.html). Cada idioma tiene sus reglas, en inglés NLTK utiliza el algoritmo de stemming de Porters, en Español toma algunas reglas que, en resumen, remueven diversos sufijos que se atribuyen a acciones (ar, er, ir, ía, en, es, etc), terminaciones plurales, generos y otros. Una vez pasada la palabra por el algoritmo de stemming devuelve un **stem**, que es la palabra sin la terminación o sufijos. Veamos (continuando el ejemplo anterior):


```python
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')
stemmed_text = [stemmer.stem(i) for i in pablito_tokenizado]

print(stemmed_text)
# Output: ['pablit', 'clav', 'un', 'clavit', 'cuant', 'clavit', 'clav', 'pablit']
```

Vemos que todava puede ser mejorable. A priori se me ocurre que, si bien en esta horación una vez toma el lugar de verbo y otra de sustantivo, clavit y clav hacen referencia a clavo. De la misma manera que pablit hace referencia a Pablo. Es decir, no nos extrajo diminutivos de las palabras.

La **lemmatization** es un poco más compleja y busca la mejor palabra de origen de las que tenemos. Veamos qué pasa si aplicamos el lemmatizer de NLTK:


```python
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
wnl = WordNetLemmatizer()
lemmatized_text = [wnl.lemmatize(i) for i in pablito_tokenizado]
print(lemmatized_text)
# Output: ['Pablito', 'clavó', 'un', 'clavito', 'cuantos', 'clavitos', 'clava', 'pablito']
```
En primer lugar aclaración sobre download, es necesario bajar el wordnet para nltk antes de usarlo. Pero de todas formas lo que vemos en el output es que... no hizo nada.

El problema es que al menos hasta la fecha NLTK no tiene incorporada lemmatización para español. Por suerte existen otras librerías que permiten lemmatizar y trabajar el texto como [spacy](https://spacy.io/) o [stanza](https://stanfordnlp.github.io/stanza/index.html). Utilizaré para este ejemplo *Stanza*, una librería del departamento de NLP de la universidad de Stanford.

```python
import stanza
stanza.download("es")
nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma')
texto_pablito = "Pablito clavó un clavito cuantos clavitos clava pablito"
doc = nlp(texto_pablito)
print(*[f'Palabra: {word.text+" "}\tLemma: {word.lemma}' for sent in doc.sentences for word in sent.words], sep='\n')
'''
Output:
Palabra: Pablito        Lemma: Pablito
Palabra: clavó          Lemma: clavar
Palabra: un             Lemma: uno
Palabra: clavito        Lemma: clavito
Palabra: cuantos        Lemma: cuanto
Palabra: clavitos       Lemma: clavito
Palabra: clava          Lemma: clavar
Palabra: pablito        Lemma: pablito
'''
```

En primer lugar además de importar stanza (previamente instalado) se descarga el modelo para el idioma Español, Ancora, esto puede demorar algunos minutos ya que pesa aproximadamente 600mb. Luego, a diferencia de los ejemplos anteriores, le pasamos el texto completo. Lo que nos devuelve no sólo los lemmas correspondientes a cada palabra sino algunas características más. Finalmente recorremos la lista para imprimir las palabras con el lemma obtenido.

Hemos obtenido un listado de tokens bastante bueno. En el que los significados de clavar a pesar de su conjugación se tokenizan como una sola palabra. Lo mismo para clavito y pablito. Como último detalle podriamos pasar todos los tokens a minúsculas. Para simplificar el proceso podemos modificar nuestro string al principio:

```python
texto_pablito = "Pablito clavó un clavito cuantos clavitos clava pablito".lower()
print(texto_pablito)
# Output: pablito clavó un clavito cuantos clavitos clava pablito
```

Es todo por ahora ¡Podemos continuar!
