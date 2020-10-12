# ¿Dónde están los datos?

En general como analistas de datos o programadores de inteligencia artificial lo primero que queremos es desarrollar nuestro modelo. Sin embargo en mi experiencia he visto que uno de los funnels más grandes de los científicos de datos está en buscar y ordenar la información. 
En general los cursos de data science comienzan en la etapa de preprocesado de los datos, su limpieza, tokenización, etc. ¿Pero de dónde sacamos esos datos sucios? En varias ocasiones me he encontrado con científicos de datos que ante la propuesta de un problema y su solución me contestaron: ¿Y de dónde saco el dataset? Cuando esto debería ser la tarea más sencilla dada la abundancia de datos en la actualidad.
En este artículo expondré algunas ideas de cómo abordar ese problema.

## Datos en todas partes.

En el mundo actual datos es lo que sobran, en eso estamos todos de acuerdo. Pero en general estamos tan abrumados por la abundancia de datos que no sabemos organizar nuestra mirada para tomar una parte de ellos, organizarlos y construir nuestro dataset. 

Me gusta pensar en Neo de Matrix cuando logra ver lo que hay detrás de lo que la matrix nos oculta. Es exactamente así. Por un lado es una práctica cotidiana de la mayoría de las personas compartir datos de manera pública: comentarios en redes sociales, foros y comunidades, videos, albumes de fotos. Por otro hay espacios colaborativos como Wikipedia y/o grandes repositorios de datos en los que la humanidad se ha dedicado a cargar su manera de interpretar e interactuar con el mundo.

Algunas de las fuentes más habituales son:

Existe una enorme cantidad de APIs y datasets públicos que nos permiten tener un primer acercamiento para entrenar nuestros modelos. Algunas de las empresas más grandes (y por ende con más datos) del mundo comparten habitualmente datasets. Por ejemplo (clic en los links para ver los datasets):
- [Amazon](http://jmcauley.ucsd.edu/data/amazon/): Empresas como Amazon y Google suelen compartir datasets publicos.
- [Google Dataset](https://datasetsearch.research.google.com/): es un potente buscador de datasets.
- [Stanford](https://nlp.stanford.edu/sentiment/code.html): la universidad de Stanford tiene un departamento muy importante de NLP. Una libreria muy famosa (stanza) y multilenguaje está desarrollada por ellos.
- [John Hopkings](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/): La universidad John Hopkings también está a la vanguardia en NLP y suele compartir datasets y papers SOTA.
- [IMDB](https://www.kaggle.com/iarunava/imdb-movie-reviews-dataset) la famosa página de películas tiene un dataset enorme que se suele usar para sentiment analysis.
- [Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle](https://www.kaggle.com/): La meca de los data scientists. Allí suelen compartirse no sólo miles de datasets, también maneras de analizarlos, desafíos y cuenta con una gran comunidad.
- Datos de instituciones públicas: Acá hay demasiado y de muchas fuentes, lo dejo para que googlees... la mayoría de los gobiernos tienen repositorios de datos a partir de sus instituciones de estadísticas. Por poner el ejemplo Argentino: https://datos.gob.ar/ o el de México: https://datos.gob.mx/

## Consiguiendo / armando un dataset que no está armado

De todas formas hay muchas veces que no tenemos un dataset armado para nuestras necesidades. Entonces entra nuestra creatividad. Hay dos enfoques principales cuando no tenemos un dataset que nos brinda de una la información que necesitamos:

### Utilizar un dataset o modelo y adaptarlo
Supongamos que queremos hacer un análisis de sentimientos de comentarios de nuestra página de películas. Pero todavía tenemos muy pocos comentarios y performa muy mal. Bueno, una opción sería entrenarlo con otro dataset y despues testearlo contra el nuestro. Por ejemplo si quisieramos ver los comentarios positivos y negativos podríamos entrenar el modelo de IMDB, que además de los comentarios nos brinda las estrellitas que le pusieron los usuarios a una película, por ende tenemos ya una clasificación de comentarios positivos y negativos vinculando el comentario con la calificación otorgada. Una vez que tengamos ese modelo entrenado podemos utilizarlo con nuestros comentarios.

Un enfoque similar es el de **fine tuning** y **transfer learning**, un proceso mediante el cual podemos utilizar un modelo entrenado con un objetivo pero que se ajusta muy bien a otro. Este enfoque se utilizó mucho en los últimos dos años sobre todo para imagenes, donde podemos tener un modelo de clasificación de imágenes entrenado con miles o millones de fotos de objetos o animales (perros, gatos, caballos) entre los que tal vez no se encuentra la clasificación que nosotros queremos hacer (supongamos unicornios rojos y unicornios verdes). En ese caso podemos tomar el modelo ya entrenado y transferir sus pesos al nuestro, reentrenar el nuestro con pocos datos y obtener un resultado muy satisfactorio. En general (y muy resumido) el proceso de *transfer learning* consiste en tomar el modelo ya reentrenado y agregar una capa de entrenamiento con los features de nuestro dataset (unicornios rojos vs verdes) con lo cual tomará de todas las capas previas otras características como formas, bordes, texturas y tomará de la nuestra el color o el cuerno del unicornio para diferenciar la forma del que ya reconocía antes (caballo).

### Data scraping
El término de data scraping suele asustar. Y en realidad es mucho más sencillo de lo que parece. Existe una variopinta cantidad de librerías destinadas a obtener datos de fuentes online. Entre las fuentes tenemos algunas ya mencionadas como comunidades, foros, redes sociales. Y otras que pueden provenir de datos que nosotros tengamos en emails, en la computadora o nuestras bases de datos o bien que queramos extraer de una web ajena.

Tenés que analizar precios productos en el mercado? Se puede hacer scraping de productos de retailers. Necesitás comentarios de alguna noticia en particular? Scraping de páginas de noticias (en general suele ser más fácil porque ofrecen versiones en XML y/u otros feeds)

Entre las librerías más comunes para scraping que podemos encontrar estan:
- [Requests](https://requests.readthedocs.io/es/latest/)
- [Beautifoul Soup](https://pypi.org/project/beautifulsoup4/)
- [lxml](https://lxml.de/)
- [selenium](https://selenium-python.readthedocs.io/)
- [scrapy](https://scrapy.org/)

Entre otras a las que hay que sumarle API's de empresas, cuya diferencia con las mencionadas al principio del artículo es que no nos proveen datasets ya armados pero contienen abundante información a veces accidentalmente tagueada por los propios usuarios. El ejemplo más claro es [Twitter](https://developer.twitter.com/) de donde podemos obtener twits relativos a un tema sólo filtrando por un hashtag. Eso nos permite construir muy fácilmente un dataset. También es muy sencillo armarse un dataset de imágenes o videos recurriendo a páginas como google y scrapeando fotos del resultado, de manera que estamos usando todo el potencial de la IA de google para armar un dataset ad hoc a nuestras necesidades.


## Creatividad
Ante todo sé creativo. Los datos están por todas partes. No hay excusas para no desarrollar tu dataset y ponerte manos a la obra. A veces la tarea es más tediosa pero abunda la información, un caso que suelo poner de ejemplo es una clasificación de señales de audio que tuve que realizar basada en el género de la persona. En un primer aproach encontré datasets de audios ya clasificados en hombre/mujer pero en inglés. Como los datos no eran suficientes y las variaciones en lenguaje español eran sustanciales complementé el dataset con audios extraídos de una web de clases de español, en aquella ocasión como no tenía aún suficientes herramientas bajé más de 500 audios a mano, uno por uno, clasificandolos en dos carpetas según el género. Hoy los bajaría con alguna de las librerías mencionadas antes y en lugar de taguear uno por uno desarrollaría una clasificación por los nombres de los usuarios para determinar automáticamente cuáles son de un varón y cuál de una mujer. 

En fin, se trata de pensar el problema una y otra vez, buscar y bucear sobre el oceano de datos e ingeniarselas para ordenarlos. Una vez tenemos eso ya podemos divertirnos con nuestro modelo, en el tema que nosotros queramos y sin depender de que nadie nos arme un dataset.