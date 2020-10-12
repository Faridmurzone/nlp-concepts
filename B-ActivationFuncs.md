# Funciones de activación para redes neuronales
Imaginémonos que nuestra red se encuentra en un camino frente a una bifurcación. ¿Cómo decide por qué lado ir? Esa es la tarea de las funciones de activación.

La función de activación recibe un valor de entrada y se encarga de devolver una salida que corresponde a la decisión o el peso determinado a partir de la entrada. Habitualmente los valores de salida están normalizados en rangos entre [0,1] o [-1,1].

Se trata de funciones matemáticas simples, característica que les permite no ser pesadas computacionalmente, lo que es sumamente importante teniendo en consideración que las funciones de activación se ejecutan para cada neurona de la red, por cada muestra y por cada ciclo. Por ejemplo si tenemos 1500 muestras con 10 features cada un con 4 capas de 64 neuronas cada una esto se traduce en millones de ejecuciones en un entrenamiento (3840000 ejecuciones para el ejemplo dado por cada iteración).

Veamos alguna de las funciones de activación más comunes:

## Sigmoid – Sigmoide
La función sigmoide transforma los valores introducidos a una escala (0,1), donde los valores altos tienen de manera asintótica a 1 y los valores muy bajos tienden de manera asintótica a 0.

[Dibujo]

Características de la función signoide:

Satura y mata el gradiente.
Lenta convergencia.
No esta centrada en el cero.
Esta acotada entre 0 y 1.
Buen rendimiento en la última capa.
 

## Tanh – Tangent Hyperbolic – Tangente hiperbólica
La función tangente hiperbólica transforma los valores introducidos a una escala (-1,1), donde los valores altos tienen de manera asintótica a 1 y los valores muy bajos tienden de manera asintótica a -1.
[Dibujo]

Características de la función tangente hiperbólica:

Muy similar a la signoide
Satura y mata el gradiente.
Lenta convergencia.
Centrada en 0.
Esta acotada entre -1 y 1.
Se utiliza para decidir entre uno opción y la contraria.
Buen desempeño en redes recurrentes.
 

## ReLU – Rectified Lineal Unit
La función ReLU transforma los valores introducidos anulando los valores negativos y dejando los positivos tal y como entran.

[Dibujo] 

Características de la función ReLU:

Activación Sparse – solo se activa si son positivos.
No está acotada.
Se pueden morir demasiadas neuronas.
Se comporta bien con imágenes.
Buen desempeño en redes convolucionales.
 

## Leaky ReLU – Rectified Lineal Unit
La función Leaky ReLU transforma los valores introducidos multiplicando los negativos por un coeficiente rectificativo y dejando los positivos según entran.

[Dibujo]

Características de la función Leaky ReLU:

Similar a la función ReLU.
Penaliza los negativos mediante un coeficiente rectificador.
No está acotada.
Se comporta bien con imágenes.
Buen desempeño en redes convolucionales.
 

## Softmax – Rectified Lineal Unit
La función Softmax transforma las salidas a una representación en forma de probabilidades, de tal manera que el sumatorio de todas las probabilidades de las salidas de 1.

[Dibujo]

Características de la función Softmax:

Se utiliza cuando queremos tener una representación en forma de probabilidades.
Esta acotada entre 0 y 1.
Muy diferenciable.
Se utiliza para para normalizar tipo multiclase.
Buen rendimiento en las últimas capas.