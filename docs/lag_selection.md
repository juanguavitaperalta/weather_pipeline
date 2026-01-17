Variable target vs variables independientes para forecasting de temperatura.
La correlación dirigida permite evaluar que tanto ayuda la variable independiente analizada k pasos hacia atras para predecir y k pasos hacia el futuro. El objetivo es identificar cuantos instantes temporales de las variables independientes contienen información util para estimar k instantes de tiempo hacia el futuro.

$$
\text{temperature}_{2\text{m}}(t+h)
$$

Marco conseptual:
Para cada variable exogena x, se evalua la correlación:

$$
\rho(k) = \text{corr}\left(x(t-k),\, y(t+h)\right)
$$

donde y(t+h) es la variable objetivo desplazada al horizonte de prediccón.
x(t-k) es la variable independiente observada k pasos en el pasado.
k debe ser mnayor a cero pues no es conceptualmente correcto analizar información futura de las variables objetivo.

Reglas:
1. No seleccionar lags aislados. Ojo, en un fenomeno fisico debemos evaluar la inercia intrinseca de los fenomenos. Como en la teoria del control  todo tiene una inercia intrinseca, por lo que debemos tener en cuenta este tema dentro de la selección de lags.

2. Coherencia de Blques. Lags con mismo signo, magnitud similar reflejar fenomeno fisico o estacionario. Sin embargo, no utilizamos todos los lags del bloque. Suele seleccionarse uno tardio, uno temprano o el de mayopr magnitud absoluta. Reucira colinealidad y complejidad inncesaria.

Wind speed lags seleccionados: 1, 2, 6, 12, 24.
relative humidity: 1, 6, 12, 24

