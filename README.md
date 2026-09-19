# Práctica: Clasificación con Árbol de Decisión (Wine Dataset)

## 1. Descripción del Proyecto
En esta práctica se entrenó un clasificador basado en árboles de decisión utilizando el conjunto de datos **Wine Dataset** de `scikit-learn`. Este dataset contiene 178 muestras de vinos clasificados en 3 clases con 13 características químicas.

## 2. Resultados de Precisión
- **Modelo con `max_depth=2`:** Logró una precisión del **88.89%** en el conjunto de prueba.
- **Modelo con `max_depth=None`:** Logró una precisión del **94.44%** en el conjunto de prueba.

## 3. Análisis de `max_depth`
- **¿Qué sucedió al cambiar `max_depth`?:** Al limitar la profundidad a 2, el árbol genera solo unas cuantas reglas condicionales sencillas (basadas principalmente en características clave como *proline* y *flavanoids*), haciendo que el modelo sea muy fácil de interpretar visualmente.
- **Diferencia al no limitar la profundidad (`max_depth=None`):** Al no limitar la profundidad, el árbol sigue dividiendo los nodos hasta lograr la máxima pureza posible en los datos de entrenamiento. Esto genera un conjunto de reglas mucho más extenso y complejo. Aunque incrementó ligeramente la precisión de prueba en este ejemplo, existe el riesgo de sobreajuste (*overfitting*).

## 4. Opiniones sobre los Resultados y el Dataset
Considero que el **Wine Dataset** cumple adecuadamente con los requerimientos para aprender clasificación supervisada simbólica. Sus características químicas están bien delimitadas y permiten al algoritmo construir reglas lógicas comprensibles sobre qué variables definen cada tipo de vino.