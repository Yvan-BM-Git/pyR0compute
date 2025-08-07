# pyR0Compute

Este repositorio contiene una herramienta simbólica en Python para calcular el número básico de reproducción $R_0$ mediante el método de próxima generación, aplicable a modelos epidemiológicos definidos por sistemas de ecuaciones diferenciales.

## Instrucciones de uso

1. **Descarga y abre el archivo** [`pyR0compute_examples.ipynb`](https://github.com/Yvan-BM-Git/pyR0compute/blob/main/pyR0compute_examples.ipynb)  
   Puedes usar [Google Colab](https://colab.research.google.com/) o tu entorno local de Jupyter.

2. **Ejecuta la primera celda**, que carga las clases necesarias del archivo `pyR0compute_examples.ipynb`.

3. **Explora los ejemplos incluidos**: el notebook contiene modelos como SIR, SEIR y vectoriales, ya preparados para cálculo de $R_0$.

4. **Para usar tu propio modelo**, define:
   - Las **variables del sistema**.
   - Los **parámetros** del modelo.
   - El **sistema de ecuaciones diferenciales**.
   - Asegúrate de **ordenar primero las ecuaciones de los compartimentos infectados**, seguidas por las restantes (ver nota IMPORTANTE abajo).

5. **Ejecuta el método `calculate_R0()`** para obtener el valor simbólico de $R_0$.

---

## IMPORTANTE:

El sistema de ecuaciones debe ser ingresado en el siguiente orden:

- **Primero**: ecuaciones de los compartimentos **infectados** (ej., $I, V $, etc.).
- **Después**: ecuaciones del resto de los compartimentos (ej., $S, R$, etc.).

Este orden es clave para que el algoritmo construya correctamente las matrices $\mathbf{F}$ y $\mathbf{V}$.
