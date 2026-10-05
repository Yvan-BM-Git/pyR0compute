# pyR0compute

Cálculo simbólico del número reproductivo básico $R_0$ en modelos compartimentales de EDO mediante el método de la matriz de próxima generación (van den Driessche y Watmough, 2002).

Escribe el modelo, indica qué compartimentos están infectados y pyR0compute hace el resto: todo símbolo que no sea una variable de estado se trata como parámetro, se identifican los términos de nuevas infecciones $\mathcal{F}$ y de transiciones $\mathcal{V}$, se resuelve el equilibrio libre de enfermedad (DFE, por sus siglas en inglés) y se obtiene $R_0 = \rho(FV^{-1})$ como expresión de SymPy.

[![PyPI](https://img.shields.io/pypi/v/pyR0compute)](https://pypi.org/project/pyR0compute/)
[![Python](https://img.shields.io/pypi/pyversions/pyR0compute)](https://pypi.org/project/pyR0compute/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yvan-BM-Git/pyR0compute/blob/main/examples/pyR0compute_ejemplos.ipynb)

## Instalación

```bash
pip install pyR0compute
```

En un notebook de Jupyter o Colab usa `%pip install pyR0compute`. La versión en desarrollo se puede instalar desde GitHub con `pip install git+https://github.com/Yvan-BM-Git/pyR0compute`.

Requiere Python ≥ 3.9, SymPy y NumPy.

## Inicio rápido

```python
from pyr0compute import R0Model

model = R0Model("""
    dS/dt = Lambda - beta*S*I - mu*S
    dE/dt = beta*S*I - (sigma + mu)*E
    dI/dt = sigma*E - (gamma + mu)*I
    dR/dt = gamma*I - mu*R
""", infected=["E", "I"])

model.parameters   # (Lambda, beta, gamma, mu, sigma)  <- detectados automáticamente
model.dfe          # {S: Lambda/mu, E: 0, I: 0, R: 0}
model.R0           # Lambda*beta*sigma/(mu*(gamma + mu)*(mu + sigma))
```

Las ecuaciones pueden escribirse en cualquier orden; no es necesario poner primero los compartimentos infectados.

### Tres formas de ingresar un modelo

```python
# 1. Texto: una línea "dX/dt = ..." (o "X' = ...") por variable.
#    Las líneas auxiliares como "N = S + I + R" se sustituyen.
R0Model("""
    N = S + I + R
    dS/dt = Lambda - beta*S*I/N - mu*S
    dI/dt = beta*S*I/N - (gamma + mu)*I
    dR/dt = gamma*I - mu*R
""", infected=["I"])

# 2. Un diccionario {variable: lado derecho} (texto o expresiones de SymPy)
R0Model({"S": "Lambda - beta*S*I - mu*S",
         "I": "beta*S*I - (gamma + mu)*I"}, infected=["I"])

# 3. Símbolos de SymPy
import sympy as sp
S, I = sp.symbols("S I")
beta, gamma, mu, Lam = sp.symbols("beta gamma mu Lambda")
R0Model({S: Lam - beta*S*I - mu*S, I: beta*S*I - (gamma + mu)*I}, infected=[I])
```

Usa subíndices en los nombres para obtener un LaTeX limpio: `mu_h`, `alpha_hv` y `Delta_h` se escriben $\mu_h$, $\alpha_{hv}$ y $\Delta_h$. Los nombres que son especiales en SymPy (`I`, `S`, `E`, `N`, `beta`, `gamma`, `Lambda`, incluso `lambda`) son símbolos comunes dentro de los modelos en texto, y `^` indica potencia.

### Qué se obtiene

| Atributo / método | Contenido |
|---|---|
| `R0` | Número reproductivo básico (radio espectral de $FV^{-1}$) |
| `parameters` | Parámetros detectados automáticamente |
| `new_infections`, `transitions` | Términos $\mathcal{F}_i$ y $\mathcal{V}_i$ de cada compartimento infectado |
| `dfe`, `dfe_candidates` | Equilibrio libre de enfermedad utilizado, y todos los equilibrios no negativos encontrados |
| `F`, `V`, `K` | Jacobianas en el DFE y matriz de próxima generación $K = FV^{-1}$ |
| `next_generation_matrix_small` | $K$ restringida a los compartimentos que reciben nuevas infecciones |
| `eigenvalues` | Valores propios de $K$ |
| `R0_numeric(values)` | Radio espectral numérico (para modelos grandes sin forma cerrada) |
| `sensitivity_indices(values=None)` | Índices de sensibilidad normalizados $\Upsilon_p = \frac{\partial R_0}{\partial p}\frac{p}{R_0}$ |
| `report()` | Descripción paso a paso de todo el cálculo |
| `report_latex(style, standalone, mat_str)` | El mismo informe en LaTeX (`"document"`, compilable con `standalone=True`) o en Markdown para notebooks (`"markdown"`) |
| `latex()` | Código LaTeX de $R_0$ |

### Cuándo las elecciones automáticas necesitan ayuda

* **Poblaciones cerradas** (sin nacimientos), por ejemplo el SIR clásico: el DFE no es único, así que debes indicarlo con `dfe={"S": "N"}`. El mensaje de error señala qué valor falta.
* **Varios DFE** (por ejemplo, poblaciones de vectores con crecimiento logístico): se usa el que tiene más compartimentos distintos de cero y se muestra una advertencia. Puedes elegir otro con `dfe={...}`; todos están en `model.dfe_candidates`.
* **Descomposiciones personalizadas**: la separación $\mathcal{F}$/$\mathcal{V}$ no es única. Puedes reemplazar la automática con `new_infections={"E": "beta*S*I/N"}`.

### Cómo se detectan las nuevas infecciones

En la ecuación de un compartimento infectado, un término positivo es una nueva infección cuando involucra un compartimento infectado y además (i) se pierde desde un compartimento no infectado (una transferencia como $S \to E$), o (ii) involucra un compartimento no infectado y no es una transferencia entre compartimentos infectados. La progresión ($E \to I$), la falla del tratamiento, la superinfección entre cepas, las muertes y las recuperaciones van a $\mathcal{V}$. `model.report()` muestra la clasificación de cada término y su justificación.

## Validación

El conjunto de pruebas (`pytest`) reproduce resultados conocidos: SIR (con y sin dinámica vital, con acción de masas y con incidencia dependiente de la frecuencia), SEIR, un modelo con vacunación, Ross-Macdonald, un modelo huésped-vector SEIR/SEI con transmisión humano-humano y vectores logísticos, el modelo con tratamiento de van den Driessche y Watmough (2002, §4.1), modelos intrahuésped de células blanco, células infectadas y virus, y un modelo de dos cepas con superinfección. Los resultados simbólicos también se contrastan con el radio espectral numérico.

## Interfaz anterior

El código escrito para el notebook original sigue funcionando:

```python
from pyr0compute import GeneralEpidemiologicalModel
model = GeneralEpidemiologicalModel(variables, parameters, equations, infected_indices)
model.calculate_R0()
```

## Ejemplos

El notebook `examples/pyR0compute_ejemplos.ipynb` contiene ejemplos listos para ejecutar en Google Colab: SEIR, SIR, Ross-Macdonald, un modelo huésped-vector, un modelo intrahuésped, el modelo con tratamiento de van den Driessche y Watmough, y dos cepas con superinfección.

## Cita

Si usas pyR0compute en tu investigación, cítalo (ver `CITATION.cff`) junto con:

* van den Driessche, P., & Watmough, J. (2002). Reproduction numbers and sub-threshold endemic equilibria for compartmental models of disease transmission. *Mathematical Biosciences*, 180(1-2), 29-48.
* Diekmann, O., Heesterbeek, J. A. P., & Roberts, M. G. (2010). The construction of next-generation matrices for compartmental epidemic models. *Journal of the Royal Society Interface*, 7(47), 873-885.

## Licencia

MIT, ver `LICENSE`.