# pyR0compute

Cálculo simbólico del número reproductivo básico $R_0$ en modelos compartimentales de EDO mediante el método de la matriz de próxima generación (van den Driessche y Watmough, 2002).

Escribe el modelo, indica qué compartimentos están infectados y pyR0compute hace el resto: todo símbolo que no sea una variable de estado se trata como parámetro, se identifican los términos de nuevas infecciones $\mathcal{F}$ y de transiciones $\mathcal{V}$, se resuelve el equilibrio libre de enfermedad (DFE, por sus siglas en inglés) y se obtiene $R_0 = \rho(FV^{-1})$ como expresión de SymPy.

[![PyPI](https://img.shields.io/pypi/v/pyR0compute)](https://pypi.org/project/pyR0compute/)
[![Python](https://img.shields.io/pypi/pyversions/pyR0compute)](https://pypi.org/project/pyR0compute/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Yvan-BM-Git/pyR0compute/blob/main/LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yvan-BM-Git/pyR0compute/blob/main/examples/01_ejemplos.ipynb)

## Instalación

```bash
pip install pyR0compute
```

En un notebook de Jupyter o Colab usa `%pip install pyR0compute`. La versión en desarrollo se puede instalar desde GitHub con `pip install git+https://github.com/Yvan-BM-Git/pyR0compute`.

Requiere Python ≥ 3.9. `pip install pyR0compute` instala también sus dependencias: SymPy, NumPy, SciPy (sensibilidad global, DFE numérico y simulaciones), matplotlib (gráficos) y pandas (tablas de resultados).

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
| `R0_compact`, `dfe_compact`, `dfe_definitions` | $R_0$ y el DFE escritos con los valores $X^*$ que no tienen una forma cerrada corta, y sus definiciones en el orden en que se resuelven (ver *Sistemas con DFE no lineal*) |
| `R0_numeric(values)`, `dfe_numeric(values)` | Radio espectral y DFE numéricos (para modelos grandes, o con un DFE sin forma cerrada) |
| `dfe_blocks`, `dfe_is_implicit` | Cómo se resolvió cada bloque del DFE (forma cerrada, sin forma cerrada o límite de tiempo `dfe_timeout`) y cuánto tardó |
| `simulate(t_span, values=None, initial=None, n_runs=1, ...)` | Simulación numérica de la EDO con parámetros fijos o aleatorios y gráficos editables (ver *Simulación*) |
| `check_assumptions(values=None, language="en")` | Verifica los supuestos (A1)-(A5) de van den Driessche y Watmough (2002) y entrega un informe en LaTeX (ver *Supuestos de van den Driessche y Watmough*) |
| `dfe_stability(values=None)` | Condición (A5) de van den Driessche y Watmough: estabilidad del DFE en ausencia de infección |
| `sensitivity_indices(values=None)` | Índices de sensibilidad normalizados $\Upsilon_p = \frac{\partial R_0}{\partial p}\frac{p}{R_0}$ (locales) |
| `prcc(distributions, n, ...)` | Sensibilidad global por muestreo de hipercubo latino y coeficientes de correlación parcial de rangos (LHS-PRCC) |
| `sobol_indices(distributions, n, ...)` | Índices de Sobol de primer orden $S_1$ y totales $S_T$, con intervalos de confianza bootstrap |
| `report()` | Descripción paso a paso de todo el cálculo |
| `report_latex(style, standalone, mat_str)` | El mismo informe en LaTeX (`"document"`, compilable con `standalone=True`) o en Markdown para notebooks (`"markdown"`) |
| `latex()` | Código LaTeX de $R_0$ |

### Sensibilidad global

El índice $\Upsilon_p$ es local: depende del punto en que se evalúa y no recoge la incertidumbre de los parámetros ni sus interacciones. `prcc` y `sobol_indices` recorren todo el espacio de parámetros. Como $R_0$ está en forma cerrada, se evalúa vectorizado con NumPy y $10^5$ evaluaciones toman segundos; si los valores propios no tienen forma cerrada, se usa el radio espectral numérico de $K$.

```python
dists = {
    "beta":  ("loguniform", 1e-3, 4e-3),   # log p uniforme
    "mu":    ("truncnormal", 0.02, 0.005, 0.005, 0.05),
    "gamma": (0.05, 0.2),                  # uniforme
}
res = model.prcc(dists, fixed={"Lambda": 10, "sigma": 0.2}, n=2000, seed=1)
res = model.prcc(baseline={"Lambda": 10, "beta": 0.002, "mu": 0.02, "sigma": 0.2, "gamma": 0.1},
                 spread=0.3, n=2000, seed=1)       # todos uniformes en ±30 %
sob = model.sobol_indices(dists, fixed={"Lambda": 10, "sigma": 0.2}, n=4096, seed=1)

res.to_dataframe()           # tabla ordenada por influencia (requiere pandas)
res.to_latex()               # tabla LaTeX ("markdown" para notebooks)
sob.plot()                   # S1 y ST con intervalos (requiere matplotlib)
```

* **LHS-PRCC** (Marino et al., 2008) mide la fuerza y el signo de una relación **monótona**. La librería verifica la monotonía con el signo de $\partial R_0/\partial p$ en cada muestra y advierte si no se cumple.
* **Sobol** (Saltelli et al., 2010) descompone la varianza de $R_0$; $S_T - S_1$ cuantifica las interacciones. Usa $n \cdot (k + 2)$ evaluaciones, con $n$ potencia de 2, y supone parámetros independientes.
* Con `log_output=True` se analiza $\log R_0$. Si $R_0 = c\prod p_i^{a_i}$ y los parámetros son log-uniformes e independientes, $S_{1,i} = S_{T,i} = a_i^2\,\mathrm{Var}(\log p_i) / \sum_j a_j^2\,\mathrm{Var}(\log p_j)$, lo que conecta el índice local $\Upsilon_{p_i} = a_i$ con el global. Este resultado se usa como prueba de validación.

Distribuciones: `(low, high)` o `("uniform", low, high)`, `("loguniform", low, high)`, `("normal", media, de)`, `("truncnormal", media, de, low, high)`, `("triangular", low, moda, high)` o cualquier distribución congelada de `scipy.stats`.

### Sistemas con DFE no lineal

En modelos intrahuésped con respuesta inmune, el equilibrio libre de infección puede ser un sistema no lineal acoplado (por ejemplo, linfocitos T helper $H$ y reguladores $R$ que se regulan mutuamente, y células $C$, $B$ activadas por ambos). Resolverlo de una vez con `sympy.solve` puede no terminar. La librería:

1. Con los compartimentos infectados en cero, separa las ecuaciones de los no infectados en las componentes fuertemente conexas de su grafo de dependencias y las resuelve en orden topológico, por ejemplo $[E] \to [H, R] \to [C] \to [B]$.
2. Los valores ya resueltos entran a los bloques siguientes como símbolos $X^*$, no como su expresión explícita. Los valores cortos y sin radicales (como $E^* = g_E/d_E$) se escriben completos.
3. Las Jacobianas $F$ y $V$ se evalúan en ese DFE compacto, de modo que $R_0$ se simplifica cuando todavía es pequeño. `model.R0_compact` lo entrega en esa forma y `model.R0` con los valores explícitos sustituidos (sin simplificar).
4. Un bloque sin forma cerrada (por ejemplo la raíz de una quíntica) queda implícito: `dfe_numeric` y `R0_numeric` lo resuelven numéricamente (requiere SciPy).
5. Cada bloque acoplado o de grado mayor que 2 se resuelve en un proceso aparte con un límite de tiempo, `dfe_timeout` (20 s por defecto). Si SymPy no termina a tiempo, el proceso se detiene, el bloque queda implícito y se muestra un aviso: el modelo se construye igual y $R_0$ se entrega en forma compacta y numérica. `model.dfe_blocks` informa cómo se resolvió cada bloque y cuánto tardó; `dfe_timeout=None` quita el límite.
6. La sensibilidad global evalúa $R_0$ en cadena ($X^*$ en orden y luego $R_0$), con derivadas por regla de la cadena, en lugar de la expresión explícita.

```python
model = R0Model('''
    dE/dt = g_E - (d_E + tau_E*V/(V + z_VE))*E
    dI/dt = tau_E*V*E/(V + z_VE) - (d_I + tau_I*C/(C + z_CI))*I
    dV/dt = nu*I - (d_V + tau_V*B/(B + z_BV))*V
    dH/dt = g_H - (d_H - tau_H*I/(I + z_IH) + rho_H*R/(R + z_RH))*H
    dC/dt = g_C - (d_C - tau_C*H/(H + z_HC) + rho_C*R/(R + z_RC))*C
    dB/dt = g_B - (d_B - tau_B*H/(H + z_HB) + rho_B*R/(R + z_RB))*B
    dR/dt = g_R - (d_R - tau_R*H/(H + z_HR) + rho_R)*R
''', infected=["I", "V"])

model.R0_compact       # g_E*nu*tau_E*(B_star + z_BV)*(C_star + z_CI)/(d_E*z_VE*(...)*(...))
model.dfe_definitions  # {H_star: ..., R_star: ..., C_star: ..., B_star: ...}
model.latex()          # usa X^{*} cuando el DFE tiene valores compactos
model.dfe_stability(values)   # {"eigenvalues": ..., "stable": True, "dfe": {...}}
```

Los símbolos $X^*$ se llaman `X_star` en texto (para que `S_star*beta` no se lea como potencia) y se escriben $X^{*}$ en LaTeX. El notebook `examples/03_sistemas_no_lineales.ipynb` desarrolla este modelo completo.

La condición (A5) de van den Driessche y Watmough exige que el DFE sea estable cuando no hay infección; sin ella $R_0$ no es un umbral. `dfe_stability()` la verifica en un punto (`values=...`) o en muestras aleatorias de parámetros, y entrega los valores propios simbólicos cuando el bloque no infectado es triangular.

### Supuestos de van den Driessche y Watmough

El Teorema 2 de van den Driessche y Watmough (2002) garantiza que $R_0 = \rho(FV^{-1})$ es un umbral (el DFE es localmente asintóticamente estable si $R_0 < 1$ e inestable si $R_0 > 1$) solo si se cumplen los supuestos (A1)-(A5): flujos no negativos, sin salidas de un compartimento vacío, sin nuevas infecciones en los no infectados, subespacio libre de infección invariante, y DFE estable en ausencia de nuevas infecciones. `check_assumptions()` decide cada uno como *cumple*, *no cumple* o *no se pudo decidir*, con su fundamento: por construcción, demostración simbólica para todo parámetro positivo, en los valores dados con `values=`, o muestras aleatorias (una violación en todas las muestras es *no cumple*; un supuesto que no se pudo demostrar es *no se pudo decidir*).

```python
informe = model.check_assumptions(language="es")   # "en" por defecto
informe.holds                     # True si se cumplen los cinco
informe.status                    # {"A1": "holds", ..., "A5": "undecided"}
informe["A5"].basis               # "symbolic", "values", "sampling" o "construction"
informe.to_latex(standalone=True) # informe LaTeX compilable; "markdown" para notebooks
model.check_assumptions(values={...})   # decide en un punto lo que no se pudo demostrar
```

El informe incluye la descomposición $\mathcal{F}_i$, $\mathcal{V}_i^+$, $\mathcal{V}_i^-$ utilizada, el detalle de cada supuesto con contraejemplos cuando no se cumple, las consecuencias del Lema 1 ($F \ge 0$, $V$ con patrón de signos Z y M-matriz no singular) y, si el DFE elegido no es estable, los otros DFE encontrados que sí lo son. En Jupyter, evaluar el informe lo muestra en Markdown. El notebook `examples/05_supuestos_vdw.ipynb` presenta un modelo que cumple los cinco supuestos y modelos que violan cada uno.

### Simulación

`simulate()` integra la EDO (SciPy) y `plot()` / `plot_phase()` grafican el resultado (matplotlib). Los parámetros dados en `values` quedan fijos y los demás se sortean (log-uniformes en `default_range`, o en `ranges={...}`). Por defecto los compartimentos no infectados parten del DFE y los infectados de una perturbación pequeña, que es la situación que describe $R_0$.

```python
res = model.simulate((0, 200), values={"Lambda": 10, "mu": 0.1, "gamma": 0.5, "beta": 0.01})
res.R0, res.final()                          # R0 de la simulación y estado final
res.plot(["S", "I"], title="SIR con R0 = {R0}", xlabel="días", labels={"I": "infectados"},
         colors={"I": "#e34948"}, linestyles={"I": "--"}, logy=True, figsize=(7, 3.5),
         show_dfe=True, save="outputs/sir.png")

ens = model.simulate((0, 200), values={"Lambda": 10, "mu": 0.1, "gamma": 0.5},
                     ranges={"beta": (0.001, 0.02)}, n_runs=40, seed=1)   # beta aleatorio
ens.plot(["S", "I"], subplots=True)          # cada simulación coloreada por R0 < 1 o R0 > 1
ens.plot_phase("S", "I")                      # plano de fase con el DFE
model.simulate(..., R0_range=(1, None))       # solo sorteos con R0 > 1
ens.summary(); ens.to_dataframe()             # parámetros y R0 de cada simulación; datos en formato largo
```

`plot()` acepta `variables`, `title` (`"{R0}"` se reemplaza por su valor), `titles` por panel, `xlabel`, `ylabel`, `labels`, `colors` y `linestyles` (diccionario, lista o un valor), `linewidth`, `alpha`, `legend`, `legend_kw`, `figsize`, `subplots`, `ncols`, `logy`, `logx`, `xlim`, `ylim`, `grid`, `color_by_R0`, `show_dfe`, `ax` (para comparar escenarios en los mismos ejes), `save` y `dpi`. El notebook `examples/06_simulaciones.ipynb` muestra casos con $R_0 < 1$ y $R_0 > 1$ en modelos epidemiológicos e intrahuésped.

### Cuándo las elecciones automáticas necesitan ayuda

* **Poblaciones cerradas** (sin nacimientos), por ejemplo el SIR clásico: el DFE no es único, así que debes indicarlo con `dfe={"S": "N"}`. El mensaje de error señala qué valor falta.
* **Varios DFE** (por ejemplo, poblaciones de vectores con crecimiento logístico): se usa el que tiene más compartimentos distintos de cero y se muestra una advertencia. Puedes elegir otro con `dfe={...}`; todos están en `model.dfe_candidates`.
* **Descomposiciones personalizadas**: la separación $\mathcal{F}$/$\mathcal{V}$ no es única. Puedes reemplazar la automática con `new_infections={"E": "beta*S*I/N"}`.

### Cómo se detectan las nuevas infecciones

En la ecuación de un compartimento infectado, un término positivo es una nueva infección cuando involucra un compartimento infectado y además (i) se pierde desde un compartimento no infectado (una transferencia como $S \to E$), o (ii) involucra un compartimento no infectado y no es una transferencia entre compartimentos infectados. La progresión ($E \to I$), la falla del tratamiento, la superinfección entre cepas, las muertes y las recuperaciones van a $\mathcal{V}$. `model.report()` muestra la clasificación de cada término y su justificación.

## Validación

El conjunto de pruebas (`pytest`) reproduce resultados conocidos: SIR (con y sin dinámica vital, con acción de masas y con incidencia dependiente de la frecuencia), SEIR, un modelo con vacunación, Ross-Macdonald, un modelo huésped-vector SEIR/SEI con transmisión humano-humano y vectores logísticos, el modelo con tratamiento de van den Driessche y Watmough (2002, §4.1), modelos intrahuésped de células blanco, células infectadas y virus, un modelo de dos cepas con superinfección, el modelo con linfocitos T helper de Cuesta-Herrera et al. (2025, Ec. 2.4 y valores de la Figura 3) y un modelo inmune de 7 ecuaciones con DFE no lineal, cuyo DFE y $R_0$ se contrastan con la integración numérica del sistema y cuyo umbral se contrasta con la estabilidad del DFE del sistema completo. `check_assumptions()` se prueba con modelos que cumplen los cinco supuestos y con modelos que violan cada uno: nuevas infecciones negativas (A1), cosecha constante (A2), asignación de nuevas infecciones a un no infectado (A3, rechazada por construcción), recaída desde un compartimento declarado no infectado (A4), y un DFE inestable por efecto Allee o una matriz $V$ que no es M-matriz (A5); los informes LaTeX se compilan con `pdflatex`. Los índices de Sobol se contrastan con su valor analítico en un modelo de forma producto y el PRCC con su definición por regresión de residuos. Los resultados simbólicos también se contrastan con el radio espectral numérico.

## Interfaz anterior

El código escrito para el notebook original sigue funcionando:

```python
from pyr0compute import GeneralEpidemiologicalModel
model = GeneralEpidemiologicalModel(variables, parameters, equations, infected_indices)
model.calculate_R0()
```

## Ejemplos

El notebook `examples/01_ejemplos.ipynb` contiene ejemplos listos para ejecutar en Google Colab: SEIR, SIR, Ross-Macdonald, un modelo huésped-vector, un modelo intrahuésped, el modelo con tratamiento de van den Driessche y Watmough, y dos cepas con superinfección. El notebook `examples/02_sensibilidad_global.ipynb` muestra el análisis de sensibilidad global (LHS-PRCC y Sobol) y su relación con el índice local. El notebook `examples/03_sistemas_no_lineales.ipynb` muestra el cálculo de $R_0$ en sistemas con DFE no lineal: el modelo de Cuesta-Herrera et al. (2025), un modelo inmune de 7 ecuaciones y un DFE sin forma cerrada. El notebook `examples/04_limite_de_tiempo_dfe.ipynb` muestra el límite de tiempo `dfe_timeout` con un modelo cuyo lazo de regulación inmune (interferón, células NK y macrófagos) no tiene solución simbólica a tiempo. El notebook `examples/05_supuestos_vdw.ipynb` muestra `check_assumptions()` con un modelo que cumple los supuestos (A1)-(A5) y con modelos que violan cada uno. El notebook `examples/06_simulaciones.ipynb` muestra `simulate()`: SIR con $R_0 < 1$ y $R_0 > 1$, parámetros aleatorios y fijos, conjuntos coloreados por $R_0$, plano de fase, bifurcación transcrítica, vacunación, modelos intrahuésped, el modelo con linfocitos T helper y Ross-Macdonald. Las figuras de los notebooks se guardan en `examples/outputs/`.

## Cita

Si usas pyR0compute en tu investigación, cítalo (ver `CITATION.cff`) junto con:

* van den Driessche, P., & Watmough, J. (2002). Reproduction numbers and sub-threshold endemic equilibria for compartmental models of disease transmission. *Mathematical Biosciences*, 180(1-2), 29-48.
* Diekmann, O., Heesterbeek, J. A. P., & Roberts, M. G. (2010). The construction of next-generation matrices for compartmental epidemic models. *Journal of the Royal Society Interface*, 7(47), 873-885.

Para el análisis de sensibilidad global:

* Marino, S., Hogue, I. B., Ray, C. J., & Kirschner, D. E. (2008). A methodology for performing global uncertainty and sensitivity analysis in systems biology. *Journal of Theoretical Biology*, 254(1), 178-196.
* Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S. (2010). Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index. *Computer Physics Communications*, 181(2), 259-270.

## Licencia

MIT, ver `LICENSE`.