"""R0 de sistemas con equilibrio libre de infección (DFE) no lineal.

Ejecutar desde la raíz del repositorio:

    pip install -e ".[global]"
    python examples/r0_sistemas_no_lineales.py

Contenido
---------
1. Modelo con linfocitos T helper de Cuesta-Herrera et al. (2025), Math. Biosci.
   Eng. 22(11):2807-2825, doi:10.3934/mbe.2025103. Se reproducen la Ec. (2.4) y
   los valores de R0* de la Figura 3.
2. Modelo intrahuésped de 7 ecuaciones con respuesta inmune (células blanco E,
   infectadas I, virus V, T helper H, T citotóxicos C, células B y T
   reguladores R). Su DFE es no lineal: H y R están acoplados y C, B dependen
   de ambos. La librería lo resuelve por bloques y entrega R0 en forma compacta.
3. Un DFE sin forma cerrada (raíz de una quíntica), que se mantiene implícito y
   se calcula numéricamente.

Cómo funciona el cálculo por bloques
------------------------------------
Con los compartimentos infectados en cero, las ecuaciones de los no infectados
se separan en las componentes fuertemente conexas de su grafo de dependencias y
se resuelven en orden topológico, por ejemplo [E] -> [H, R] -> [C] -> [B]. Los
valores ya resueltos entran a los bloques siguientes como símbolos X* (y no como
su expresión explícita), lo que mantiene pequeñas las expresiones. Los valores
cortos (como E* = g_E/d_E) se escriben completos; los demás quedan como X* en
model.R0_compact y se definen en model.dfe_definitions.
"""

import time

import sympy as sp

from pyr0compute import R0Model


def titulo(texto):
    print("\n" + "=" * 72 + f"\n{texto}\n" + "=" * 72)


# ---------------------------------------------------------------------------
# 1. Modelo con linfocitos T helper (Cuesta-Herrera et al., 2025)
# ---------------------------------------------------------------------------
titulo("1. Modelo (2.2) de Cuesta-Herrera et al. (2025): E, I, V, Th")

modelo_th = R0Model("""
    dE/dt  = lambda - d*E - kappa*E*V
    dI/dt  = kappa*E*V - alpha*I - beta*Th*I
    dV/dt  = nu*I - mu*V - tau*Th*V
    dTh/dt = b - c*Th + gamma*I*Th
""", infected=["I", "V"])

print("DFE:", modelo_th.dfe)
print("R0* =", modelo_th.R0)
p = {s.name: s for s in modelo_th.parameters}
ec_24 = (p["kappa"] * p["lambda"] * p["nu"] / p["d"]) / (
    (p["alpha"] + p["beta"] * p["b"] / p["c"]) * (p["mu"] + p["tau"] * p["b"] / p["c"]))
print("Coincide con la Ec. (2.4) del artículo:", sp.simplify(modelo_th.R0 - ec_24) == 0)

# Tabla 1 del artículo (sistema 2.2). Figura 3a: R0* = 6.0303e3
tabla1 = dict(d=0.1, kappa=1e-7, alpha=0.1, beta=1e-7, nu=995, mu=0.5,
              tau=1e-7, b=5e4, c=0.1, gamma=1e-7)
tabla1["lambda"] = 5e5
print(f"Figura 3a: R0* = {modelo_th.R0_numeric(tabla1):.4f}  (artículo: 6.0303e3)")

# Figura 3c: aumenta beta y tau; cambian d, lambda, kappa y b
fig3c = dict(tabla1, beta=1.1e-7, tau=1e-4, d=0.2, kappa=1e-8, b=2e4)
fig3c["lambda"] = 1e6
print(f"Figura 3c: R0* = {modelo_th.R0_numeric(fig3c):.4f}  (artículo: 19.8920)")

# Condición (A5) de van den Driessche y Watmough: el DFE es estable sin infección.
# En la demostración del Teorema 1 aparecen los valores propios -d y -c.
estab = modelo_th.dfe_stability(n_samples=50)
print("Valores propios del bloque no infectado:", estab["eigenvalues"])
print(f"Estable en {estab['stable']} de {estab['feasible']} muestras factibles")

# ---------------------------------------------------------------------------
# 2. Modelo de 7 ecuaciones con respuesta inmune
# ---------------------------------------------------------------------------
titulo("2. Modelo de 7 ecuaciones: E, I, V, H, C, B, R")

texto = """
    dE/dt = g_E - (d_E + tau_E*V/(V + z_VE))*E
    dI/dt = tau_E*V*E/(V + z_VE) - (d_I + tau_I*C/(C + z_CI))*I
    dV/dt = nu*I - (d_V + tau_V*B/(B + z_BV))*V
    dH/dt = g_H - (d_H - tau_H*I/(I + z_IH) + rho_H*R/(R + z_RH))*H
    dC/dt = g_C - (d_C - tau_C*H/(H + z_HC) + rho_C*R/(R + z_RC))*C
    dB/dt = g_B - (d_B - tau_B*H/(H + z_HB) + rho_B*R/(R + z_RB))*B
    dR/dt = g_R - (d_R - tau_R*H/(H + z_HR) + rho_R)*R
"""

t0 = time.time()
modelo = R0Model(texto, infected=["I", "V"])
print(f"Modelo construido en {time.time() - t0:.1f} s ({len(modelo.parameters)} parámetros)")

print("\nDFE compacto:")
for x, valor in modelo.dfe_compact.items():
    print(f"  {x} = {valor}")

print("\nDefiniciones, en el orden en que se resuelven:")
for s, valor in modelo.dfe_definitions.items():
    texto_valor = str(valor)
    if len(texto_valor) > 110:
        texto_valor = texto_valor[:110] + " ..."
    print(f"  {s} = {texto_valor}")

print("\nR0 compacto:")
sp.pprint(modelo.R0_compact)
print("\nEs decir R0 = g_E nu tau_E / (d_E z_VE k_I k_V), con")
print("  k_I = d_I + tau_I C*/(C* + z_CI)   (eliminación de infectadas por T citotóxicos)")
print("  k_V = d_V + tau_V B*/(B* + z_BV)   (neutralización del virus por células B)")
print(f"\nR0 explícito (DFE sustituido): {sp.count_ops(modelo.R0)} operaciones; "
      "por eso se reporta la forma compacta.")
print("LaTeX:", modelo.latex())

# Valores ilustrativos (no provienen de un ajuste a datos)
valores = dict(
    g_E=1.0, d_E=0.1, tau_E=2.0, z_VE=1.0, d_I=0.5, tau_I=0.5, z_CI=1.0,
    nu=10.0, d_V=1.0, tau_V=0.5, z_BV=1.0,
    g_H=1.0, d_H=1.0, tau_H=0.5, rho_H=0.3, z_IH=1.0, z_RH=1.0,
    g_C=1.0, d_C=1.0, tau_C=0.5, rho_C=0.3, z_HC=1.0, z_RC=1.0,
    g_B=1.0, d_B=1.0, tau_B=0.5, rho_B=0.3, z_HB=1.0, z_RB=1.0,
    g_R=1.0, d_R=1.0, tau_R=0.5, rho_R=0.3, z_HR=1.0,
)

print("\nDFE numérico:")
for x, valor in modelo.dfe_numeric(valores).items():
    print(f"  {x} = {valor:.6f}")
print(f"R0 = {modelo.R0_numeric(valores):.6f}")

estab = modelo.dfe_stability(valores)
print("Condición (A5): valores propios del bloque no infectado =",
      ", ".join(f"{z.real:.4f}" for z in estab["eigenvalues"]),
      "->", "estable" if estab["stable"] else "inestable")

# Umbral: el DFE del sistema completo es inestable exactamente cuando R0 > 1
print("\nUmbral (tau_E variable):")
f = sp.Matrix([modelo.equations[v] for v in modelo.variables])
J = f.jacobian(list(modelo.variables))
for tau_E in [0.005, 0.01, 2.0]:
    v = dict(valores, tau_E=tau_E)
    dfe = modelo.dfe_numeric(v)
    Jn = J.subs(dfe).subs({s: v[s.name] for s in J.free_symbols - set(modelo.variables)})
    abscisa = max(sp.re(z) for z in Jn.evalf().eigenvals())
    print(f"  tau_E = {tau_E:<5} R0 = {modelo.R0_numeric(v):8.4f}   "
          f"máx Re(lambda) del sistema completo = {float(abscisa):+.4f}")

# Sensibilidad global: R0 se evalúa en cadena (H*, R* -> C*, B* -> R0)
try:
    t0 = time.time()
    res = modelo.prcc(baseline=valores, spread=0.2, n=1000, seed=1)
    print(f"\nLHS-PRCC (n = 1000, ±20 %) en {time.time() - t0:.1f} s; los 8 más influyentes:")
    tabla = res.as_dict()
    for nombre in res.ranking()[:8]:
        print(f"  {nombre:<7} PRCC = {tabla[nombre]['PRCC']:+.3f}")
except ImportError:
    print("\n(Instala SciPy para la sensibilidad global: pip install 'pyR0compute[global]')")

# ---------------------------------------------------------------------------
# 3. DFE sin forma cerrada
# ---------------------------------------------------------------------------
titulo("3. DFE implícito: raíz de una quíntica")

modelo_q = R0Model("""
    dS/dt = Lambda - mu*S - k*S^5 - beta*S*I
    dI/dt = beta*S*I - (gamma + mu)*I
""", infected=["I"])
print("DFE implícito:", modelo_q.dfe_is_implicit, "->", modelo_q.dfe_definitions)
print("R0 =", modelo_q.R0_compact)
v = dict(Lambda=2.0, mu=0.5, k=0.1, beta=1.5, gamma=0.3)
print("S* numérico:", modelo_q.dfe_numeric(v)[sp.Symbol("S")])
print("R0 numérico:", modelo_q.R0_numeric(v))

# ---------------------------------------------------------------------------
# Informe completo (para pegar en un artículo: report_latex(standalone=True))
# ---------------------------------------------------------------------------
titulo("Informe paso a paso del modelo de 7 ecuaciones")
print(modelo.report())
