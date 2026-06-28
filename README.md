# Simulación de Cristales de Tiempo Continuos y Dinámica No Lineal (`Time_crystal_sim`)

Este repositorio contiene una suite completa de simulación y análisis computacional en Python diseñada para modelar, integrar y caracterizar la dinámica no lineal macroscópica de un **Cristal de Tiempo Continuo (CTC)**. El sistema físico modelado se basa en la interacción hiperfina no lineal acoplada entre espines electrónicos y nucleares en semiconductores (estructuras de pozos cuánticos de InGaAs).

El marco físico y matemático de las simulaciones implementadas replica y extiende el modelo macroscópico teórico desarrollado en las investigaciones de *Greilich et al.*:
1. **Nature Communications (2025):** *Exploring nonlinear dynamics in periodically driven time crystal from synchronization to chaotic motion*.
2. **arXiv Preprint (2023):** *Continuous time crystal in an electron-nuclear spin system: stability and melting of periodic auto-oscillations*.

---

## Descripción del Proyecto

Un cristal de tiempo continuo representa una fase de la materia fuera del equilibrio termodinámico que rompe espontáneamente la simetría de traslación temporal continua del hamiltoniano del sistema. Bajo la influencia de un bombeo óptico constante y continuo (es decir, una excitación completamente independiente del tiempo y sin frecuencias externas impuestas), el campo acoplado de espines induce auto-oscilaciones periódicas autosostenidas y macroscópicamente estables.

Este proyecto abarca el modelado completo de dicha fase cuántica/clásica no lineal, analizando:
* La emergencia autónoma del cristal de tiempo a través de una bifurcación de Hopf.
* La respuesta del cristal ante modulaciones periódicas externas de la polarización del láser, dando lugar a fenómenos de enganche de fase (*phase-locking*), la formación de **Lenguas de Arnold** y la emergencia de la estructura fractal de la **Escalera del Diablo**.
* El comportamiento crítico en las fronteras caóticas del sistema y las limitaciones espectrales de los algoritmos de detección.
* La extensión hacia un modelo espacial extendido unidimensional acoplado por transporte difusivo de espín, visualizando la propagación de ondas de sincronización coherentes.

---

## Estructura del Repositorio

El repositorio se organiza en cuatro scripts principales e independientes en Python, optimizados para alto rendimiento mediante vectorización masiva:

* **`Time_crystal.py`**: Modela el cristal de tiempo continuo en su estado autónomo puro. Resuelve la evolución temporal libre, revelando las auto-oscilaciones sostenidas asimétricas con su característica geometría en forma de "M", y reconstruye el retrato de fase bidimensional mediante mapas de retardo para evidenciar el atractor del ciclo límite estable.
* **`Periodically_Driven.py`**: Integra la dinámica del cristal de tiempo bajo el efecto de una modulación periódica externa de frecuencia fija ($f_m$) y amplitud ($S_m$). Permite inspeccionar las perturbaciones topológicas en el ciclo límite y estudiar las interacciones no lineales directas en el dominio del tiempo y del espacio de fase.
* **`devils_staircase_optimized.py`**: Código de cómputo paralelo vectorizado en lote (*batch processing*). Realiza un barrido simultáneo sobre cientos de frecuencias de modulación ($f_m$), procesando las señales en estado estacionario para extraer el número de enrollamiento fraccionario ($W$) y mapear la estructura de la Escalera del Diablo sin bucles anidados en Python.
* **`1D_solver_HeatMap.py`**: Extiende el modelo físico a un sistema espacial continuo de una dimensión (PDE). Incorpora un perfil de bombeo Gaussiano para el láser y un operador Laplaciano discreto para simular la difusión magnética de espín, generando un mapa de contorno térmico (*heatmap*) que documenta la propagación de ondas macroscópicas de sincronización.

---

## Fundamentos Físicos y Métodos Computacionales Avanzados

### 1. Aproximación Adiabática y Reducción Dimensional
El sistema físico presenta una separación severa de escalas temporales: la polarización del espín electrónico responde en la escala de los nanosegundos, mientras que el campo nuclear de Overhauser evoluciona lentamente en el orden de los segundos. Esta disparidad introduce una rigidez matemática extrema (*stiffness*) que inutiliza los métodos de integración estándar si se intentan resolver ambas derivadas en simultáneo.

Para solucionar este cuello de botella computacional, se aplicó una aproximación adiabática, asumiendo que el espín electrónico alcanza instantáneamente su estado cuasi-estacionario para cualquier configuración dada del campo nuclear. Esto permite resolver analíticamente la ecuación de Bloch acoplada mediante inversión matricial en cada paso temporal:

$$\mathbf{S} = (\mathbf{I} - \mathbf{M}_K)^{-1} \mathbf{S}_0$$

Donde $\mathbf{M}_K$ es la matriz antisimétrica asociada al producto cruz con el campo magnético efectivo total $\mathbf{K} = k(\mathbf{B}_{\text{ext}} + \mathbf{B}_N)$, e $\mathbf{I}$ es la matriz identidad de $3 \times 3$. Gracias a esta reducción analítica, el solucionador numérico integra únicamente la evolución macroscópica del campo nuclear de Overhauser ($\mathbf{B}_N$):

$$\frac{d\mathbf{B}_N}{dt} = -\frac{1}{T_N} (\mathbf{B}_N - \hat{\alpha}\mathbf{S})$$

### 2. Integración Numérica Adaptativa (`RK45`)
Al eliminar la rigidez mediante la aproximación adiabática, la ecuación diferencial ordinaria (EDO) resultante para el campo de Overhauser se vuelve numéricamente dócil. En los scripts `Time_crystal.py` y `Periodically_Driven.py`, la integración temporal se realiza utilizando el método explícito de Runge-Kutta de orden 5(4) adaptativo (`RK45` de SciPy). 

Para garantizar la captura exacta de la bifurcación de Hopf y evitar la acumulación de errores de redondeo en trayectorias caóticas de larga duración, se impusieron estrictos controles de paso mediante tolerancias finas: una tolerancia relativa (`rtol`) de $10^{-6}$ y una tolerancia absoluta (`atol`) de $10^{-8}$.

### 3. Integración Vectorizada en Lote (Batching)
Para construir la Escalera del Diablo en `devils_staircase_optimized.py`, se requiere simular el sistema bajo cientos de frecuencias de modulación concurrentes. Ejecutar llamadas secuenciales en bucles `for` con el solver de SciPy resultaba prohibitivo. 

Para optimizar el rendimiento, se desarrolló un integrador de Runge-Kutta de 4to orden (RK4) de paso fijo **masivamente vectorizado**. Aprovechando el *broadcasting* multidimensional de NumPy, el integrador avanza en paralelo el estado de todas las frecuencias de modulación en una sola operación tensorial, explotando las capacidades nativas de subrutinas LAPACK para resolver simultáneamente miles de sistemas lineales de $3 \times 3$ por segundo, acelerando el cómputo en más de un 30x.

### 4. Análisis Espectral mediante FFT y Zero-Padding
Tradicionalmente, la frecuencia de auto-oscilación ($f_{AO}$) se calcula midiendo el intervalo medio entre cruces por cero en el dominio del tiempo. Sin embargo, en regímenes cuasi-periódicos o cercanos al caos, la señal experimenta modulaciones de amplitud extremas que rozan el cero, induciendo conteos falsos y saltos ruidosos espurios en el gráfico.

En su lugar, este proyecto implementa un análisis espectral en estado estacionario mediante la Transformada Rápida de Fourier Real (`np.fft.rfft`). Para refinar la resolución en el dominio de las frecuencias sin extender el tiempo físico de simulación, se incorpora la técnica de **zero-padding** (relleno de ceros con un `pad_factor = 10`). Esta aproximación suaviza el espacio de frecuencias digital y permite localizar el pico espectral exacto con alta fidelidad, logrando estabilizar las mesetas horizontales de sincronización correspondientes a lenguas de Arnold de orden superior (como el escalón de $1/2$ y $2/3$).

### 5. Retratos de Fase e Incrustación de Retardo (*Delay Embedding*)
Para mapear la topología de los atractores del cristal de tiempo, se emplea el teorema de incrustación de retardo (delay embedding), proyectando la componente $S_z(t)$ frente a su versión desplazada en el tiempo $S_z(t - \tau)$.

** Clarificación sobre la topología geométrica:** De acuerdo con el Teorema de Existencia y Unicidad de Picard-Lindelöf para sistemas dinámicos deterministas, las trayectorias en el espacio de fase real (el cual posee un mínimo de tres dimensiones físicas independientes) **nunca pueden cruzarse consigo mismas**. No obstante, en las proyecciones bidimensionales generadas en los scripts, se observarán intersecciones visibles en las curvas del retrato de fase. 

Es fundamental destacar que estos cruces **no constituyen un error físico del modelo ni una falla de convergencia del integrador numérico**, sino que corresponden a un *artefacto topológico de proyección* (un "efecto sombra" que ocurre al colapsar una trayectoria tridimensional sobre un plano 2D, condicionado por la magnitud del retardo discreto $\tau$). La unicidad y causalidad física de la órbita del ciclo límite o del atractor extraño permanecen intactas en las dimensiones superiores del sistema.

### 6. Dinámica Espaciotemporal Unidimensional Vectorizada
El script `1D_solver_HeatMap.py` modela el comportamiento del cristal a lo largo de una dimensión espacial extendida añadiendo un operador difusivo Laplaciano discreto (diferencias finitas de segundo orden) a la EDO local del campo nuclear. 

Para acoplar la difusión espacial de forma eficiente con un integrador explícito (`RK45`), se vectorizaron por completo los gradientes espaciales y el tensor de polarización a lo largo de la grilla de $N=41$ puntos. Esto elimina los bucles iterativos espaciales dentro de la función derivada de la PDE, permitiendo simular con fluidez la formación de ondas de sincronización macroscópicas estables que se propagan radialmente desde el centro óptico Gaussiano.

---

## Parámetros Físicos y Configuración por Defecto

Los scripts ejecutan por defecto los parámetros físicos calibrados para reproducir los regímenes fenomenológicos clave analizados en la literatura:
* **Factor DNP Isotrópico ($a_N$):** $20.0\text{ mT}$ (Fuerza de la polarización nuclear dinámica isotrópica)
* **Factor DNP Anisotrópico ($b_N$):** $21.0\text{ mT}$ (Alineación anisotrópica con el campo externo)
* **Tiempo de Relajación Nuclear ($T_N$):** $0.5\text{ s}$ (Establece la escala de tiempo macroscópica de la simulación)
* **Campo Magnético Transversal ($B_x$):** $-1.0\text{ mT}$
* **Campo Magnético Longitudinal ($B_z$):** $0.1\text{ mT}$ (Sintonizado para forzar la bifurcación de Hopf a un ángulo de inclinación crítico $\approx 10^\circ$)
* **Intensidad de Acoplamiento de Retroalimentación ($k$ o `k_scale`):** $0.7$
* **Frecuencia Natural Intrínseca Autogenerada ($f_0$):** $\approx 0.125\text{ Hz}$ (Período de auto-oscilación libre $T_0 \approx 8.0\text{ s}$)

---

## Instalación y Uso

El proyecto requiere una instalación estándar de Python 3 equipada con el ecosistema de computación científica optimizado de NumPy y SciPy.

```bash
pip install numpy scipy matplotlib
```


