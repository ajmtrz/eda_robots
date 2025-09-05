# Auditoría técnica del repositorio (Trading cuantitativo y ML)

Fecha: 2025-09-05

## Executive summary
El repositorio implementa una búsqueda de estrategias con Optuna, backtesting JIT con Numba, filtrado de confiabilidad (MAPIE + filtro causal OOB), y exportación a ONNX/MQL5. La separación temporal IS/OOS está bien cuidada en la construcción de datasets y en la evaluación final. El Monkey Test está presente, respeta direccionalidad y usa block bootstrap circular con p-valor estable (k+1)/(N+1), pero no incluye el modo windowed ni la combinación Holm–Bonferroni exigida, y no considera comisiones; además, la decisión de ejecutar el test solo se dispara cuando el score supera al best actual, lo que puede dejar pasar falsos positivos intermedios.

Riesgos principales:
- Falta de Monkey Test “windowed” OOS con combinación de p-values (Holm–Bonferroni) y umbrales por ventana; no se reporta percentile rank/z-score por ventana; no hay gating universal (se condiciona a superar el best actual).
- Validación interna de modelos `main`/`meta` con `train_test_split` aleatorio dentro del segmento IS, rompiendo causalidad temporal; debería usarse `TimeSeriesSplit` o split por fecha.
- Backtest y Monkey Test sin comisiones/deslizamiento, potencial sobreestimación de rendimiento y significancia.
- Limpieza de artefactos con `os.remove` condicionada al trial “best” que podría eliminar archivos aún necesarios si hay condiciones de carrera o reentradas.
- Reproducibilidad: uso extenso de `np.random` sin semilla controlada fuera de tests; resultados no reproducibles por diseño.

## Tabla de issues

| ID | Severidad | Archivo(s):líneas | Resumen | Repro | Impacto | Sugerencia |
|----|-----------|-------------------|---------|-------|---------|------------|
| I-01 | Critical | `studies/modules/StrategySearcher.py`:1000-1143; `studies/modules/tester_lib.py`:873-923 | Monkey Test sin modo windowed/comb. Holm–Bonferroni ni umbrales por ventana; gating condicionado al best actual | Ejecutar búsqueda con múltiples trials; observar que `run_monkey_test` se llama solo si `score` > best y no hay segmentación OOS | Falsos positivos, aceptación de estrategias no robustas temporalmente | Ejecutar Monkey Test siempre para candidatos con score>0, en modo windowed OOS con p-values combinados (Holm–Bonferroni) y umbrales por ventana (percentil y z-score); gatear aceptación por `monkey_pass` global |
| I-02 | High | `studies/modules/StrategySearcher.py`:882-889, 942-948 | Uso de `train_test_split(shuffle=True)` para validación interna IS | Re-entrenar y validar con datos IS produce fuga temporal | Sobreajuste y estimaciones optimistas | Reemplazar por `TimeSeriesSplit` o split por fecha mantenida; sin barajar |
| I-03 | High | `studies/modules/tester_lib.py`:228-384; 858-871; 754-764 | Backtest y simulaciones nulas sin costes (comisiones/deslizamiento) | Correr `tester` y comparar PnL con y sin coste hipotético | Sobreestimación de rendimiento y Sharpe; p-valor sesgado | Introducir costes coherentes (p.ej., 0.005% por orden) tanto en backtest como en `run_monkey_test` |
| I-04 | Medium | `studies/modules/StrategySearcher.py`:166-236 | Limpieza de artefactos con `os.remove` de modelos/datasets previos | Simular interrupciones/hilos; verificar borrado de `best_*` en transiciones | Riesgo de borrar artefactos no relacionados si hay carreras o etiquetas inconsistentes | Añadir verificación de `model_seed`/tag y existencia antes de borrar; logs y try/except con contexto |
| I-05 | Medium | `studies/modules/StrategySearcher.py`:1073-1107; `studies/modules/tester_lib.py`:783-871 | Tamaño de bloque del bootstrap estimado por mediana de rachas, pero no documentado ni persistido | Correr `run_monkey_test` y revisar ausencia de doc/persistencia | Dificulta trazabilidad y auditoría estadística | Documentar y persistir `block_size` usado y distribución de sharpes |
| I-06 | Medium | `studies/modules/StrategySearcher.py`:1120-1125; `studies/modules/export_lib.py`:31-34 | `export_dataset_to_csv` usa archivos temporales fuera de `./data/<tag>` hasta copia posterior | Forzar fallo entre generación y copia | Riesgo de perder dataset si proceso interrumpe antes de copiar | Guardar directamente en `./data/<tag>/<model_seed>.csv` o usar atomic moves |
| I-07 | Medium | `studies/modules/StrategySearcher.py`:1863-1894; 1847-1851 | Separación IS/OOS correcta pero `get_train_test_data` no se usa en todos los flujos | Traza en `fit_final_models` | Riesgo de inconsistencias si futuros cambios usan otro camino | Centralizar creación de IS/OOS vía una sola función |
| I-08 | Medium | `studies/modules/StrategySearcher.py`:1149-1294; 1300-1495 | MAPIE con `CrossConformalClassifier` configurado adecuadamente, pero sin evaluación OOS del propio filtro | Revisar `apply_mapie_filter` | Posible sobre-ajuste del filtro a IS | Validar el filtro en ventanas OOS o nested-TS CV |
| I-09 | Low | `studies/modules/tester_lib.py`:221-510 | Métrica WFV custom con pesos fijos y sin hipers ajustables | Ejecutar con diversas series | Control limitado sobre sensibilidad/robustez | Exponer hiperparámetros de WFV y documentar |
| I-10 | Low | `studies/tests/` | No hay tests para Monkey Test, WFV ni exportación MQL5 end-to-end | Correr pytest | Cobertura incompleta | Añadir tests unitarios y E2E de estos componentes |
| I-11 | Low | `studies/modules/StrategySearcher.py`:881-889; 942-948 | Reproducibilidad: `random_state=None` y sin semillas globales | Repetir ejecuciones | No reproducible | Loggear semillas efectivas y opción de fijarlas |
| I-12 | Low | `studies/modules/tester_lib.py`:60-67; 754-766 | Tipos/dtypes: conversiones simultáneas dentro de `tester`; posibles copias | Revisar perfiles | Overhead menor | Asegurar contigüidad y dtypes aguas arriba |

## Secciones detalladas

### Integridad estadística y Monkey Test
- Direccionalidad: sanitización correcta de posiciones según modo long/short/both: ver `studies/modules/tester_lib.py`:766-780.
- Bootstrap por bloques circular: implementado en `_block_bootstrap_positions` con wrap-around: ver `studies/modules/tester_lib.py`:831-856.
- p-valor estable (k+1)/(N+1) y percentile rank: implementado en `run_monkey_test`: `studies/modules/tester_lib.py`:911-923.
- Falta modo “windowed” OOS, combinación Holm–Bonferroni y umbrales mínimos por ventana (percentil y z-score): no hay referencias: `studies/modules/tester_lib.py` y `StrategySearcher.py` sin coincidencias de “Holm”, “Bonferroni”, “windowed”.
- No aplica SR scaling ni batching: no se observa SR scaling ni batching; `n_simulations` se ejecuta en un bucle simple: `studies/modules/tester_lib.py`:905-910.
- Estimación dinámica de block size: presente como mediana de rachas acotada [5,64], pero no documentada externamente ni persistida: `studies/modules/tester_lib.py`:783-829.
- JIT Numba en funciones críticas: `_compute_sharpe`, sanitización, estimación de bloque, bootstrap y simulación están `@njit(cache=True)`: `studies/modules/tester_lib.py`:754-871.
- Persistencia en Optuna: `monkey_p_value`, `monkey_percentile`, `monkey_pass` se guardan en `trial.user_attrs` y se promueven a `study.user_attrs` al actualizar el “best”: `studies/modules/StrategySearcher.py`:178-186; 198-205; 405-413; 498-503.
- Gating de aceptación: si `monkey_pass` es False, se fuerza `score = -1.0`, pero solo se ejecuta el test si el score supera al best actual: `studies/modules/StrategySearcher.py`:1061-1107. Esto no bloquea sistemáticamente estrategias candidatas que no superen el test.
- Comisiones: tanto el backtest como el null no aplican comisiones/deslizamiento, confirmando 0.0% actual y no coherente con 0.005% esperada: `studies/modules/tester_lib.py`:228-384 y 858-871.

Reproducción mínima:
1) Configurar un `config` en `studies/main_searcher.ipynb` y ejecutar una búsqueda. 2) Observar en logs que `run_monkey_test` se ejecuta solo cuando `score` supera el best y que no hay segmentación/windowing en el test. 3) Comparar resultados con un backtest hipotético con coste fijo para verificar sensibilidad.

Recomendación: ejecutar Monkey Test siempre para candidatos viables, con segmentación por ventanas OOS y combinar p-values con Holm–Bonferroni. Aplicar umbrales por ventana (percentil y z-score). Incluir costes transaccionales coherentes en backtest y null.

### WFV/OOS y estabilidad
- `_walk_forward_validation` implementa una métrica de consistencia temporal con varias escalas y solapamiento: `studies/modules/tester_lib.py`:389-510. No hay relación explícita con tamaño mínimo OOS del Monkey Test ni combinación de p-values por ventanas.
- Peso del score final favorece consistencia (55%): `studies/modules/tester_lib.py`:127-135.

Recomendación: alinear ventanas del WFV con las ventanas del Monkey Test windowed y documentar tamaños/solapamientos.

### No-leakage y splits temporales
- `get_labeled_full_data` recorta rigurosamente IS/OOS por fecha sin solapamiento: `studies/modules/StrategySearcher.py`:1847-1851; 1853-1857.
- `get_train_test_data` aplica máscaras temporales consistentes: `studies/modules/StrategySearcher.py`:1863-1894.
- Problema: en `fit_final_models`, la validación de `main` y `meta` usa `train_test_split(..., shuffle=True, random_state=None)`: `studies/modules/StrategySearcher.py`:882-889; 942-948. Esto rompe causalidad incluso dentro de IS.

Recomendación: usar `TimeSeriesSplit` para validación interna o cortes al final del período IS sin aleatoriedad.

### Exportaciones ONNX y MQL5
- Subcarpeta por `tag` y `model_seed` por archivo: `studies/modules/export_lib.py`:83-91; 925-929.
- `#resource` en MQL5 referencia `\Files\{tag}\{filename}`: `studies/modules/export_lib.py`:789-795.
- Limpieza: se borran modelos temporales y dataset temporal tras copiar; riesgo si se interrumpe antes de copiar (dataset temporal): `studies/modules/export_lib.py`:31-34; 107-119.
- `export_dataset_to_csv()` guarda primero en temp y luego copia a `./data/<tag>/<model_seed>.csv`: `studies/modules/export_lib.py`:31-34 y 108-116. No viola rutas, pero mejor escribir directo al destino.

### Confiabilidad (MAPIE / causal filtering / meta-model)
- MAPIE configurado con `confidence_level`, `cv` y `conformity_score='lac'`: `studies/modules/StrategySearcher.py`:1257-1262. Combina con precisión: `combined_scores = (set_sizes==1) & (predicted==y)`: `studies/modules/StrategySearcher.py`:1267-1272.
- Filtro causal OOB con múltiples learners y umbralización por percentil: `studies/modules/StrategySearcher.py`:1322-1376; 1456-1476; 1458-1466.
- Construcción OOF para meta-modelo: `create_oof_meta_mask` con `TimeSeriesSplit` para OOF del main y umbral por residuo/ confianza: `studies/modules/StrategySearcher.py`:1929-1972; 1974-2022. Índices alineados correctamente.

Recomendación: validar ambos filtros en OOS puro o nested CV para evitar overfitting de filtros.

### Config/entorno/logging
- No hay uso de `.env`, `LOG_LEVEL`, ni scripts `run_all.sh`; no hay RPCs ni `TAKER_ADDR`: búsquedas negativas en repo.
- `.gitignore` ignora `optuna_dbs/`, `data/`, `logs/`: `/.gitignore`:1-10.

Recomendación: documentar variables de entorno si se añaden integraciones futuras.

### Seguridad y robustez
- Borrados con `os.remove` están envueltos en checks de existencia, pero sin logs de contexto; podrían borrar artefactos de otro tag si se comparte estado: `studies/modules/StrategySearcher.py`:166-236; 636-654; 650-654; `export_lib.py`:101-106; 116-119.
- Manejo de errores: varias rutas devuelven `-1.0` o arrays vacíos; errores no se elevan, pero se registran en debug. Aceptable, aunque puede ocultar fallos si `debug=False`.

Recomendación: mejorar mensajes de error y asociar borrados a `model_seed`/tag explícito en logs.

### Rendimiento y reproducibilidad
- Numba: firmas y dtypes adecuados; arrays contiguos en `tester`: `studies/modules/tester_lib.py`:63-66. Evitan conversiones en bucle.
- Aleatoriedad sin semilla global en múltiples puntos (`labeling_lib.py`, `_block_bootstrap_positions`): ver coincidencias de `np.random.randint`.
- Coste de `n_simulations`: loop simple en Python + Numba-jitted helpers; escalable pero costoso si `5000` elevado.

Recomendación: permitir semilla opcional para reproducibilidad y logging de semilla efectiva; paralelizar simulaciones si es necesario (sin batching que cambie distribución).

### Calidad de código y notebooks
- `StrategySearcher.py` y `labeling_lib.py` son largos y multifunción; responsabilidades mezcladas; nombres razonables pero muchos prints `DEBUG`.
- Notebook `main_searcher.ipynb` con ejecución paralela y configuración embebida; genera `tag` por config; usa `optuna` directamente y puede tener side-effects al crear DBs: `studies/main_searcher.ipynb`:12-81.

Recomendación: mover configuración a módulos o YAML, y usar scripts reproducibles.

### Tests/E2E
- Tests cubren backtest direccionalidad: `studies/tests/test_backtest_parity.py`:1-78.
- Test de convertidor ONNX: `studies/tests/test_onnx_converter.py`:1-115.
- No hay tests para Monkey Test, WFV, exportación MQL5 end-to-end.

## Hallazgos “Needs verification”
- Confirmar política oficial de comisiones actuales (0.005% por orden). No hay código aplicándolo; se asume inactivo. Verificar si debe activarse.
- Verificar requerimiento exacto de “windowed” Monkey Test (número de ventanas, tamaños mínimos, solapamientos) para implementar umbrales por ventana y combinación Holm–Bonferroni.

## Anexos

- Fragmentos citados clave:

```1000:1143:studies/modules/StrategySearcher.py
if self.debug:
    print(f"🔍 DEBUG: Inicializando backtest del periodo OOS")
score_oos, equity_curve, returns_series, pos_series = tester(...)
...
monkey_res = run_monkey_test(
    actual_returns=returns_series if returns_series is not None else None,
    price_series=price_series,
    pos_series=pos_series if pos_series is not None else None,
    direction=self.direction,
    n_simulations=self.monkey_n_simulations,
)
...
if not monkey_pass:
    score = -1.0
```

```754:871:studies/modules/tester_lib.py
@njit(cache=True)
def _compute_sharpe(returns: np.ndarray) -> float: ...
@njit(cache=True)
def _sanitize_positions_for_direction_code(...): ...
@njit(cache=True)
def _estimate_block_size_jit(pos: np.ndarray) -> int: ...
@njit(cache=True)
def _block_bootstrap_positions(...): ...
@njit(cache=True)
def _simulate_returns_from_positions(...): ...
```

```882:889:studies/modules/StrategySearcher.py
train_df_main, val_df_main = train_test_split(
    model_main_train_data,
    test_size=0.2,
    stratify=model_main_train_data['labels_main'],
    shuffle=True,
    random_state=None
)
```

```389:510:studies/modules/tester_lib.py
@njit(cache=True)
def _walk_forward_validation(eq, trade_profits):
    ...  # ventanas, pesos, penalización de estabilidad
```

```83:91:studies/modules/export_lib.py
# subcarpeta por tag
filename_model_main = f"{model_seed}_main.onnx"
filename_model_meta = f"{model_seed}_meta.onnx"
```

- Archivos relevantes inspeccionados: `studies/modules/StrategySearcher.py`, `studies/modules/tester_lib.py`, `studies/modules/export_lib.py`, `studies/modules/labeling_lib.py`, `studies/tests/*`, `studies/main_searcher.ipynb`, `.gitignore`.

- Checklist de conformidad
  - Integridad estadística y Monkey Test: ✗ (faltan windowed + Holm–Bonferroni, costes)
  - WFV/OOS y estabilidad: ✓ (WFV presente), parcial ✗ (no ligado al Monkey Test)
  - No-leakage y splits temporales: ✗ (uso de `train_test_split` aleatorio en IS)
  - Exportaciones ONNX/MQL5: ✓ (tag/seed/rutas correctas), riesgo menor de temp
  - Confiabilidad (MAPIE/causal/meta): ✓ (config correcta), parcial ✗ (val. OOS)
  - Config/entorno/logging: ✓/N.A. (sin `.env` ni RPCs)
  - Seguridad/robustez: ✓ con observaciones de borrado
  - Rendimiento/reproducibilidad: ✓ con observaciones de semillas
  - Calidad de código y notebooks: ✓ con recomendaciones
  - Tests/E2E: ✗ (faltan para Monkey Test, WFV, export MQL5)

## Recomendaciones priorizadas
1) Implementar Monkey Test windowed OOS con combinación Holm–Bonferroni y umbrales por ventana; ejecutar siempre que `score>0`; gatear aceptación a `monkey_pass=true`. Done when: p-valor global y por ventana se reportan y `study.user_attrs` guarda `monkey_*` y `monkey_pass`.
2) Sustituir `train_test_split` por `TimeSeriesSplit` o cortes temporales en `fit_final_models` para `main` y `meta`. Done when: no hay mezclas temporales y las métricas IS usan validación causal.
3) Añadir costes transaccionales (p.ej., 0.005% por orden) en backtest y `run_monkey_test`, coherentes entre ambos. Done when: tests muestran paridad de coste y documentación lo refleja.
4) Amarrar ventanas WFV a ventanas del Monkey Test windowed y exponer hiperparámetros de tamaño/pesos. Done when: parámetros configurables y documentación.
5) Persistir y documentar `block_size` y estadísticas de simulación del Monkey Test (media, std, percentiles). Done when: `trial/study.user_attrs` incluyen estos campos.
6) Guardar dataset final directamente en `./data/<tag>/<model_seed>.csv` o mover atómicamente; registrar ruta en logs. Done when: no se usa archivo temporal intermedio sin copia garantizada.
7) Endurecer borrados: validar tag/model_seed antes de `os.remove`, incluir logs de contexto. Done when: registros muestran qué se borra y por qué.
8) Reproducibilidad opcional: opción para fijar semilla global y loggear semillas efectivas en cada trial. Done when: logs reproducibles y bandera de semilla.
9) Añadir tests unitarios para Monkey Test (direccionalidad, bootstrap, p-valor) y WFV; tests E2E de exportación MQL5 compilable. Done when: pytest cubre estos casos.
10) Documentar configuraciones en un README y parametrizar notebook hacia scripts reproducibles. Done when: hay guía de ejecución sin estado del notebook.
