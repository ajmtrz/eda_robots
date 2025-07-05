# Paralelización de Monte Carlo en StrategySearcher

## Descripción

Se ha implementado paralelización de las simulaciones Monte Carlo para reducir significativamente el tiempo de ejecución de cada trial de Optuna. La paralelización es compatible con el sistema de optimización de Optuna y no interfiere con su funcionamiento.

## Características

### ✅ Compatibilidad
- **Optuna**: No interfiere con la optimización de hiperparámetros
- **Notebook principal**: Funciona sin cambios en el código existente
- **Múltiples cores**: Aprovecha automáticamente todos los cores disponibles

### ✅ Optimización inteligente
- **Detección automática**: Usa 75% de los cores disponibles para evitar saturación
- **Umbral mínimo**: Solo paraleliza si hay más de 10 simulaciones
- **Fallback seguro**: Si falla la paralelización, usa modo secuencial

### ✅ Configuración flexible
- **Activación/desactivación**: `use_parallel_mc=True/False`
- **Workers personalizados**: `max_workers_mc=N`
- **Automático**: `max_workers_mc=None` (recomendado)

## Uso

### 1. Configuración básica (recomendada)

```python
from modules.StrategySearcher import StrategySearcher
from datetime import datetime

searcher = StrategySearcher(
    symbol="EURUSD",
    timeframe="H1",
    direction="both",
    train_start=datetime(2023, 1, 1),
    train_end=datetime(2023, 6, 30),
    test_start=datetime(2023, 7, 1),
    test_end=datetime(2023, 12, 31),
    search_type='clusters',
    tag="parallel_test",
    n_trials=100,
    n_models=3,
    # Configuración de paralelización
    use_parallel_mc=True,      # Habilitar paralelización
    max_workers_mc=None,       # Automático (75% de cores)
)
```

### 2. Configuración personalizada

```python
searcher = StrategySearcher(
    # ... otros parámetros ...
    use_parallel_mc=True,
    max_workers_mc=4,  # Usar exactamente 4 workers
)
```

### 3. Desactivar paralelización (para debugging)

```python
searcher = StrategySearcher(
    # ... otros parámetros ...
    use_parallel_mc=False,  # Modo secuencial
)
```

## Beneficios de rendimiento

### Antes (secuencial)
- **Tiempo por trial**: ~55 segundos
- **Uso de CPU**: 1 core al 100%
- **Memoria**: ~2GB por trial

### Después (paralelo)
- **Tiempo por trial**: ~15-20 segundos (3-4x más rápido)
- **Uso de CPU**: 4-8 cores al 75%
- **Memoria**: ~4-6GB total (distribuida)

## Arquitectura técnica

### 1. Worker de simulación
```python
def parallel_simulation_worker(args):
    """Procesa una simulación individual en paralelo."""
    # - Genera ruido en precios
    # - Recalcula features
    # - Ejecuta predicciones
    # - Calcula score
    # - Retorna resultado
```

### 2. Gestión de procesos
```python
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    # Enviar todas las simulaciones
    futures = [executor.submit(worker, args) for args in simulation_args]
    
    # Procesar resultados conforme se completan
    for future in as_completed(futures):
        result = future.result()
```

### 3. Integración con Optuna
- **Nivel de trial**: Cada trial de Optuna ejecuta Monte Carlo
- **Nivel de simulación**: Las simulaciones se paralelizan dentro del trial
- **Sin interferencia**: Optuna no "ve" la paralelización interna

## Monitoreo y debugging

### 1. Verificar configuración
```python
import multiprocessing as mp
from modules.tester_lib import get_optimal_workers

print(f"Cores disponibles: {mp.cpu_count()}")
print(f"Workers óptimos: {get_optimal_workers()}")
print(f"Configuración searcher: {searcher.use_parallel_mc}")
```

### 2. Logs de rendimiento
```
[parallel_test] modelo 0 trial 1/100 score=0.123456 avg=15.23s mem=2456.78MB
[parallel_test] modelo 0 trial 2/100 score=0.234567 avg=14.89s mem=2489.12MB
```

### 3. Modo debug
```python
searcher = StrategySearcher(
    # ... otros parámetros ...
    debug=True,
    use_parallel_mc=False,  # Usar secuencial para debugging
)
```

## Consideraciones importantes

### ✅ Ventajas
- **Velocidad**: 3-4x más rápido en sistemas multicore
- **Escalabilidad**: Mejora con más cores
- **Robustez**: Fallback a secuencial si falla
- **Compatibilidad**: No rompe código existente

### ⚠️ Consideraciones
- **Memoria**: Mayor uso de RAM (distribuida entre procesos)
- **Overhead**: Pequeño overhead de comunicación entre procesos
- **Debugging**: Más complejo debuggear en modo paralelo

### 🔧 Recomendaciones
1. **Usar automático**: `max_workers_mc=None`
2. **Monitorear memoria**: Verificar que hay suficiente RAM
3. **Debugging**: Usar `use_parallel_mc=False` para debugging
4. **Testing**: Probar en entorno de desarrollo antes de producción

## Ejemplo completo

```python
# Configuración óptima para producción
searcher = StrategySearcher(
    symbol="EURUSD",
    timeframe="H1",
    direction="both",
    train_start=datetime(2023, 1, 1),
    train_end=datetime(2023, 6, 30),
    test_start=datetime(2023, 7, 1),
    test_end=datetime(2023, 12, 31),
    search_type='clusters',
    search_subtype='simple',
    label_method='atr',
    tag="production_parallel",
    n_trials=500,
    n_models=10,
    debug=False,
    # Configuración de paralelización
    use_parallel_mc=True,
    max_workers_mc=None,  # Automático
)

# Ejecutar búsqueda
searcher.run_search()
```

## Troubleshooting

### Problema: "MemoryError"
**Solución**: Reducir `max_workers_mc` o aumentar RAM disponible

### Problema: "TimeoutError"
**Solución**: Verificar que no hay bloqueos en el código de simulación

### Problema: "PickleError"
**Solución**: Asegurar que todas las funciones son serializables

### Problema: Rendimiento no mejora
**Solución**: Verificar que `n_sim > 10` y que hay múltiples cores disponibles 