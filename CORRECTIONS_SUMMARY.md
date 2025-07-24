# Correcciones para Sincronizar Python y MQL5

## Problemas Identificados y Solucionados

### 1. **Error Crítico en evaluate_report (Python)**
**Problema**: El número de trades se calculaba incorrectamente usando `np.diff(equity_curve).size` en lugar del tamaño real del array `trade_profits`.

**Impacto**: Esto causaba discrepancias significativas en:
- Número total de trades reportados
- Cálculo de la métrica `trade_nl` (normalización de número de trades)
- Score final de evaluación

**Solución Aplicada**:
```python
# ANTES (INCORRECTO):
returns = np.diff(equity_curve)
n_trades = returns.size

# DESPUÉS (CORREGIDO):
n_trades = trade_profits.size
```

### 2. **Inconsistencias en OnTester (MQL5)**

#### 2.1 Cálculo de Regresión Lineal
**Problema**: El método de cálculo no coincidía exactamente con la implementación manual de Python.

**Solución**: Implementar la misma lógica paso a paso que usa Python:
```mql5
// Calcular medias
for(int i = 0; i < N; i++) {
   x_mean += i;
   y_mean += equity[i];
}
x_mean /= N;
y_mean /= N;

// Calcular pendiente usando método exacto de Python
double numerator = 0.0;
double denominator = 0.0;
for(int i = 0; i < N; i++) {
   double x_diff = i - x_mean;
   double y_diff = equity[i] - y_mean;
   numerator += x_diff * y_diff;
   denominator += x_diff * x_diff;
}
```

#### 2.2 Cálculo de Drawdown Máximo
**Problema**: La lógica de `running_max` no coincidía con Python.

**Solución**: Implementar el mismo algoritmo iterativo:
```mql5
running_max[0] = equity[0];
for(int i = 1; i < n_trades + 1; i++) {
   running_max[i] = (running_max[i-1] > equity[i]) ? running_max[i-1] : equity[i];
}
```

#### 2.3 Walk-Forward Analysis
**Problema**: La implementación no consideraba correctamente las ventanas deslizantes y el cálculo del ratio de trades ganadores.

**Solución**: Replicar exactamente la lógica de Python con ventanas de 5 trades y paso de 1.

### 3. **Lógica de Apertura de Posiciones**

#### 3.1 Timing de last_trade_bar
**Problema**: En ambas plataformas, cuando se abrían múltiples posiciones en la misma barra, `last_trade_bar` se actualizaba múltiples veces.

**Solución**:
```python
# Python
trade_opened_this_bar = False
if buy_sig and (max_orders == 0 or n_open < max_orders):
    # ... abrir posición BUY ...
    trade_opened_this_bar = True
if sell_sig and (max_orders == 0 or n_open < max_orders):
    # ... abrir posición SELL ...
    trade_opened_this_bar = True

if trade_opened_this_bar:
    last_trade_bar = bar
```

```mql5
// MQL5
bool trade_opened_this_bar = false;
if(buy_sig && (max_orders == 0 || live_pos < max_orders)) {
    // ... abrir posición BUY ...
    trade_opened_this_bar = true;
}
if(sell_sig && (max_orders == 0 || live_pos < max_orders)) {
    // ... abrir posición SELL ...
    trade_opened_this_bar = true;
}
if(trade_opened_this_bar)
    last_trade_bar_index = bar_counter;
```

### 4. **Validaciones de Consistencia**

#### 4.1 Mínimo de Trades
**Ambas plataformas**: Asegurar que se requieren exactamente 200 trades mínimos.

#### 4.2 Pesos del Score Final
**Verificado**: Ambas plataformas usan los mismos pesos:
```
score = 0.12 * r2 + 0.15 * slope_nl + 0.24 * rdd_nl + 0.19 * trade_nl + 0.30 * wf_nl
```

## Archivos Modificados

### Python
- `studies/modules/tester_lib.py`: 
  - Función `evaluate_report`: Corregido cálculo de `n_trades`
  - Función `backtest`: Corregido timing de `last_trade_bar`

### MQL5
- `mql5_corrections.mq5`: Archivo nuevo con funciones corregidas
  - `OnTester()`: Completamente reescrito para coincidir con Python
  - `OnTick()`: Corregida lógica de apertura de posiciones

## Resultados Esperados

Con estas correcciones, ambas implementaciones deberían producir:

1. **Idéntico número de trades**
2. **Idénticas curvas de equity**
3. **Idénticos ratios de trades ganadores/perdedores**
4. **Idénticos scores finales de evaluación**
5. **Idénticas métricas intermedias** (R², slope_nl, rdd_nl, trade_nl, wf_nl)

## Verificación

Para verificar que las correcciones funcionan:

1. Ejecutar la misma estrategia en ambas plataformas con parámetros idénticos
2. Comparar las métricas reportadas
3. Verificar que las diferencias sean menores a 0.001% (tolerancia por precisión numérica)

## Notas Importantes

- Las correcciones mantienen la lógica de negocio intacta
- Solo se corrigieron discrepancias de implementación
- Ambas plataformas siguen la misma secuencia de decisiones de trading
- La performance debería ser idéntica en backtesting controlado