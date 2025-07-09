"""
Sistema de simulación de curvas de equity para testear y optimizar evaluate_report.

Este módulo genera miles de simulaciones de diferentes tipos de curvas de equity
para validar que la función evaluate_report optimizada encuentra efectivamente
las estrategias con curvas perfectamente lineales.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Importar la función optimizada
from tester_lib import evaluate_report, metrics_tuple_to_dict, print_detailed_metrics


class EquitySimulator:
    """
    Simulador de curvas de equity para testing y benchmarking.
    """
    
    def __init__(self, periods_per_year: float = 6240.0):
        self.periods_per_year = periods_per_year
        self.simulation_results = []
        
    def generate_perfect_linear(self, length: int = 1000, slope: float = 0.5, noise: float = 0.0) -> np.ndarray:
        """Genera una curva perfectamente lineal con crecimiento constante."""
        t = np.arange(length, dtype=np.float64)
        equity = t * slope + 100.0  # Empezar en 100
        
        if noise > 0:
            # Agregar ruido gaussiano controlado
            noise_component = np.random.normal(0, noise, length)
            equity += noise_component
            
        return equity
    
    def generate_exponential_growth(self, length: int = 1000, growth_rate: float = 0.001) -> np.ndarray:
        """Genera crecimiento exponencial."""
        t = np.arange(length, dtype=np.float64)
        equity = 100.0 * np.exp(growth_rate * t)
        return equity
    
    def generate_volatile_growth(self, length: int = 1000, base_slope: float = 0.3, volatility: float = 0.5) -> np.ndarray:
        """Genera crecimiento con alta volatilidad."""
        t = np.arange(length, dtype=np.float64)
        base = t * base_slope + 100.0
        
        # Añadir volatilidad variable
        volatility_component = np.cumsum(np.random.normal(0, volatility, length))
        equity = base + volatility_component
        
        return equity
    
    def generate_drawdown_curve(self, length: int = 1000, max_drawdown: float = 0.2) -> np.ndarray:
        """Genera curva con drawdowns significativos."""
        t = np.arange(length, dtype=np.float64)
        base = t * 0.4 + 100.0
        
        # Añadir drawdowns periódicos
        drawdown_points = np.random.choice(length, size=int(length * 0.1), replace=False)
        
        equity = base.copy()
        for point in drawdown_points:
            # Crear drawdown local
            dd_length = min(50, length - point)
            dd_magnitude = max_drawdown * np.random.uniform(0.5, 1.0)
            
            for i in range(dd_length):
                if point + i < length:
                    equity[point + i:] -= base[point] * dd_magnitude * np.exp(-i/20.0)
                    
        return equity
    
    def generate_sideways_trending(self, length: int = 1000, trend_strength: float = 0.1) -> np.ndarray:
        """Genera curva que se mueve lateralmente con poco crecimiento."""
        base_value = 100.0
        equity = np.full(length, base_value)
        
        # Añadir movimiento aleatorio con muy poco trend
        for i in range(1, length):
            change = np.random.normal(trend_strength, 2.0)
            equity[i] = equity[i-1] + change
            
        return equity
    
    def generate_negative_slope(self, length: int = 1000, decline_rate: float = -0.2) -> np.ndarray:
        """Genera curva con pendiente negativa."""
        t = np.arange(length, dtype=np.float64)
        equity = t * decline_rate + 200.0  # Empezar más alto para evitar negativos
        return np.maximum(equity, 1.0)  # Evitar valores negativos
    
    def generate_step_function(self, length: int = 1000, steps: int = 5) -> np.ndarray:
        """Genera crecimiento en escalones (no suave)."""
        equity = np.ones(length) * 100.0
        step_size = length // steps
        
        for i in range(steps):
            start_idx = i * step_size
            end_idx = min((i + 1) * step_size, length)
            growth = 20.0 * (i + 1)  # Crecimiento por escalón
            equity[start_idx:end_idx] += growth
            
        return equity
    
    def generate_mixed_performance(self, length: int = 1000) -> np.ndarray:
        """Genera performance mixta: períodos buenos y malos."""
        equity = np.ones(length) * 100.0
        
        # Dividir en segmentos
        segment_length = length // 6
        
        for i in range(0, length, segment_length):
            end_idx = min(i + segment_length, length)
            segment_length_actual = end_idx - i
            
            if i % (segment_length * 2) == 0:  # Segmentos buenos
                t = np.arange(segment_length_actual)
                growth = t * 0.8 + equity[i]
                equity[i:end_idx] = growth
            else:  # Segmentos malos
                t = np.arange(segment_length_actual)
                decline = t * (-0.3) + equity[i]
                equity[i:end_idx] = decline
                
        return equity
    
    def run_single_simulation(self, curve_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una simulación individual y retorna métricas."""
        
        # Generar curva según el tipo
        if curve_type == 'perfect_linear':
            equity = self.generate_perfect_linear(**params)
        elif curve_type == 'exponential':
            equity = self.generate_exponential_growth(**params)
        elif curve_type == 'volatile':
            equity = self.generate_volatile_growth(**params)
        elif curve_type == 'drawdown':
            equity = self.generate_drawdown_curve(**params)
        elif curve_type == 'sideways':
            equity = self.generate_sideways_trending(**params)
        elif curve_type == 'negative':
            equity = self.generate_negative_slope(**params)
        elif curve_type == 'step':
            equity = self.generate_step_function(**params)
        elif curve_type == 'mixed':
            equity = self.generate_mixed_performance(**params)
        else:
            raise ValueError(f"Tipo de curva no reconocido: {curve_type}")
        
        # Medir tiempo de ejecución
        start_time = time.perf_counter()
        score, metrics_tuple = evaluate_report(equity, self.periods_per_year)
        execution_time = time.perf_counter() - start_time
        
        # Convertir métricas a diccionario
        metrics_dict = metrics_tuple_to_dict(score, metrics_tuple, self.periods_per_year)
        
        # Agregar metadata
        result = {
            'curve_type': curve_type,
            'params': params,
            'equity_curve': equity,
            'execution_time': execution_time,
            **metrics_dict
        }
        
        return result
    
    def run_massive_simulation(self, simulations_per_type: int = 1000) -> pd.DataFrame:
        """
        Ejecuta miles de simulaciones para diferentes tipos de curvas.
        """
        print(f"🚀 Iniciando simulación masiva: {simulations_per_type} simulaciones por tipo...")
        
        # Definir tipos de curvas y sus parámetros
        simulation_configs = [
            # Curvas lineales perfectas (diferentes pendientes)
            ('perfect_linear', {'slope': 0.1, 'noise': 0.0}),
            ('perfect_linear', {'slope': 0.3, 'noise': 0.0}),
            ('perfect_linear', {'slope': 0.5, 'noise': 0.0}),
            ('perfect_linear', {'slope': 0.8, 'noise': 0.0}),
            ('perfect_linear', {'slope': 1.0, 'noise': 0.0}),
            ('perfect_linear', {'slope': 1.5, 'noise': 0.0}),
            
            # Curvas lineales con ruido ligero
            ('perfect_linear', {'slope': 0.5, 'noise': 0.1}),
            ('perfect_linear', {'slope': 0.5, 'noise': 0.2}),
            ('perfect_linear', {'slope': 0.8, 'noise': 0.1}),
            
            # Curvas exponenciales
            ('exponential', {'growth_rate': 0.0005}),
            ('exponential', {'growth_rate': 0.001}),
            ('exponential', {'growth_rate': 0.002}),
            
            # Curvas volátiles
            ('volatile', {'base_slope': 0.3, 'volatility': 0.5}),
            ('volatile', {'base_slope': 0.5, 'volatility': 1.0}),
            ('volatile', {'base_slope': 0.8, 'volatility': 1.5}),
            
            # Curvas con drawdowns
            ('drawdown', {'max_drawdown': 0.1}),
            ('drawdown', {'max_drawdown': 0.2}),
            ('drawdown', {'max_drawdown': 0.3}),
            
            # Curvas laterales
            ('sideways', {'trend_strength': 0.05}),
            ('sideways', {'trend_strength': 0.1}),
            
            # Curvas negativas
            ('negative', {'decline_rate': -0.1}),
            ('negative', {'decline_rate': -0.3}),
            
            # Curvas en escalones
            ('step', {'steps': 3}),
            ('step', {'steps': 5}),
            ('step', {'steps': 10}),
            
            # Performance mixta
            ('mixed', {}),
        ]
        
        # Preparar todas las simulaciones
        all_simulations = []
        for curve_type, base_params in simulation_configs:
            for i in range(simulations_per_type):
                # Añadir variación en la longitud de la serie
                params = base_params.copy()
                params['length'] = np.random.randint(200, 2000)
                all_simulations.append((curve_type, params))
        
        print(f"📊 Total de simulaciones: {len(all_simulations)}")
        
        # Ejecutar simulaciones en paralelo para mayor velocidad
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Enviar todas las tareas
            future_to_config = {
                executor.submit(self.run_single_simulation, curve_type, params): (curve_type, params)
                for curve_type, params in all_simulations
            }
            
            # Recopilar resultados
            for i, future in enumerate(as_completed(future_to_config)):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Progreso cada 1000 simulaciones
                    if (i + 1) % 1000 == 0:
                        elapsed = time.time() - start_time
                        rate = (i + 1) / elapsed
                        eta = (len(all_simulations) - i - 1) / rate
                        print(f"  ✅ Completadas {i + 1}/{len(all_simulations)} simulaciones "
                              f"({rate:.1f}/seg, ETA: {eta:.1f}s)")
                        
                except Exception as e:
                    print(f"❌ Error en simulación: {e}")
                    continue
        
        total_time = time.time() - start_time
        print(f"🎉 Simulación completada en {total_time:.2f} segundos")
        print(f"⚡ Velocidad promedio: {len(results)/total_time:.1f} simulaciones/segundo")
        
        # Convertir a DataFrame
        df = pd.DataFrame(results)
        
        return df
    
    def analyze_results(self, df: pd.DataFrame) -> None:
        """
        Analiza los resultados de las simulaciones y genera reportes.
        """
        print("\n" + "="*80)
        print("📈 ANÁLISIS DE RESULTADOS DE SIMULACIÓN")
        print("="*80)
        
        # 1. Verificar tiempos de ejecución
        print(f"\n⏱️  PERFORMANCE DE EJECUCIÓN:")
        print(f"   • Tiempo promedio: {df['execution_time'].mean()*1000:.2f}ms")
        print(f"   • Tiempo máximo: {df['execution_time'].max()*1000:.2f}ms")
        print(f"   • 95th percentil: {df['execution_time'].quantile(0.95)*1000:.2f}ms")
        
        if df['execution_time'].max() > 1.0:
            print("   ⚠️  ADVERTENCIA: Algunas ejecuciones superan 1 segundo!")
        else:
            print("   ✅ OBJETIVO CUMPLIDO: Todas las ejecuciones < 1 segundo")
        
        # 2. Análisis por tipo de curva
        print(f"\n📊 SCORES POR TIPO DE CURVA:")
        score_by_type = df.groupby('curve_type')['score'].agg(['mean', 'std', 'max', 'count'])
        score_by_type = score_by_type.sort_values('mean', ascending=False)
        
        for curve_type, stats in score_by_type.iterrows():
            print(f"   • {curve_type:15}: Score={stats['mean']:.4f}±{stats['std']:.4f} "
                  f"(max={stats['max']:.4f}, n={stats['count']})")
        
        # 3. Verificar que las curvas lineales perfectas obtienen los mejores scores
        perfect_linear = df[df['curve_type'] == 'perfect_linear']
        other_curves = df[df['curve_type'] != 'perfect_linear']
        
        print(f"\n🎯 EFECTIVIDAD PARA DETECTAR LINEALIDAD:")
        print(f"   • Score promedio curvas lineales: {perfect_linear['score'].mean():.4f}")
        print(f"   • Score promedio otras curvas: {other_curves['score'].mean():.4f}")
        print(f"   • Diferencia: {perfect_linear['score'].mean() - other_curves['score'].mean():.4f}")
        
        # Top 10 mejores scores
        top_10 = df.nlargest(10, 'score')[['curve_type', 'score', 'r2', 'perfect_linearity', 'slope_reward']]
        print(f"\n🏆 TOP 10 MEJORES SCORES:")
        for i, (_, row) in enumerate(top_10.iterrows()):
            print(f"   {i+1:2d}. {row['curve_type']:15} - Score: {row['score']:.4f} "
                  f"(R²={row['r2']:.3f}, PerfLin={row['perfect_linearity']:.3f})")
        
        # 4. Análisis de métricas específicas
        print(f"\n🔍 MÉTRICAS ESPECÍFICAS PARA CURVAS LINEALES PERFECTAS:")
        linear_subset = perfect_linear[perfect_linear['score'] > 0.8]  # Solo las mejores
        
        if len(linear_subset) > 0:
            print(f"   • R² promedio: {linear_subset['r2'].mean():.4f}")
            print(f"   • Perfect Linearity: {linear_subset['perfect_linearity'].mean():.4f}")
            print(f"   • Monotonic Growth: {linear_subset['monotonic_growth'].mean():.4f}")
            print(f"   • Smoothness: {linear_subset['smoothness'].mean():.4f}")
            print(f"   • Slope Reward: {linear_subset['slope_reward'].mean():.4f}")
        
        # 5. Detectar posibles problemas
        print(f"\n⚠️  DETECCIÓN DE PROBLEMAS:")
        
        # Curvas lineales con score bajo
        poor_linear = perfect_linear[perfect_linear['score'] < 0.5]
        if len(poor_linear) > 0:
            print(f"   • {len(poor_linear)} curvas lineales con score < 0.5")
        
        # Curvas no lineales con score alto
        high_nonlinear = other_curves[other_curves['score'] > 0.8]
        if len(high_nonlinear) > 0:
            print(f"   • {len(high_nonlinear)} curvas no lineales con score > 0.8")
            print("     Tipos:", high_nonlinear['curve_type'].value_counts().to_dict())
        
        if len(poor_linear) == 0 and len(high_nonlinear) == 0:
            print("   ✅ No se detectaron problemas mayores en la discriminación")
    
    def plot_results(self, df: pd.DataFrame, save_plots: bool = True) -> None:
        """
        Genera visualizaciones de los resultados.
        """
        print("\n📊 Generando visualizaciones...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Análisis de Simulación de Curvas de Equity', fontsize=16, fontweight='bold')
        
        # 1. Distribución de scores por tipo
        ax1 = axes[0, 0]
        df.boxplot(column='score', by='curve_type', ax=ax1)
        ax1.set_title('Distribución de Scores por Tipo de Curva')
        ax1.set_xlabel('Tipo de Curva')
        ax1.set_ylabel('Score')
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # 2. Tiempos de ejecución
        ax2 = axes[0, 1]
        ax2.hist(df['execution_time'] * 1000, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(1000, color='red', linestyle='--', label='1 segundo límite')
        ax2.set_title('Distribución de Tiempos de Ejecución')
        ax2.set_xlabel('Tiempo (ms)')
        ax2.set_ylabel('Frecuencia')
        ax2.legend()
        
        # 3. Score vs R²
        ax3 = axes[0, 2]
        scatter = ax3.scatter(df['r2'], df['score'], c=df['curve_type'].astype('category').cat.codes, 
                             alpha=0.6, s=10)
        ax3.set_title('Score vs R²')
        ax3.set_xlabel('R²')
        ax3.set_ylabel('Score')
        
        # 4. Score vs Perfect Linearity
        ax4 = axes[1, 0]
        ax4.scatter(df['perfect_linearity'], df['score'], 
                   c=df['curve_type'].astype('category').cat.codes, alpha=0.6, s=10)
        ax4.set_title('Score vs Perfect Linearity')
        ax4.set_xlabel('Perfect Linearity')
        ax4.set_ylabel('Score')
        
        # 5. Métricas de curvas lineales perfectas
        ax5 = axes[1, 1]
        linear_data = df[df['curve_type'] == 'perfect_linear']
        metrics_to_plot = ['r2', 'perfect_linearity', 'monotonic_growth', 'smoothness', 'slope_reward']
        linear_metrics = linear_data[metrics_to_plot].mean()
        
        bars = ax5.bar(range(len(metrics_to_plot)), linear_metrics.values)
        ax5.set_title('Métricas Promedio - Curvas Lineales Perfectas')
        ax5.set_xticks(range(len(metrics_to_plot)))
        ax5.set_xticklabels(metrics_to_plot, rotation=45)
        ax5.set_ylabel('Valor Promedio')
        
        # Añadir valores en las barras
        for bar, value in zip(bars, linear_metrics.values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 6. Score promedio por pendiente (para curvas lineales)
        ax6 = axes[1, 2]
        linear_data_clean = df[(df['curve_type'] == 'perfect_linear') & 
                               (df['params'].apply(lambda x: 'slope' in x))]
        
        if len(linear_data_clean) > 0:
            # Extraer pendientes
            slopes = [params['slope'] for params in linear_data_clean['params']]
            linear_data_clean = linear_data_clean.copy()
            linear_data_clean['slope'] = slopes
            
            slope_scores = linear_data_clean.groupby('slope')['score'].mean().sort_index()
            
            ax6.plot(slope_scores.index, slope_scores.values, 'o-', linewidth=2, markersize=8)
            ax6.set_title('Score vs Pendiente (Curvas Lineales)')
            ax6.set_xlabel('Pendiente')
            ax6.set_ylabel('Score Promedio')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('studies/modules/simulation_results.png', dpi=300, bbox_inches='tight')
            print("📁 Gráficos guardados en: studies/modules/simulation_results.png")
        
        plt.show()
    
    def generate_sample_curves(self, save_examples: bool = True) -> None:
        """
        Genera y visualiza curvas de ejemplo de cada tipo.
        """
        print("\n🎨 Generando curvas de ejemplo...")
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Ejemplos de Curvas de Equity Simuladas', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        curve_examples = [
            ('Lineal Perfecta (pendiente 0.5)', 'perfect_linear', {'slope': 0.5, 'noise': 0.0}),
            ('Lineal con Ruido', 'perfect_linear', {'slope': 0.5, 'noise': 0.2}),
            ('Exponencial', 'exponential', {'growth_rate': 0.001}),
            ('Volátil', 'volatile', {'base_slope': 0.5, 'volatility': 1.0}),
            ('Con Drawdowns', 'drawdown', {'max_drawdown': 0.2}),
            ('Lateral', 'sideways', {'trend_strength': 0.1}),
            ('Pendiente Negativa', 'negative', {'decline_rate': -0.2}),
            ('En Escalones', 'step', {'steps': 5}),
            ('Performance Mixta', 'mixed', {})
        ]
        
        for i, (title, curve_type, params) in enumerate(curve_examples):
            ax = axes[i]
            
            # Generar curva
            params['length'] = 1000
            result = self.run_single_simulation(curve_type, params)
            equity = result['equity_curve']
            score = result['score']
            
            # Plotear
            ax.plot(equity, linewidth=1.5)
            ax.set_title(f'{title}\nScore: {score:.4f}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Tiempo')
            ax.set_ylabel('Equity')
        
        plt.tight_layout()
        
        if save_examples:
            plt.savefig('studies/modules/example_curves.png', dpi=300, bbox_inches='tight')
            print("📁 Ejemplos guardados en: studies/modules/example_curves.png")
        
        plt.show()


def main():
    """
    Función principal para ejecutar la simulación completa.
    """
    print("🚀 INICIANDO SISTEMA DE SIMULACIÓN Y BENCHMARKING")
    print("="*60)
    
    # Crear simulador
    simulator = EquitySimulator()
    
    # Generar ejemplos de curvas
    simulator.generate_sample_curves()
    
    # Ejecutar simulación masiva (reducido para demo, aumentar en producción)
    print("\n" + "="*60)
    print("🎯 EJECUTANDO SIMULACIÓN MASIVA")
    print("="*60)
    
    # Para demo rápida: 100 simulaciones por tipo
    # Para benchmarking completo: aumentar a 1000+
    df_results = simulator.run_massive_simulation(simulations_per_type=100)
    
    # Analizar resultados
    simulator.analyze_results(df_results)
    
    # Generar visualizaciones
    simulator.plot_results(df_results)
    
    # Guardar resultados detallados
    df_results.to_csv('studies/modules/simulation_results.csv', index=False)
    print(f"\n💾 Resultados guardados en: studies/modules/simulation_results.csv")
    
    print("\n🎉 ¡SIMULACIÓN COMPLETADA CON ÉXITO!")
    print("="*60)


if __name__ == "__main__":
    main()