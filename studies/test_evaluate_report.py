import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Importar el módulo a testear
import sys
sys.path.append('/workspace/studies/modules')
from tester_lib import evaluate_report, metrics_tuple_to_dict

class EquityCurveGenerator:
    """Generador de curvas de equity controladas para testing"""
    
    def __init__(self, base_length: int = 1000):
        self.base_length = base_length
        
    def generate_perfect_linear(self, slope: float = 0.5, noise: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Genera una curva perfectamente lineal con ruido opcional"""
        t = np.arange(self.base_length, dtype=np.float64)
        equity = t * slope + 100.0
        
        if noise > 0:
            equity += np.random.normal(0, noise, size=self.base_length)
            
        # Generar trade_stats realistas
        n_trades = int(self.base_length * 0.05)  # 5% de actividad
        win_rate = 0.85 if noise < 0.1 else 0.7
        positive_trades = int(n_trades * win_rate)
        negative_trades = n_trades - positive_trades
        
        trade_stats = np.array([
            n_trades, positive_trades, negative_trades, 0,
            win_rate, slope/n_trades*positive_trades, -slope/n_trades*negative_trades*0.5
        ], dtype=np.float64)
        
        return equity, trade_stats
    
    def generate_exponential(self, rate: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
        """Genera una curva exponencial"""
        t = np.arange(self.base_length, dtype=np.float64)
        equity = 100.0 * np.exp(rate * t)
        
        n_trades = int(self.base_length * 0.04)
        win_rate = 0.75
        positive_trades = int(n_trades * win_rate)
        negative_trades = n_trades - positive_trades
        
        trade_stats = np.array([
            n_trades, positive_trades, negative_trades, 0,
            win_rate, 0.02, -0.01
        ], dtype=np.float64)
        
        return equity, trade_stats
    
    def generate_stepped(self, step_size: float = 10.0, step_freq: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Genera una curva escalonada"""
        equity = np.zeros(self.base_length)
        current_value = 100.0
        
        for i in range(self.base_length):
            if i > 0 and i % step_freq == 0:
                current_value += step_size
            equity[i] = current_value
            
        n_trades = self.base_length // step_freq
        win_rate = 1.0 if step_size > 0 else 0.0
        positive_trades = n_trades if step_size > 0 else 0
        negative_trades = 0 if step_size > 0 else n_trades
        
        trade_stats = np.array([
            n_trades, positive_trades, negative_trades, 0,
            win_rate, step_size, 0.0
        ], dtype=np.float64)
        
        return equity, trade_stats
    
    def generate_volatile(self, trend: float = 0.1, volatility: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """Genera una curva volátil con tendencia"""
        returns = np.random.normal(trend, volatility, self.base_length)
        equity = 100.0 + np.cumsum(returns)
        
        n_trades = int(self.base_length * 0.1)  # Alta actividad
        win_rate = 0.55
        positive_trades = int(n_trades * win_rate)
        negative_trades = n_trades - positive_trades
        
        trade_stats = np.array([
            n_trades, positive_trades, negative_trades, 0,
            win_rate, volatility, -volatility*0.9
        ], dtype=np.float64)
        
        return equity, trade_stats
    
    def generate_drawdown_recovery(self, dd_depth: float = 0.3, dd_location: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Genera una curva con drawdown y recuperación"""
        t = np.arange(self.base_length, dtype=np.float64)
        equity = t * 0.5 + 100.0
        
        # Añadir drawdown
        dd_start = int(self.base_length * dd_location)
        dd_end = dd_start + int(self.base_length * 0.2)
        dd_bottom = dd_start + int(self.base_length * 0.1)
        
        peak_value = equity[dd_start]
        bottom_value = peak_value * (1 - dd_depth)
        
        # Crear drawdown
        for i in range(dd_start, dd_bottom):
            progress = (i - dd_start) / (dd_bottom - dd_start)
            equity[i] = peak_value - (peak_value - bottom_value) * progress
            
        # Recuperación
        for i in range(dd_bottom, dd_end):
            progress = (i - dd_bottom) / (dd_end - dd_bottom)
            equity[i] = bottom_value + (peak_value - bottom_value) * progress
            
        # Continuar tendencia
        for i in range(dd_end, self.base_length):
            equity[i] = equity[dd_end-1] + (i - dd_end + 1) * 0.5
            
        n_trades = int(self.base_length * 0.06)
        win_rate = 0.65
        positive_trades = int(n_trades * win_rate)
        negative_trades = n_trades - positive_trades
        
        trade_stats = np.array([
            n_trades, positive_trades, negative_trades, 0,
            win_rate, 0.015, -0.02
        ], dtype=np.float64)
        
        return equity, trade_stats
    
    def generate_sawtooth(self, period: int = 100, amplitude: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
        """Genera una curva en diente de sierra"""
        equity = np.zeros(self.base_length)
        base_value = 100.0
        
        for i in range(self.base_length):
            cycle_pos = i % period
            equity[i] = base_value + (cycle_pos / period) * amplitude + (i // period) * amplitude * 0.5
            
        n_trades = self.base_length // (period // 2)
        win_rate = 0.5
        positive_trades = int(n_trades * win_rate)
        negative_trades = n_trades - positive_trades
        
        trade_stats = np.array([
            n_trades, positive_trades, negative_trades, 0,
            win_rate, amplitude/period, -amplitude/period
        ], dtype=np.float64)
        
        return equity, trade_stats
    
    def generate_random_walk(self, drift: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Genera un random walk"""
        steps = np.random.choice([-1, 1], size=self.base_length) + drift
        equity = 100.0 + np.cumsum(steps)
        
        n_trades = int(self.base_length * 0.08)
        win_rate = 0.5 + drift * 0.1
        positive_trades = int(n_trades * win_rate)
        negative_trades = n_trades - positive_trades
        
        trade_stats = np.array([
            n_trades, positive_trades, negative_trades, 0,
            win_rate, 1.0, -1.0
        ], dtype=np.float64)
        
        return equity, trade_stats
    
    def generate_flat(self, value: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """Genera una curva plana"""
        equity = np.full(self.base_length, value)
        
        trade_stats = np.array([
            0, 0, 0, 0, 0.0, 0.0, 0.0
        ], dtype=np.float64)
        
        return equity, trade_stats
    
    def generate_sine_trend(self, trend: float = 0.1, amplitude: float = 10.0, frequency: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Genera una curva sinusoidal con tendencia"""
        t = np.arange(self.base_length, dtype=np.float64)
        equity = 100.0 + t * trend + amplitude * np.sin(2 * np.pi * frequency * t)
        
        n_trades = int(self.base_length * frequency * 2)  # 2 trades por ciclo
        win_rate = 0.6 if trend > 0 else 0.4
        positive_trades = int(n_trades * win_rate)
        negative_trades = n_trades - positive_trades
        
        trade_stats = np.array([
            n_trades, positive_trades, negative_trades, 0,
            win_rate, amplitude/10, -amplitude/10
        ], dtype=np.float64)
        
        return equity, trade_stats
    
    def generate_compound_curve(self, components: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Genera una curva compuesta de múltiples componentes"""
        equity = np.zeros(self.base_length)
        
        for comp in components:
            if comp['type'] == 'linear':
                equity += np.arange(self.base_length) * comp.get('slope', 0.1)
            elif comp['type'] == 'sine':
                t = np.arange(self.base_length)
                equity += comp.get('amplitude', 5.0) * np.sin(2 * np.pi * comp.get('frequency', 0.01) * t)
            elif comp['type'] == 'noise':
                equity += np.random.normal(0, comp.get('std', 1.0), self.base_length)
                
        equity += 100.0  # Base value
        
        # Trade stats aproximadas
        n_trades = int(self.base_length * 0.07)
        win_rate = 0.6
        positive_trades = int(n_trades * win_rate)
        negative_trades = n_trades - positive_trades
        
        trade_stats = np.array([
            n_trades, positive_trades, negative_trades, 0,
            win_rate, 0.02, -0.015
        ], dtype=np.float64)
        
        return equity, trade_stats


class EvaluateReportTester:
    """Clase principal para testear la función evaluate_report"""
    
    def __init__(self):
        self.generator = EquityCurveGenerator()
        self.test_results = []
        
    def run_single_test(self, curve_type: str, params: Dict[str, Any], expected_score_range: Tuple[float, float]) -> Dict[str, Any]:
        """Ejecuta un test individual"""
        # Generar curva
        if curve_type == 'perfect_linear':
            equity, trade_stats = self.generator.generate_perfect_linear(**params)
        elif curve_type == 'exponential':
            equity, trade_stats = self.generator.generate_exponential(**params)
        elif curve_type == 'stepped':
            equity, trade_stats = self.generator.generate_stepped(**params)
        elif curve_type == 'volatile':
            equity, trade_stats = self.generator.generate_volatile(**params)
        elif curve_type == 'drawdown_recovery':
            equity, trade_stats = self.generator.generate_drawdown_recovery(**params)
        elif curve_type == 'sawtooth':
            equity, trade_stats = self.generator.generate_sawtooth(**params)
        elif curve_type == 'random_walk':
            equity, trade_stats = self.generator.generate_random_walk(**params)
        elif curve_type == 'flat':
            equity, trade_stats = self.generator.generate_flat(**params)
        elif curve_type == 'sine_trend':
            equity, trade_stats = self.generator.generate_sine_trend(**params)
        elif curve_type == 'compound':
            equity, trade_stats = self.generator.generate_compound_curve(**params)
        else:
            raise ValueError(f"Unknown curve type: {curve_type}")
            
        # Evaluar
        metrics_tuple = evaluate_report(equity, trade_stats)
        metrics_dict = metrics_tuple_to_dict(metrics_tuple)
        score = metrics_dict['final_score']
        
        # Verificar si está en el rango esperado
        in_range = expected_score_range[0] <= score <= expected_score_range[1]
        
        result = {
            'curve_type': curve_type,
            'params': params,
            'expected_range': expected_score_range,
            'actual_score': score,
            'in_range': in_range,
            'metrics': metrics_dict,
            'equity': equity,
            'trade_stats': trade_stats
        }
        
        self.test_results.append(result)
        return result
    
    def run_comprehensive_tests(self) -> pd.DataFrame:
        """Ejecuta una batería completa de tests"""
        test_cases = [
            # Curvas perfectamente lineales (deberían tener score muy alto)
            ('perfect_linear', {'slope': 0.1, 'noise': 0.0}, (0.85, 1.0)),
            ('perfect_linear', {'slope': 0.5, 'noise': 0.0}, (0.90, 1.0)),
            ('perfect_linear', {'slope': 1.0, 'noise': 0.0}, (0.90, 1.0)),
            ('perfect_linear', {'slope': 2.0, 'noise': 0.0}, (0.85, 1.0)),
            
            # Curvas lineales con ruido (score alto pero menor)
            ('perfect_linear', {'slope': 0.5, 'noise': 0.5}, (0.70, 0.90)),
            ('perfect_linear', {'slope': 0.5, 'noise': 1.0}, (0.60, 0.85)),
            ('perfect_linear', {'slope': 0.5, 'noise': 2.0}, (0.50, 0.75)),
            
            # Curvas exponenciales (score medio-alto)
            ('exponential', {'rate': 0.0005}, (0.60, 0.85)),
            ('exponential', {'rate': 0.001}, (0.55, 0.80)),
            ('exponential', {'rate': 0.002}, (0.50, 0.75)),
            
            # Curvas escalonadas (score medio)
            ('stepped', {'step_size': 5.0, 'step_freq': 50}, (0.50, 0.75)),
            ('stepped', {'step_size': 10.0, 'step_freq': 100}, (0.45, 0.70)),
            ('stepped', {'step_size': -5.0, 'step_freq': 50}, (0.0, 0.20)),
            
            # Curvas volátiles (score bajo)
            ('volatile', {'trend': 0.1, 'volatility': 5.0}, (0.20, 0.50)),
            ('volatile', {'trend': 0.0, 'volatility': 10.0}, (0.10, 0.40)),
            ('volatile', {'trend': -0.1, 'volatility': 5.0}, (0.0, 0.20)),
            
            # Curvas con drawdown (score medio-bajo)
            ('drawdown_recovery', {'dd_depth': 0.1, 'dd_location': 0.5}, (0.60, 0.85)),
            ('drawdown_recovery', {'dd_depth': 0.3, 'dd_location': 0.5}, (0.40, 0.65)),
            ('drawdown_recovery', {'dd_depth': 0.5, 'dd_location': 0.3}, (0.20, 0.45)),
            
            # Curvas en diente de sierra (score bajo-medio)
            ('sawtooth', {'period': 100, 'amplitude': 20.0}, (0.30, 0.60)),
            ('sawtooth', {'period': 50, 'amplitude': 10.0}, (0.35, 0.65)),
            
            # Random walks (score muy bajo)
            ('random_walk', {'drift': 0.0}, (0.0, 0.30)),
            ('random_walk', {'drift': 0.01}, (0.10, 0.40)),
            ('random_walk', {'drift': -0.01}, (0.0, 0.20)),
            
            # Curvas planas (score muy bajo)
            ('flat', {'value': 100.0}, (0.0, 0.10)),
            
            # Curvas sinusoidales con tendencia (score medio)
            ('sine_trend', {'trend': 0.1, 'amplitude': 10.0, 'frequency': 0.05}, (0.40, 0.70)),
            ('sine_trend', {'trend': 0.2, 'amplitude': 5.0, 'frequency': 0.1}, (0.50, 0.80)),
            
            # Curvas compuestas
            ('compound', {'components': [
                {'type': 'linear', 'slope': 0.3},
                {'type': 'sine', 'amplitude': 5.0, 'frequency': 0.02},
                {'type': 'noise', 'std': 0.5}
            ]}, (0.65, 0.85)),
        ]
        
        print(f"Ejecutando {len(test_cases)} casos de prueba...\n")
        
        for curve_type, params, expected_range in test_cases:
            result = self.run_single_test(curve_type, params, expected_range)
            status = "✓" if result['in_range'] else "✗"
            print(f"{status} {curve_type} {params} - Score: {result['actual_score']:.4f} (esperado: {expected_range})")
            
        # Crear DataFrame con resultados
        df_results = pd.DataFrame([
            {
                'curve_type': r['curve_type'],
                'params': str(r['params']),
                'expected_min': r['expected_range'][0],
                'expected_max': r['expected_range'][1],
                'actual_score': r['actual_score'],
                'in_range': r['in_range'],
                'r2': r['metrics']['r2'],
                'perfect_linearity': r['metrics']['perfect_linearity'],
                'consistency': r['metrics']['consistency'],
                'trade_activity': r['metrics']['trade_activity']
            }
            for r in self.test_results
        ])
        
        return df_results
    
    def run_statistical_tests(self, n_iterations: int = 100) -> Dict[str, Any]:
        """Ejecuta tests estadísticos con múltiples iteraciones"""
        print(f"\nEjecutando tests estadísticos con {n_iterations} iteraciones por tipo de curva...\n")
        
        curve_configs = [
            ('perfect_linear_ideal', lambda: self.generator.generate_perfect_linear(slope=0.5, noise=0.0)),
            ('perfect_linear_noisy', lambda: self.generator.generate_perfect_linear(slope=0.5, noise=1.0)),
            ('volatile_positive', lambda: self.generator.generate_volatile(trend=0.1, volatility=5.0)),
            ('volatile_negative', lambda: self.generator.generate_volatile(trend=-0.1, volatility=5.0)),
            ('random_walk', lambda: self.generator.generate_random_walk(drift=0.0)),
        ]
        
        stats_results = {}
        
        for curve_name, curve_generator in curve_configs:
            scores = []
            
            for _ in range(n_iterations):
                equity, trade_stats = curve_generator()
                metrics_tuple = evaluate_report(equity, trade_stats)
                score = metrics_tuple[0]
                scores.append(score)
                
            scores = np.array(scores)
            
            stats_results[curve_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores),
                'q25': np.percentile(scores, 25),
                'q75': np.percentile(scores, 75),
                'scores': scores
            }
            
            print(f"{curve_name}:")
            print(f"  Mean: {stats_results[curve_name]['mean']:.4f} ± {stats_results[curve_name]['std']:.4f}")
            print(f"  Range: [{stats_results[curve_name]['min']:.4f}, {stats_results[curve_name]['max']:.4f}]")
            print(f"  Median: {stats_results[curve_name]['median']:.4f}")
            print(f"  IQR: [{stats_results[curve_name]['q25']:.4f}, {stats_results[curve_name]['q75']:.4f}]\n")
            
        return stats_results
    
    def visualize_results(self, n_examples: int = 6):
        """Visualiza ejemplos de curvas y sus scores"""
        # Seleccionar ejemplos representativos
        examples = [
            ('Perfect Linear', {'slope': 0.5, 'noise': 0.0}, 'perfect_linear'),
            ('Linear with Noise', {'slope': 0.5, 'noise': 1.0}, 'perfect_linear'),
            ('Exponential', {'rate': 0.001}, 'exponential'),
            ('Volatile', {'trend': 0.1, 'volatility': 5.0}, 'volatile'),
            ('Drawdown Recovery', {'dd_depth': 0.3, 'dd_location': 0.5}, 'drawdown_recovery'),
            ('Random Walk', {'drift': 0.0}, 'random_walk')
        ]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (title, params, curve_type) in enumerate(examples):
            ax = axes[idx]
            
            # Generar curva
            if curve_type == 'perfect_linear':
                equity, trade_stats = self.generator.generate_perfect_linear(**params)
            elif curve_type == 'exponential':
                equity, trade_stats = self.generator.generate_exponential(**params)
            elif curve_type == 'volatile':
                equity, trade_stats = self.generator.generate_volatile(**params)
            elif curve_type == 'drawdown_recovery':
                equity, trade_stats = self.generator.generate_drawdown_recovery(**params)
            elif curve_type == 'random_walk':
                equity, trade_stats = self.generator.generate_random_walk(**params)
                
            # Evaluar
            metrics_tuple = evaluate_report(equity, trade_stats)
            metrics_dict = metrics_tuple_to_dict(metrics_tuple)
            score = metrics_dict['final_score']
            
            # Plotear
            ax.plot(equity, linewidth=1.5)
            ax.set_title(f'{title}\nScore: {score:.4f}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Equity')
            ax.grid(True, alpha=0.3)
            
            # Añadir métricas clave
            metrics_text = (f"R²: {metrics_dict['r2']:.3f}\n"
                          f"Lin: {metrics_dict['perfect_linearity']:.3f}\n"
                          f"Trades: {int(trade_stats[0])}")
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('/workspace/studies/evaluate_report_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_metric_correlations(self):
        """Analiza correlaciones entre métricas"""
        if not self.test_results:
            print("No hay resultados para analizar. Ejecuta tests primero.")
            return
            
        # Extraer todas las métricas
        metrics_data = []
        for result in self.test_results:
            metrics = result['metrics']
            metrics_data.append({
                'final_score': metrics['final_score'],
                'r2': metrics['r2'],
                'perfect_linearity': metrics['perfect_linearity'],
                'linearity_bonus': metrics['linearity_bonus'],
                'consistency': metrics['consistency'],
                'slope_reward': metrics['slope_reward'],
                'monotonic_growth': metrics['monotonic_growth'],
                'smoothness': metrics['smoothness'],
                'dd_penalty': metrics['dd_penalty'],
                'trade_activity': metrics['trade_activity']
            })
            
        df_metrics = pd.DataFrame(metrics_data)
        
        # Calcular correlaciones
        corr_matrix = df_metrics.corr()
        
        # Visualizar
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Correlaciones entre Métricas')
        plt.tight_layout()
        plt.savefig('/workspace/studies/metric_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Imprimir correlaciones con final_score
        print("\nCorrelaciones con final_score:")
        corr_with_score = corr_matrix['final_score'].sort_values(ascending=False)
        for metric, corr in corr_with_score.items():
            if metric != 'final_score':
                print(f"  {metric}: {corr:.4f}")
                
        return df_metrics
    
    def generate_report(self, df_results: pd.DataFrame, stats_results: Dict[str, Any]):
        """Genera un reporte completo de los resultados"""
        report = []
        report.append("=" * 80)
        report.append("REPORTE DE EVALUACIÓN DE LA FUNCIÓN evaluate_report")
        report.append("=" * 80)
        report.append("")
        
        # Resumen de precisión
        accuracy = df_results['in_range'].mean()
        report.append(f"PRECISIÓN GENERAL: {accuracy:.2%} ({df_results['in_range'].sum()}/{len(df_results)} tests pasados)")
        report.append("")
        
        # Análisis por tipo de curva
        report.append("ANÁLISIS POR TIPO DE CURVA:")
        report.append("-" * 40)
        
        curve_summary = df_results.groupby('curve_type').agg({
            'actual_score': ['mean', 'std', 'min', 'max'],
            'in_range': 'mean'
        }).round(4)
        
        for curve_type in curve_summary.index:
            stats = curve_summary.loc[curve_type]
            report.append(f"\n{curve_type.upper()}:")
            report.append(f"  Score promedio: {stats['actual_score']['mean']:.4f} ± {stats['actual_score']['std']:.4f}")
            report.append(f"  Rango: [{stats['actual_score']['min']:.4f}, {stats['actual_score']['max']:.4f}]")
            report.append(f"  Precisión: {stats['in_range']['mean']:.2%}")
            
        # Casos problemáticos
        report.append("\n" + "=" * 40)
        report.append("CASOS PROBLEMÁTICOS (fuera del rango esperado):")
        report.append("-" * 40)
        
        failed_tests = df_results[~df_results['in_range']]
        if len(failed_tests) > 0:
            for _, row in failed_tests.iterrows():
                report.append(f"\n{row['curve_type']} {row['params']}:")
                report.append(f"  Score actual: {row['actual_score']:.4f}")
                report.append(f"  Rango esperado: [{row['expected_min']:.4f}, {row['expected_max']:.4f}]")
                report.append(f"  Desviación: {min(abs(row['actual_score'] - row['expected_min']), abs(row['actual_score'] - row['expected_max'])):.4f}")
        else:
            report.append("  ¡Ningún caso problemático encontrado!")
            
        # Estadísticas de robustez
        report.append("\n" + "=" * 40)
        report.append("ANÁLISIS DE ROBUSTEZ ESTADÍSTICA:")
        report.append("-" * 40)
        
        for curve_name, stats in stats_results.items():
            report.append(f"\n{curve_name}:")
            report.append(f"  Consistencia (1-CV): {1 - stats['std']/max(stats['mean'], 0.001):.4f}")
            report.append(f"  Rango relativo: {(stats['max'] - stats['min'])/max(stats['mean'], 0.001):.4f}")
            
        # Recomendaciones
        report.append("\n" + "=" * 40)
        report.append("RECOMENDACIONES DE MEJORA:")
        report.append("-" * 40)
        
        # Analizar debilidades
        if accuracy < 0.9:
            report.append("\n- La precisión general es menor al 90%. Revisar los pesos de las métricas.")
            
        # Curvas lineales perfectas
        perfect_linear_scores = df_results[df_results['curve_type'] == 'perfect_linear']['actual_score'].values
        if len(perfect_linear_scores) > 0 and perfect_linear_scores.min() < 0.85:
            report.append("\n- Las curvas lineales perfectas no siempre obtienen scores muy altos.")
            report.append("  Considerar aumentar el peso de 'perfect_linearity' y 'linearity_bonus'.")
            
        # Curvas volátiles
        volatile_scores = df_results[df_results['curve_type'] == 'volatile']['actual_score'].values
        if len(volatile_scores) > 0 and volatile_scores.max() > 0.5:
            report.append("\n- Las curvas volátiles obtienen scores demasiado altos.")
            report.append("  Considerar penalizaciones más estrictas por volatilidad.")
            
        # Trade activity
        low_trade_activity = df_results[df_results['trade_activity'] < 0.05]
        if len(low_trade_activity) > 0:
            report.append(f"\n- {len(low_trade_activity)} casos con actividad de trades muy baja.")
            report.append("  La métrica trade_activity podría necesitar ajustes.")
            
        report.append("\n" + "=" * 80)
        report.append("FIN DEL REPORTE")
        report.append("=" * 80)
        
        # Guardar reporte
        with open('/workspace/studies/evaluate_report_analysis.txt', 'w') as f:
            f.write('\n'.join(report))
            
        # Imprimir reporte
        print('\n'.join(report))
        
        return report


def main():
    """Función principal para ejecutar el análisis completo"""
    print("Iniciando análisis exhaustivo de la función evaluate_report...")
    print("=" * 80)
    
    # Crear tester
    tester = EvaluateReportTester()
    
    # 1. Ejecutar tests comprehensivos
    print("\n1. EJECUTANDO TESTS COMPREHENSIVOS")
    print("-" * 40)
    df_results = tester.run_comprehensive_tests()
    
    # 2. Ejecutar tests estadísticos
    print("\n2. EJECUTANDO TESTS ESTADÍSTICOS")
    print("-" * 40)
    stats_results = tester.run_statistical_tests(n_iterations=100)
    
    # 3. Visualizar ejemplos
    print("\n3. GENERANDO VISUALIZACIONES")
    print("-" * 40)
    tester.visualize_results()
    
    # 4. Analizar correlaciones
    print("\n4. ANALIZANDO CORRELACIONES ENTRE MÉTRICAS")
    print("-" * 40)
    df_metrics = tester.analyze_metric_correlations()
    
    # 5. Generar reporte final
    print("\n5. GENERANDO REPORTE FINAL")
    print("-" * 40)
    report = tester.generate_report(df_results, stats_results)
    
    # 6. Guardar resultados
    print("\n6. GUARDANDO RESULTADOS")
    print("-" * 40)
    df_results.to_csv('/workspace/studies/evaluate_report_test_results.csv', index=False)
    print("Resultados guardados en: evaluate_report_test_results.csv")
    print("Reporte guardado en: evaluate_report_analysis.txt")
    print("Visualizaciones guardadas en: evaluate_report_examples.png y metric_correlations.png")
    
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETO")
    print("=" * 80)
    
    return df_results, stats_results, df_metrics


if __name__ == "__main__":
    df_results, stats_results, df_metrics = main()