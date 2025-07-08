import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))
from tester_lib import evaluate_report, metrics_tuple_to_dict, print_detailed_metrics

# Crear diferentes tipos de curvas para probar el scoring
def test_scoring_system():
    """Prueba el nuevo sistema de scoring con diferentes tipos de curvas más realistas."""
    
    print("🔍 DEBUG: Probando nuevo sistema de scoring...")
    periods_per_year = 6240.0  # H1
    
    # 1) Curva ideal: crecimiento suave y consistente (baja volatilidad)
    n = 1000
    np.random.seed(42)  # Para reproducibilidad
    
    # Tendencia suave con poco ruido
    trend = np.linspace(0, 100, n)
    smooth_noise = np.random.normal(0, 1.5, n)  # Volatilidad baja
    ideal_returns = np.diff(trend) + smooth_noise[1:]
    ideal_curve = np.cumsum(np.concatenate([[0], ideal_returns]))
    
    score1, metrics1 = evaluate_report(ideal_curve, periods_per_year)
    metrics_dict1 = metrics_tuple_to_dict(score1, metrics1, periods_per_year)
    
    print(f"\n=== CURVA IDEAL (suave, consistente) ===")
    print_detailed_metrics(metrics_dict1, "Curva Ideal")
    
    # 2) Curva volátil: misma tendencia pero con mucha volatilidad
    volatile_noise = np.random.normal(0, 8, n)  # Volatilidad muy alta
    volatile_returns = np.diff(trend) + volatile_noise[1:]
    volatile_curve = np.cumsum(np.concatenate([[0], volatile_returns]))
    
    score2, metrics2 = evaluate_report(volatile_curve, periods_per_year)
    metrics_dict2 = metrics_tuple_to_dict(score2, metrics2, periods_per_year)
    
    print(f"\n=== CURVA VOLÁTIL (mucho ruido, misma tendencia) ===")
    print_detailed_metrics(metrics_dict2, "Curva Volátil")
    
    # 3) Curva con drawdown severo realista
    # Simular una estrategia que va bien y luego colapsa
    phase1 = np.linspace(0, 60, 400)  # Subida inicial
    phase2 = np.linspace(60, 15, 200)  # Drawdown severo -75%
    phase3 = np.linspace(15, 45, 400)  # Recuperación parcial
    dd_curve = np.concatenate([phase1, phase2, phase3])
    
    score3, metrics3 = evaluate_report(dd_curve, periods_per_year)
    metrics_dict3 = metrics_tuple_to_dict(score3, metrics3, periods_per_year)
    
    print(f"\n=== CURVA CON DRAWDOWN SEVERO (-75%) ===")
    print_detailed_metrics(metrics_dict3, "Curva DD Severo")
    
    # 4) Curva lateral realista (sin tendencia clara)
    lateral_base = 50  # Nivel base
    lateral_returns = np.random.normal(0, 3, n)  # Solo ruido
    lateral_curve = lateral_base + np.cumsum(lateral_returns)
    
    score4, metrics4 = evaluate_report(lateral_curve, periods_per_year)
    metrics_dict4 = metrics_tuple_to_dict(score4, metrics4, periods_per_year)
    
    print(f"\n=== CURVA LATERAL (sin tendencia clara) ===")
    print_detailed_metrics(metrics_dict4, "Curva Lateral")
    
    # 5) Curva perdedora con tendencia negativa clara
    losing_trend = np.linspace(100, 30, n)  # Tendencia bajista clara
    losing_noise = np.random.normal(0, 2, n)
    losing_returns = np.diff(losing_trend) + losing_noise[1:]
    losing_curve = np.cumsum(np.concatenate([[100], losing_returns]))
    
    score5, metrics5 = evaluate_report(losing_curve, periods_per_year)
    metrics_dict5 = metrics_tuple_to_dict(score5, metrics5, periods_per_year)
    
    print(f"\n=== CURVA PERDEDORA (tendencia bajista) ===")
    print_detailed_metrics(metrics_dict5, "Curva Perdedora")
    
    # 6) Curva con buen Sharpe pero drawdown moderado
    good_trend = np.linspace(0, 80, n)
    moderate_noise = np.random.normal(0, 3, n)
    # Agregar un drawdown moderado en el medio
    moderate_dd = np.array([max(0, -15 * np.exp(-(i-500)**2/5000)) for i in range(n)])
    moderate_returns = np.diff(good_trend) + moderate_noise[1:] + moderate_dd[1:]
    moderate_curve = np.cumsum(np.concatenate([[0], moderate_returns]))
    
    score6, metrics6 = evaluate_report(moderate_curve, periods_per_year)
    metrics_dict6 = metrics_tuple_to_dict(score6, metrics6, periods_per_year)
    
    print(f"\n=== CURVA MODERADA (buen Sharpe, DD moderado) ===")
    print_detailed_metrics(metrics_dict6, "Curva Moderada")
    
    # Resumen de scores
    print(f"\n🔍 DEBUG: === RESUMEN DE SCORES ===")
    print(f"1. Curva Ideal:      {score1:.4f}")
    print(f"2. Curva Volátil:    {score2:.4f}")
    print(f"3. Curva DD Severo:  {score3:.4f}")
    print(f"4. Curva Lateral:    {score4:.4f}")
    print(f"5. Curva Perdedora:  {score5:.4f}")
    print(f"6. Curva Moderada:   {score6:.4f}")
    
    # Verificar que el ranking tiene sentido
    scores = [score1, score2, score3, score4, score5, score6]
    names = ["Ideal", "Volátil", "DD Severo", "Lateral", "Perdedora", "Moderada"]
    
    print(f"\n🔍 DEBUG: === RANKING REAL ===")
    sorted_pairs = sorted(zip(scores, names), reverse=True)
    for i, (score, name) in enumerate(sorted_pairs, 1):
        print(f"{i}. {name}: {score:.4f}")
    
    print(f"\n🔍 DEBUG: === RANKING ESPERADO ===")
    print("1. Ideal (bajo ruido, tendencia clara)")
    print("2. Moderada (balance riesgo-retorno)")
    print("3. Volátil (misma tendencia pero mucho ruido)")
    print("4. DD Severo (drawdown -75%)")
    print("5. Lateral (sin tendencia)")
    print("6. Perdedora (tendencia negativa)")
    
    # Plot para visualizar
    plt.figure(figsize=(12, 7))
    
    curves = [ideal_curve, volatile_curve, dd_curve, lateral_curve, losing_curve, moderate_curve]
    titles = [f'1. Ideal (Score: {score1:.3f})', f'2. Volátil (Score: {score2:.3f})', 
              f'3. DD Severo (Score: {score3:.3f})', f'4. Lateral (Score: {score4:.3f})',
              f'5. Perdedora (Score: {score5:.3f})', f'6. Moderada (Score: {score6:.3f})']
    
    for i, (curve, title) in enumerate(zip(curves, titles), 1):
        plt.subplot(2, 3, i)
        plt.plot(curve)
        plt.title(title)
        plt.grid(True)
        plt.xlabel('Períodos')
        plt.ylabel('P&L Acumulado')
    
    plt.tight_layout()
    plt.show()
    
    # Verificar lógica del ranking
    print(f"\n🔍 DEBUG: === ANÁLISIS DE CONSISTENCIA ===")
    if score1 > score2:
        print("✅ BIEN: Curva ideal > Curva volátil")
    else:
        print("❌ MAL: Curva volátil > Curva ideal")
        
    if score6 > score3:
        print("✅ BIEN: Curva moderada > DD severo")
    else:
        print("❌ MAL: DD severo > Curva moderada")
        
    if score3 > score4:
        print("✅ BIEN: DD severo > Lateral")
    else:
        print("❌ MAL: Lateral > DD severo")
        
    if score4 > score5:
        print("✅ BIEN: Lateral > Perdedora")
    else:
        print("❌ MAL: Perdedora > Lateral")

if __name__ == "__main__":
    test_scoring_system() 