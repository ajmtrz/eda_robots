#!/usr/bin/env python3
"""
Script de demostración de las optimizaciones realizadas en evaluate_report
para promover más trades y ganancias.
"""

import math

def simulate_trade_activity_score_old(trade_stats, eq_length):
    """Simulación de la función _trade_activity_score ANTES de la optimización"""
    if eq_length < 10:
        return 0.0
    
    total_trades = trade_stats[0]
    positive_trades = trade_stats[1]
    win_rate = trade_stats[4]
    
    if total_trades <= 0:
        return 0.0
    
    # Frecuencia normalizada
    trade_frequency = total_trades / eq_length
    
    # Sistema ANTIGUO
    if trade_frequency <= 0.01:
        freq_score = trade_frequency / 0.01 * 0.3
    elif trade_frequency <= 0.25:
        freq_score = 0.3 + (trade_frequency - 0.01) / 0.24 * 0.6
    else:
        excess = trade_frequency - 0.25
        freq_score = 0.9 * math.exp(-excess * 3.0)
    
    # Calidad de trades
    positive_ratio = positive_trades / total_trades
    
    if positive_ratio >= 0.8:
        quality_score = 0.8 + (positive_ratio - 0.8) / 0.2 * 0.2 * 2.0
    elif positive_ratio >= 0.6:
        quality_score = 0.4 + (positive_ratio - 0.6) / 0.2 * 0.4
    elif positive_ratio >= 0.5:
        quality_score = 0.2 + (positive_ratio - 0.5) / 0.1 * 0.2
    else:
        quality_score = positive_ratio / 0.5 * 0.2
    
    # Bonus por actividad positiva
    positive_activity = positive_trades / eq_length
    
    if positive_activity > 0.15:
        activity_bonus = 1.0 + min(0.3, (positive_activity - 0.15) * 2.0)
    elif positive_activity > 0.05:
        activity_bonus = 1.0 + (positive_activity - 0.05) / 0.1 * 0.15
    else:
        activity_bonus = 1.0
    
    # Score combinado
    base_score = freq_score * 0.4 + quality_score * 0.45
    final_score = base_score * activity_bonus
    
    return max(0.0, min(0.3, final_score))

def simulate_trade_activity_score_new(trade_stats, eq_length):
    """Simulación de la función _trade_activity_score DESPUÉS de la optimización"""
    if eq_length < 10:
        return 0.0
    
    total_trades = trade_stats[0]
    positive_trades = trade_stats[1]
    win_rate = trade_stats[4]
    
    if total_trades <= 0:
        return 0.0
    
    # Frecuencia normalizada - PROMOCIÓN AGRESIVA
    trade_frequency = total_trades / eq_length
    
    # NUEVO SISTEMA más agresivo
    if trade_frequency <= 0.005:
        freq_score = trade_frequency / 0.005 * 0.2
    elif trade_frequency <= 0.02:
        freq_score = 0.2 + (trade_frequency - 0.005) / 0.015 * 0.4
    elif trade_frequency <= 0.40:  # Rango expandido hasta 40%
        freq_score = 0.6 + (trade_frequency - 0.02) / 0.38 * 0.35
    else:
        excess = trade_frequency - 0.40
        freq_score = 0.95 * math.exp(-excess * 2.0)  # Decaimiento más suave
    
    # Calidad de trades - BONUS EXPONENCIAL
    positive_ratio = positive_trades / total_trades
    
    # NUEVO SISTEMA más agresivo
    if positive_ratio >= 0.85:
        quality_score = 1.0 + (positive_ratio - 0.85) / 0.15 * 0.5
    elif positive_ratio >= 0.75:
        quality_score = 0.8 + (positive_ratio - 0.75) / 0.1 * 0.2
    elif positive_ratio >= 0.65:
        quality_score = 0.6 + (positive_ratio - 0.65) / 0.1 * 0.2
    elif positive_ratio >= 0.55:
        quality_score = 0.4 + (positive_ratio - 0.55) / 0.1 * 0.2
    else:
        quality_score = positive_ratio / 0.55 * 0.4
    
    # Bonus por actividad positiva - PROMOCIÓN AGRESIVA
    positive_activity = positive_trades / eq_length
    
    # NUEVO SISTEMA más generoso
    if positive_activity > 0.20:
        activity_bonus = 1.0 + min(0.5, (positive_activity - 0.20) * 3.0)
    elif positive_activity > 0.10:
        activity_bonus = 1.0 + (positive_activity - 0.10) / 0.10 * 0.25
    elif positive_activity > 0.05:
        activity_bonus = 1.0 + (positive_activity - 0.05) / 0.05 * 0.15
    else:
        activity_bonus = 1.0
    
    # NUEVO BONUS por ganancias significativas
    if positive_trades >= 10 and positive_ratio >= 0.7:
        volume_bonus = 1.0 + min(0.3, positive_trades / 50.0)
    else:
        volume_bonus = 1.0
    
    # Score combinado - PESOS OPTIMIZADOS
    base_score = freq_score * 0.35 + quality_score * 0.40
    final_score = base_score * activity_bonus * volume_bonus
    
    return max(0.0, min(0.4, final_score))  # Cap aumentado a 40%

def simulate_slope_reward_old(slope):
    """Simulación de la función _slope_reward ANTES de la optimización"""
    if slope <= 0:
        return 0.0
    
    if slope < 0.05:
        return slope / 0.05 * 0.1
    elif slope < 0.2:
        return 0.1 + (slope - 0.05) / 0.15 * 0.3
    elif slope <= 1.5:
        base_reward = 0.4 + (slope - 0.2) / 1.3 * 0.5
        
        if 0.5 <= slope <= 1.0:
            ideal_bonus = 1.0 + 0.3  # 30% bonus
        else:
            ideal_bonus = 1.0
            
        return base_reward * ideal_bonus
    else:
        excess = slope - 1.5
        base_reward = 0.9
        decay = math.exp(-excess * 0.3)
        return base_reward * decay

def simulate_slope_reward_new(slope):
    """Simulación de la función _slope_reward DESPUÉS de la optimización"""
    if slope <= 0:
        return 0.0
    
    # NUEVO SISTEMA más agresivo
    if slope < 0.03:
        return slope / 0.03 * 0.15
    elif slope < 0.15:
        return 0.15 + (slope - 0.03) / 0.12 * 0.35
    elif slope < 0.3:
        return 0.5 + (slope - 0.15) / 0.15 * 0.3
    elif slope <= 2.0:  # Rango expandido
        base_reward = 0.8 + (slope - 0.3) / 1.7 * 0.15
        
        # NUEVO BONUS EXPONENCIAL
        if 0.5 <= slope <= 1.5:
            ideal_bonus = 1.0 + 0.5  # 50% bonus (aumentado de 30%)
        elif 0.3 <= slope <= 2.0:
            ideal_bonus = 1.0 + 0.2  # 20% bonus para rango ampliado
        else:
            ideal_bonus = 1.0
            
        return base_reward * ideal_bonus
    else:
        excess = slope - 2.0
        base_reward = 0.95
        decay = math.exp(-excess * 0.2)  # Decaimiento más suave
        return base_reward * decay

def demonstrate_optimizations():
    """Demuestra las optimizaciones realizadas"""
    print("🚀 DEMOSTRACIÓN DE OPTIMIZACIONES EN EVALUATE_REPORT")
    print("="*80)
    
    # Casos de prueba
    test_cases = [
        {
            'name': 'Alta actividad de trades',
            'trade_stats': [80, 65, 15, 0, 0.8125, 0.03, -0.01, 0.8125],
            'eq_length': 200,
            'slope': 0.02
        },
        {
            'name': 'Alta ganancia',
            'trade_stats': [25, 20, 5, 0, 0.8, 0.08, -0.02, 0.8],
            'eq_length': 200,
            'slope': 0.05
        },
        {
            'name': 'Combinación perfecta',
            'trade_stats': [60, 50, 10, 0, 0.833, 0.05, -0.015, 0.833],
            'eq_length': 200,
            'slope': 0.03
        },
        {
            'name': 'Muchos trades pequeños',
            'trade_stats': [100, 75, 25, 0, 0.75, 0.02, -0.008, 0.75],
            'eq_length': 200,
            'slope': 0.015
        }
    ]
    
    print(f"\n📊 COMPARACIÓN DE MÉTRICAS ANTES vs DESPUÉS")
    print(f"{'='*80}")
    
    for case in test_cases:
        print(f"\n🔍 CASO: {case['name']}")
        print(f"   Trade Stats: {case['trade_stats'][0]} trades, {case['trade_stats'][1]} positivos, {case['trade_stats'][4]:.3f} win rate")
        print(f"   Slope: {case['slope']:.4f}")
        
        # Comparar trade activity score
        old_trade_score = simulate_trade_activity_score_old(case['trade_stats'], case['eq_length'])
        new_trade_score = simulate_trade_activity_score_new(case['trade_stats'], case['eq_length'])
        trade_improvement = ((new_trade_score - old_trade_score) / old_trade_score * 100) if old_trade_score > 0 else 0
        
        # Comparar slope reward
        old_slope_score = simulate_slope_reward_old(case['slope'])
        new_slope_score = simulate_slope_reward_new(case['slope'])
        slope_improvement = ((new_slope_score - old_slope_score) / old_slope_score * 100) if old_slope_score > 0 else 0
        
        print(f"   📈 Trade Activity Score:")
        print(f"      ANTES: {old_trade_score:.4f}")
        print(f"      DESPUÉS: {new_trade_score:.4f}")
        print(f"      MEJORA: {trade_improvement:+.1f}%")
        
        print(f"   📈 Slope Reward:")
        print(f"      ANTES: {old_slope_score:.4f}")
        print(f"      DESPUÉS: {new_slope_score:.4f}")
        print(f"      MEJORA: {slope_improvement:+.1f}%")
    
    print(f"\n🎯 RESUMEN DE OPTIMIZACIONES")
    print(f"{'='*80}")
    print(f"✅ PROMOCIÓN DE TRADES:")
    print(f"   • Frecuencia ideal expandida de 25% a 40%")
    print(f"   • Bonus por actividad positiva aumentado de 30% a 50%")
    print(f"   • Nuevo bonus por volumen de trades positivos")
    print(f"   • Cap máximo aumentado de 30% a 40%")
    
    print(f"\n✅ PROMOCIÓN DE GANANCIAS:")
    print(f"   • Rango ideal de pendiente expandido de 1.5 a 2.0")
    print(f"   • Bonus por pendiente ideal aumentado de 30% a 50%")
    print(f"   • Decaimiento más suave para pendientes altas")
    print(f"   • Recompensas más agresivas para pendientes moderadas")
    
    print(f"\n✅ PESOS REBALANCEADOS:")
    print(f"   • Linealidad: 45% → 35% (reducido)")
    print(f"   • Crecimiento: 25% → 30% (aumentado)")
    print(f"   • Calidad: 15% → 20% (aumentado)")
    print(f"   • Robustez (trades): 15% (mantenido)")
    
    print(f"\n✅ NUEVOS BONUS:")
    print(f"   • Bonus por excelencia en trading: 12% → 25%")
    print(f"   • Bonus por excelencia en ganancias: NUEVO 15%")
    print(f"   • Bonus por combinación perfecta: NUEVO 18%")
    print(f"   • Criterios más flexibles para todos los bonus")

if __name__ == "__main__":
    demonstrate_optimizations()