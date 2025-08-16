//+------------------------------------------------------------------+
//|                                                        KmeansCat |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
#include <Trade\AccountInfo.mqh>
#include <Math\Alglib\alglib.mqh>
#include <ajmtrz/include/clsOptimalF.mqh>
#include <ajmtrz\include\Dmitrievsky\XAUUSD_H1_buy_cl_lg_ma_ra_cl.mqh>
#include <Indicators\Indicators.mqh>
#property strict
#property copyright "Copyright 2025, Dmitrievsky"
#property link      "https://www.mql5.com/ru/users/dmitrievsky"
#property version   "3.00"

CTrade         m_trade;
CiOpen         m_open;
CiHigh         m_high;
CiLow          m_low;
CiClose        m_close;
CiTickVolume   m_vol;
CiATR          m_atr;
CPositionInfo  m_position;

input int      max_orders        = 1;
input int      delay_bars        = 1;
input double   manual_lot        = 0.01;
input int      atr_period        = 14;
input double   stoploss          = 0.0;
input double   takeprofit        = 0.0;
input bool     debug             = false;

static datetime last_time = 0;
static int bar_counter = 0; // Contador de barras
static int last_trade_bar_index = -delay_bars - 1;
const ulong ExtInputShape_main[]  = {1, ArraySize(PERIODS_MAIN)};
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
const ulong ExtInputShape_meta[] = {1, ArraySize(PERIODS_META)};
long     ExtHandle_main = INVALID_HANDLE, ExtHandle_meta = INVALID_HANDLE;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   TesterHideIndicators(true);
   m_trade.SetExpertMagicNumber(ulong(MAGIC_NUMBER));
   ExtHandle_main = OnnxCreateFromBuffer(ExtModel_main, ONNX_DEFAULT);
   ExtHandle_meta = OnnxCreateFromBuffer(ExtModel_meta, ONNX_DEFAULT);

   if(ExtHandle_main == INVALID_HANDLE || ExtHandle_meta == INVALID_HANDLE)
     {
      Print("OnnxCreateFromBuffer failed, error ", GetLastError());
      return(INIT_FAILED);
     }

   if(!OnnxSetInputShape(ExtHandle_main, 0, ExtInputShape_main))
     {
      Print("OnnxSetInputShape 1 failed, error ", GetLastError());
      OnnxRelease(ExtHandle_main);
      return(INIT_FAILED);
     }
   // Clasificación: salida con probabilidades
   const ulong output_shape_main[] = {1};

   if(!OnnxSetOutputShape(ExtHandle_main, 0, output_shape_main))
      {
      Print("OnnxSetOutputShape main (classification) failed, error ", GetLastError());
      return(INIT_FAILED);
      }
   if(!OnnxSetInputShape(ExtHandle_meta, 0, ExtInputShape_meta))
     {
      Print("OnnxSetInputShape meta failed, error ", GetLastError());
      OnnxRelease(ExtHandle_meta);
      return(INIT_FAILED);
     }
   const ulong output_shape_meta[] = {1};
   if(!OnnxSetOutputShape(ExtHandle_meta, 0, output_shape_meta))
     {
      Print("OnnxSetOutputShape meta failed, error ", GetLastError());
      return(INIT_FAILED);
     }
//--- initialize object
   if(debug)
     {
      if(!m_open.Create(_Symbol, _Period))
        {
         Print(__FUNCTION__ + ": error initializing open object");
         return(INIT_FAILED);
        }
      if(!m_high.Create(_Symbol, _Period))
        {
         Print(__FUNCTION__ + ": error initializing high object");
         return(INIT_FAILED);
        }
      if(!m_low.Create(_Symbol, _Period))
        {
         Print(__FUNCTION__ + ": error initializing low object");
         return(INIT_FAILED);
        }
      if(!m_close.Create(_Symbol, _Period))
        {
         Print(__FUNCTION__ + ": error initializing close object");
         return(INIT_FAILED);
        }
      if(!m_vol.Create(_Symbol, _Period))
        {
         Print(__FUNCTION__ + ": error initializing volume object");
         return(INIT_FAILED);
        }
     }
   if(!m_atr.Create(_Symbol, _Period, atr_period))
     {
      Print(__FUNCTION__ + ": error initializing atr object");
      return(INIT_FAILED);
     }
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   OnnxRelease(ExtHandle_main);
   OnnxRelease(ExtHandle_meta);
  }
//+------------------------------------------------------------------+
//| Expert tick function - VERSIÓN EQUIVALENTE A PYTHON             |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!isNewBar())
      return;
   bar_counter++;
   m_atr.Refresh();

   double features_main[ArraySize(PERIODS_MAIN)], features_meta[ArraySize(PERIODS_META)];
   fill_arays_main(features_main);
   fill_arays_meta(features_meta);

   datetime current_time = iTime(_Symbol, PERIOD_CURRENT, 0);

   static vector main_out(1), meta_out(1);
   struct output
     {
      long           label[];
      float          proba[];
     };
   output main_out2[], meta_out2[];

   double main_sig;
   double meta_sig;

// Ejecutar modelo meta (siempre clasificación)
   OnnxRun(ExtHandle_meta, ONNX_DEFAULT, features_meta, meta_out, meta_out2);
   meta_sig = meta_out2[0].proba[1];

// Ejecutar modelo main según tipo
   OnnxRun(ExtHandle_main, ONNX_DEFAULT, features_main, main_out, main_out2);
   main_sig = main_out2[0].proba[1];

// Remove prob_buy/prob_sell logic - signals are generated directly from main_sig

// EXACT PYTHON SIGNAL LOGIC
   bool buy_sig, sell_sig;
   // CLASIFICACIÓN: EXACT PYTHON LOGIC - use main_sig directly for both directions
   if(DIRECTION == "buy")
      {
      buy_sig = main_sig > MAIN_THRESHOLD;
      sell_sig = false;
      }
   else if(DIRECTION == "sell")
      {
      buy_sig = false;
      sell_sig = main_sig > MAIN_THRESHOLD;
      }
            else // "both"
      {
      // Para clasificación: probabilidad alta = BUY, probabilidad baja = SELL
      // Threshold filtra ambas señales: buy_sig cuando > threshold, sell_sig cuando < (1-threshold)
      buy_sig = main_sig > MAIN_THRESHOLD;
      sell_sig = main_sig < (1.0 - MAIN_THRESHOLD);
      }
   
   bool meta_ok  = (meta_sig > META_THRESHOLD);

   if(debug)
      print_features_debug(features_main, features_meta, main_sig, meta_sig);

//--- 1) CERRAR posiciones cuyas señales hayan desaparecido - EXACT PYTHON LOGIC
   for(int i = PositionsTotal() - 1; i >= 0; --i)
     {
      if(PositionGetSymbol(i) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != MAGIC_NUMBER)
         continue;

      ENUM_POSITION_TYPE ptype = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      bool must_close = false;

      // EXACT PYTHON CLOSING LOGIC
      // CLASIFICACIÓN: EXACT PYTHON LOGIC
      if(DIRECTION == "buy")
         {
         must_close = (ptype == POSITION_TYPE_BUY && !buy_sig);
         }
      else if(DIRECTION == "sell")
         {
         must_close = (ptype == POSITION_TYPE_SELL && !sell_sig);
         }
      else // "both"
         {
         must_close = (ptype == POSITION_TYPE_BUY && !buy_sig) || 
                     (ptype == POSITION_TYPE_SELL && !sell_sig);
         }
      
      if(debug && must_close)
        Print("🔍 DEBUG - Cerrando posición: tipo=", EnumToString(ptype), 
              " buy_sig=", buy_sig, " sell_sig=", sell_sig, " meta_ok=", meta_ok, " main_sig=", main_sig);

              if(must_close)
         {
          m_trade.PositionClose(PositionGetString(POSITION_SYMBOL));
         }
     }

//--- 2) ABRIR nuevas posiciones - EXACT PYTHON LOGIC
   int live_pos = countOrders(MAGIC_NUMBER);
   bool delay_ok = (bar_counter - last_trade_bar_index) >= delay_bars;
   bool pool_ok = (max_orders == 0 || live_pos < max_orders);

   if(meta_ok && delay_ok && pool_ok)
     {
      bool trade_opened_this_bar = false;

      // EXACT PYTHON OPENING LOGIC: BUY first, then SELL (checking position limits each time)
      // BUY signal processing
      if(buy_sig && (max_orders == 0 || live_pos < max_orders))
        {
         if(debug)
            Print("🔍 DEBUG - ABRIENDO BUY: main_sig=", main_sig, " threshold=", MAIN_THRESHOLD);
         double atr = m_atr.Main(0);
         double sl_points = stoploss * atr;
         double tp_points = takeprofit * atr;
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl_price  = sl_points == 0.0 ? 0.0 : ask - sl_points;
         double tp_price  = tp_points == 0.0 ? 0.0 : ask + tp_points;
         double lot = (manual_lot > 0.0) ? manual_lot : LotsOptimized(sl_points);
         string bot_comment = string(MAGIC_NUMBER);
         m_trade.PositionOpen(_Symbol, ORDER_TYPE_BUY, lot, ask, sl_price, tp_price, bot_comment);
         trade_opened_this_bar = true;
         live_pos = countOrders(MAGIC_NUMBER); // Re-count positions after opening
        }

      // SELL signal processing - Check position limit again after potential BUY opening
      if(sell_sig && (max_orders == 0 || live_pos < max_orders))
        {
         if(debug)
            Print("🔍 DEBUG - ABRIENDO SELL: main_sig=", main_sig, " threshold=", MAIN_THRESHOLD);
         double atr = m_atr.Main(0);
         double sl_points = stoploss * atr;
         double tp_points = takeprofit * atr;
         double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double sl_price  = sl_points == 0.0 ? 0.0 : bid + sl_points;
         double tp_price  = tp_points == 0.0 ? 0.0 : bid - tp_points;
         double lot = (manual_lot > 0.0) ? manual_lot : LotsOptimized(sl_points);
         string bot_comment = string(MAGIC_NUMBER);
         m_trade.PositionOpen(_Symbol, ORDER_TYPE_SELL, lot, bid, sl_price, tp_price, bot_comment);
         trade_opened_this_bar = true;
         live_pos = countOrders(MAGIC_NUMBER); // Re-count positions after opening
        }

      // Update last_trade_bar only once per bar, regardless of how many positions opened
      if(trade_opened_this_bar)
         last_trade_bar_index = bar_counter;
     }
  }
//+------------------------------------------------------------------+
//|   OnTester – criterio "evaluate_report"                          |
//+------------------------------------------------------------------+
double OnTester()
  {
//── 1) Recoger trades cerrados (reconstruir P&L en unidades de precio open-open como Python) ----
   HistorySelect(0, TimeCurrent());
   int total_deals = HistoryDealsTotal();

   double profits[];
   long   times[];
   int    n_trades = 0;

   // Arrays auxiliares para mapear entradas por POSITION_ID
   long   positionIDs[];
   long   entryTimes[];
   int    entryTypes[]; // DEAL_TYPE_BUY o DEAL_TYPE_SELL en la ENTRADA
   int    posCount = 0;

   // Primer recorrido: recolectar ENTRADAS para poder emparejarlas con SALIDAS
   for(int i = 0; i < total_deals; i++)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(!ticket)
         continue;
      if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol)
         continue;
      if((ulong)HistoryDealGetInteger(ticket, DEAL_MAGIC) != MAGIC_NUMBER)
         continue;

      int entryFlag = (int)HistoryDealGetInteger(ticket, DEAL_ENTRY);
      if(entryFlag == DEAL_ENTRY_IN)
        {
         ArrayResize(positionIDs, posCount + 1);
         ArrayResize(entryTimes, posCount + 1);
         ArrayResize(entryTypes, posCount + 1);

         positionIDs[posCount] = (long)HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
         entryTimes[posCount]  = (long)HistoryDealGetInteger(ticket, DEAL_TIME);
         entryTypes[posCount]  = (int)HistoryDealGetInteger(ticket, DEAL_TYPE); // BUY => posición larga; SELL => corta
         posCount++;
        }
     }

   // Segundo recorrido: construir profits en términos de delta de precio entre open(entry) y open(exit)
   for(int i = 0; i < total_deals; i++)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(!ticket)
         continue;
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT)
         continue;
      if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol)
         continue;
      if((ulong)HistoryDealGetInteger(ticket, DEAL_MAGIC) != MAGIC_NUMBER)
         continue;

      long posID = (long)HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
      long out_time = (long)HistoryDealGetInteger(ticket, DEAL_TIME);

      // Buscar la entrada correspondiente
      int idx = -1;
      for(int j = 0; j < posCount; j++)
        {
         if(positionIDs[j] == posID)
           {
            idx = j;
            break;
           }
        }
      if(idx < 0)
         continue; // entrada no encontrada; saltar

      long in_time = entryTimes[idx];
      int in_type = entryTypes[idx]; // BUY => largo; SELL => corto

      // Obtener precios 'open' de las barras de entrada y salida
      int in_shift  = iBarShift(_Symbol, PERIOD_CURRENT, (datetime)in_time, true);
      int out_shift = iBarShift(_Symbol, PERIOD_CURRENT, (datetime)out_time, true);
      if(in_shift < 0 || out_shift < 0)
         continue;

      double in_open  = iOpen(_Symbol, PERIOD_CURRENT, in_shift);
      double out_open = iOpen(_Symbol, PERIOD_CURRENT, out_shift);

      // Calcular profit en unidades de precio (como Python backtest)
      double p = 0.0;
      if(in_type == DEAL_TYPE_BUY) // posición larga
         p = out_open - in_open;
      else                         // posición corta
         p = in_open - out_open;

      ArrayResize(profits, n_trades + 1);
      ArrayResize(times, n_trades + 1);
      profits[n_trades] = p;
      times[n_trades]   = out_time;
      n_trades++;
     }

// Incluir posiciones aún abiertas al final, forzando cierre al open de la última barra
   for(int i = PositionsTotal() - 1; i >= 0; --i)
     {
      if(PositionGetSymbol(i) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != MAGIC_NUMBER)
         continue;

      datetime in_time_pos = (datetime)PositionGetInteger(POSITION_TIME);
      int in_shift = iBarShift(_Symbol, PERIOD_CURRENT, in_time_pos, true);
      int out_shift = 0; // última barra
      if(in_shift < 0 || out_shift < 0)
         continue;

      double in_open  = iOpen(_Symbol, PERIOD_CURRENT, in_shift);
      double out_open = iOpen(_Symbol, PERIOD_CURRENT, out_shift);

      ENUM_POSITION_TYPE ptype = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double p = (ptype == POSITION_TYPE_BUY) ? (out_open - in_open) : (in_open - out_open);

      ArrayResize(profits, n_trades + 1);
      ArrayResize(times, n_trades + 1);
      profits[n_trades] = p;
      times[n_trades]   = (long)TimeCurrent();
      n_trades++;
     }

   // CRITICAL: Match Python minimum trades requirement
   const int min_trades = 200;
   const double rdd_floor = 1.0;

   if(n_trades < min_trades)
      return -1.0;

//── 2) Ordenar por tiempo de cierre ------------------------------
   int order[];
   ArrayResize(order, n_trades);
   for(int i = 0; i < n_trades; i++)
      order[i] = i;

   for(int i = 0; i < n_trades - 1; i++)
     {
      int min = i;
      for(int j = i + 1; j < n_trades; j++)
         if(times[order[j]] < times[order[min]])
            min = j;

      if(min != i)
        {
         int tmp = order[i];
         order[i] = order[min];
         order[min] = tmp;
        }
     }

//── 3) Construir curva de equity y profits ordenados ------------
   double profit_ordered[];
   ArrayResize(profit_ordered, n_trades);

   double equity[];
   ArrayResize(equity, n_trades + 1);
   equity[0] = 0.0;

   for(int i = 0; i < n_trades; i++)
     {
      profit_ordered[i] = profits[order[i]];
      equity[i + 1] = equity[i] + profit_ordered[i];
     }

//── 4) Calcular métricas base -----------------------------------

// Normalización nº trades - MATCH PYTHON EXACTLY
   double trade_nl = 1.0 / (1.0 + MathExp(-(n_trades - min_trades) / (min_trades * 5.0)));

// Calcular drawdown máximo - MATCH PYTHON EXACTLY
   double running_max[];
   ArrayResize(running_max, n_trades + 1);
   running_max[0] = equity[0];

   for(int i = 1; i < n_trades + 1; i++)
      running_max[i] = (running_max[i-1] > equity[i]) ? running_max[i-1] : equity[i];

   double max_dd = 0.0;
   for(int i = 0; i < n_trades + 1; i++)
     {
      double dd = running_max[i] - equity[i];
      if(dd > max_dd)
         max_dd = dd;
     }

// Calcular retorno total y ratio retorno/drawdown - MATCH PYTHON
   double total_ret = equity[n_trades] - equity[0];
   double rdd;
   if(max_dd == 0.0)
      rdd = 0.0;
   else
      rdd = total_ret / max_dd;

// MATCH PYTHON: Umbral más exigente para el ratio retorno/drawdown
   double min_rdd = MathMax(rdd_floor * 1.5, 2.0);  // Al menos 2.0 o 1.5 veces el floor
   if(rdd < min_rdd)
      return -1.0;

// MATCH PYTHON: Normalización más estricta que penaliza ratios bajos
   double rdd_nl = 1.0 / (1.0 + MathExp(-(rdd - min_rdd) / (min_rdd * 3.0)));

//── 5) Regresión lineal sobre la curva de equity - MATCH PYTHON EXACTLY
   int N = ArraySize(equity);
   double x_mean = 0.0;
   double y_mean = 0.0;

// Calculate means
   for(int i = 0; i < N; i++)
     {
      x_mean += i;
      y_mean += equity[i];
     }
   x_mean /= N;
   y_mean /= N;

// Calculate slope using Python's exact method
   double numerator = 0.0;
   double denominator = 0.0;

   for(int i = 0; i < N; i++)
     {
      double x_diff = i - x_mean;
      double y_diff = equity[i] - y_mean;
      numerator += x_diff * y_diff;
      denominator += x_diff * x_diff;
     }

   if(MathAbs(denominator) < 1e-12)
      return -1.0;

   double slope = numerator / denominator;
   double intercept = y_mean - slope * x_mean;

// Calculate R² using Python's exact method
   double ss_res = 0.0;
   double ss_tot = 0.0;

   for(int i = 0; i < N; i++)
     {
      double y_pred = slope * i + intercept;
      double y_diff_mean = equity[i] - y_mean;
      double y_diff_pred = equity[i] - y_pred;

      ss_res += y_diff_pred * y_diff_pred;
      ss_tot += y_diff_mean * y_diff_mean;
     }
   if(slope < 0.0)
      return -1.0;

   double r2;
   if(MathAbs(ss_tot) < 1e-12)
      r2 = 0.0;
   else
      r2 = 1.0 - (ss_res / ss_tot);

// MATCH PYTHON: Penalizar más fuertemente R² bajos para favorecer linealidad
   if(r2 < 0.7)  // Umbral más estricto para R²
      r2 = r2 * 0.5;  // Penalización severa para R² < 0.7
   else
      r2 = 0.35 + (r2 - 0.7) * 2.17;  // Reescalar 0.7-1.0 a 0.35-1.0

// MATCH PYTHON: Normalize slope - más sensible a pendientes pequeñas
   double slope_nl = 1.0 / (1.0 + MathExp(-(MathLog(1.0 + slope) / 3.0)));

//── 6) Walk-Forward Analysis - MATCH PYTHON EXACTLY -----------
   double wf_nl = WalkForwardValidation(equity, profit_ordered, n_trades);

//── 7) Calcular score final - EXACT PYTHON WEIGHTS ----------------
   double score = 0.20 * r2 +        // Linealidad de la curva (R²)
                  0.20 * slope_nl +  // Pendiente positiva consistente
                  0.10 * rdd_nl +    // Ratio retorno/drawdown
                  0.05 * trade_nl +  // Número de trades (menor importancia)
                  0.45 * wf_nl;      // Consistencia temporal (máxima prioridad)

// CRITICAL: Apply same logic as Python - check for negative scores and invalid metrics
   if(score < 0.0 || trade_nl <= -1.0 || rdd_nl <= -1.0 || r2 <= -1.0 || slope_nl <= -1.0 || wf_nl <= -1.0)
      return -1.0;

   return score;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int countOrders(ulong magic)
  {
   int n_trades=0;
   for(int i= PositionsTotal()-1; i>=0; i--)
     {
      if(PositionGetSymbol(i)==_Symbol)
        {
         if(PositionGetInteger(POSITION_MAGIC)==magic)
           {
            n_trades++;
           }
        }
     }
   return(n_trades);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool isNewBar()
  {
   datetime current_bar_time = iTime(_Symbol, _Period, 0);
   if(last_time != current_bar_time)
     {
      last_time = current_bar_time;
      return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//|  Calcula el lote aplicado a la f-óptima sin depender del SL      |
//+------------------------------------------------------------------+
double LotsOptimized(double sl_points = 0.0)
  {
   if(manual_lot != 0.0)
      return manual_lot;

   COptimalF optf;
   double profits[];
   long positionIDs[];
   double inPrices[];
   double dist[];
   int profitCount = 0, posCount = 0, distCount = 0;

   HistorySelect(0, TimeCurrent());
   int totalDeals = HistoryDealsTotal();
   ArrayResize(profits, totalDeals);
   ArrayResize(positionIDs, totalDeals);
   ArrayResize(inPrices, totalDeals);

// 1. Procesar operaciones y recolectar datos
   for(int i = 0; i < totalDeals; ++i)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(!ticket || HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol)
         continue;
      if((ulong)HistoryDealGetInteger(ticket, DEAL_MAGIC) != MAGIC_NUMBER)
         continue;

      ENUM_DEAL_ENTRY entry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(ticket, DEAL_ENTRY);
      ulong posID = (ulong)HistoryDealGetInteger(ticket, DEAL_POSITION_ID);

      if(entry == DEAL_ENTRY_IN)
        {
         positionIDs[posCount] = (long)posID;
         inPrices[posCount] = HistoryDealGetDouble(ticket, DEAL_PRICE);
         posCount++;
        }
      else
         if(entry == DEAL_ENTRY_OUT)
           {
            double priceClose = HistoryDealGetDouble(ticket, DEAL_PRICE);
            double priceOpen = 0.0;

            for(int j = 0; j < posCount; ++j)
              {
               if(positionIDs[j] == posID)
                 {
                  priceOpen = inPrices[j];
                  break;
                 }
              }

            if(priceOpen != 0.0)
              {
               // Calcular profit
               profits[profitCount++] = HistoryDealGetDouble(ticket, DEAL_PROFIT)
                                        - MathAbs(HistoryDealGetDouble(ticket, DEAL_COMMISSION))
                                        - MathAbs(HistoryDealGetDouble(ticket, DEAL_SWAP));

               // Calcular distancia
               double points = MathAbs(priceOpen - priceClose)/_Point;
               if(distCount >= ArraySize(dist))
                  ArrayResize(dist, distCount + 128);
               dist[distCount++] = points;
              }
           }
     }

// 2. Añadir profits en orden inverso (para mantener consistencia original)
   for(int i = profitCount-1; i >= 0; --i)
      optf.AddProfitTrade(profits[i]);

// 3. Calcular f-óptima
   double f_opt = MathAbs(optf.GeometricMeanOptimalF()) * 0.01;
// 4. Manejar casos sin datos
   if(distCount == 0)
      return SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

// 5. Calcular media en lugar de mediana
   double sum_dist = 0.0;
   int total_elements = ArraySize(dist);
   for(int i = 0; i < total_elements; i++)
     {
      sum_dist += dist[i];
     }
   double med_dist = (total_elements > 0) ? sum_dist / total_elements : 0;
   double used_dist = (sl_points > 0.0) ? sl_points : med_dist;
// 6. Calcular lote base
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double lossPerLot = used_dist * _Point * (tickValue / tickSize);
   if(lossPerLot <= 0.0)
      return SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double lot = (balance * f_opt) / lossPerLot;

// 7. Ajustes finales
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minVol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxVol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   lot = MathFloor(lot/step) * step;
   lot = MathMin(MathMax(lot, minVol), maxVol);

// Validar margen
   double margin, price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(!OrderCalcMargin(ORDER_TYPE_BUY, _Symbol, lot, price, margin) || margin <= 0.0)
      return minVol;

   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   if(margin > freeMargin)
     {
      lot = MathFloor(freeMargin/margin * lot/step) * step;
      lot = MathMax(lot, minVol);
     }

   return lot;
  }

//+------------------------------------------------------------------+
//| Función para imprimir features_main main y meta con timestamp
//+------------------------------------------------------------------+
void print_features_debug(double &features_main[], double &features_meta[], double label_main, double label_meta)
  {
   m_open.Refresh();
   m_high.Refresh();
   m_low.Refresh();
   m_close.Refresh();
   m_vol.Refresh();

   double print_array[];
   int index = 0;
   for(int i = 0; i < ArraySize(features_main); ++i)
     {
      ArrayResize(print_array, index+1);
      print_array[index] = NormalizeDouble(features_main[i], DECIMAL_PRECISION);
      index++;
     }
   for(int i = 0; i < ArraySize(features_meta); ++i)
     {
      ArrayResize(print_array, index+1);
      print_array[index] = NormalizeDouble(features_meta[i], DECIMAL_PRECISION);
      index++;
     }
   ArrayResize(print_array, index+1);
   print_array[index] = NormalizeDouble(m_open.GetData(1), DECIMAL_PRECISION);
   index++;
   ArrayResize(print_array, index+1);
   print_array[index] = NormalizeDouble(m_high.GetData(1), DECIMAL_PRECISION);
   index++;
   ArrayResize(print_array, index+1);
   print_array[index] = NormalizeDouble(m_low.GetData(1), DECIMAL_PRECISION);
   index++;
   ArrayResize(print_array, index+1);
   print_array[index] = NormalizeDouble(m_close.GetData(1), DECIMAL_PRECISION);
   index++;
   ArrayResize(print_array, index+1);
   print_array[index] = NormalizeDouble(double(m_vol.GetData(1)), 0);
   index++;
   ArrayResize(print_array, index+1);
   print_array[index] = NormalizeDouble(label_main, DECIMAL_PRECISION);
   index++;
   ArrayResize(print_array, index+1);
   print_array[index] = NormalizeDouble(label_meta, DECIMAL_PRECISION);
   index++;
   ArrayPrint(print_array, DECIMAL_PRECISION);
  }

//+------------------------------------------------------------------+
//| Walk-Forward Validation - MATCH PYTHON EXACTLY                  |
//+------------------------------------------------------------------+
double WalkForwardValidation(double &equity[], double &trade_profits[], int n_trades)
  {
// Ventanas más largas para mayor robustez
   int base_window = MathMax(10, n_trades / 20);  // Al menos 10, o 5% del total

   if(n_trades < base_window)
      return 0.0;

// Evaluar múltiples escalas temporales con ventanas monótonamente crecientes
// Asegurar que cada ventana sea mayor que la anterior y no exceda el límite
   int window1 = base_window;
   int window2 = MathMin(window1 * 2, n_trades / 8);  // Ajustado para evitar ventanas muy pequeñas
   int window3 = MathMin(window1 * 3, n_trades / 4);  // Ajustado para evitar ventanas muy pequeñas

// Filtrar ventanas válidas (debe ser al menos 5 trades por ventana)
   int windows[];
   double weights[];
   int window_count = 0;

   if(window1 >= 5)
     {
      ArrayResize(windows, window_count + 1);
      ArrayResize(weights, window_count + 1);
      windows[window_count] = window1;
      weights[window_count] = 0.5;
      window_count++;
     }

   if(window2 > window1 && window2 >= 5)
     {
      ArrayResize(windows, window_count + 1);
      ArrayResize(weights, window_count + 1);
      windows[window_count] = window2;
      weights[window_count] = 0.3;
      window_count++;
     }

   if(window3 > window2 && window3 >= 5)
     {
      ArrayResize(windows, window_count + 1);
      ArrayResize(weights, window_count + 1);
      windows[window_count] = window3;
      weights[window_count] = 0.2;
      window_count++;
     }

// Si no hay ventanas válidas, usar solo la base
   if(window_count == 0)
     {
      ArrayResize(windows, 1);
      ArrayResize(weights, 1);
      windows[0] = base_window;
      weights[0] = 1.0;
      window_count = 1;
     }

// Normalizar pesos si hay menos ventanas de las esperadas
   if(window_count > 0)
     {
      double weight_sum = 0.0;
      for(int i = 0; i < window_count; i++)
         weight_sum += weights[i];
      for(int i = 0; i < window_count; i++)
         weights[i] = weights[i] / weight_sum;
     }

   double total_score = 0.0;

   for(int w_idx = 0; w_idx < window_count; w_idx++)
     {
      int window = windows[w_idx];
      double weight = weights[w_idx];

      if(n_trades < window)
         continue;

      int step = MathMax(1, window / 4);  // Solapamiento del 75%
      int wins = 0;
      int total = 0;
      double win_ratios_sum = 0.0;
      int win_ratios_count = 0;
      double window_returns[];
      int window_returns_count = 0;

      for(int start = 0; start <= n_trades - window; start += step)
        {
         int end = start + window;

         // 1) Rentabilidad de la ventana (equity)
         double r = equity[end] - equity[start];
         ArrayResize(window_returns, window_returns_count + 1);
         window_returns[window_returns_count] = r;
         window_returns_count++;

         if(r > 0)
            wins++;
         total++;

         // 2) Ratio de ganadoras/perdedoras en la ventana
         int n_window_trades = end - start;
         if(n_window_trades > 0)
           {
            int n_winners = 0;
            for(int j = start; j < end; j++)
               if(trade_profits[j] > 0)
                  n_winners++;
            double win_ratio = (double)n_winners / n_window_trades;
            win_ratios_sum += win_ratio;
            win_ratios_count++;
           }
        }

      if(total == 0)
         continue;

      double prop_ventanas_rentables = (double)wins / total;
      double avg_win_ratio = win_ratios_count > 0 ? win_ratios_sum / win_ratios_count : 0.0;

      // 3) Penalización por volatilidad entre ventanas
      double stability_penalty = 1.0;
      if(window_returns_count >= 3)
        {
         // Calcular volatilidad de los retornos de ventanas
         double mean_return = 0.0;
         for(int i = 0; i < window_returns_count; i++)
            mean_return += window_returns[i];
         mean_return /= window_returns_count;

         double variance = 0.0;
         for(int i = 0; i < window_returns_count; i++)
           {
            double diff = window_returns[i] - mean_return;
            variance += diff * diff;
           }
         variance /= window_returns_count;

         // Penalizar alta volatilidad (inestabilidad)
         if(variance > 0)
           {
            double cv = MathSqrt(variance) / (MathAbs(mean_return) + 1e-8);  // Coeficiente de variación
            stability_penalty = 1.0 / (1.0 + cv * 2.0);  // Más penalización por alta volatilidad
           }
        }

      // Score para esta ventana
      double window_score = prop_ventanas_rentables * avg_win_ratio * stability_penalty;
      total_score += window_score * weight;
     }

// Aplicar función sigmoide para mayor discriminación
// Penalizar más fuertemente scores bajos
   double final_score = MathPow(total_score, 1.5);  // Exponente > 1 para penalizar más los valores bajos

   return MathMin(1.0, final_score);
  }
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
