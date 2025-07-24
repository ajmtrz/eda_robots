//+------------------------------------------------------------------+
//| Corrected MQL5 Functions to Match Python Implementation          |
//| These functions should replace the corresponding ones in the     |
//| original MQL5 file to ensure consistency with Python results    |
//|                                                                  |
//| MAIN FIXES APPLIED:                                              |
//| 1. OnTester: Fixed linear regression calculation to match Python |
//| 2. OnTester: Fixed walk-forward validation logic                 |
//| 3. OnTester: Fixed minimum trades validation (200 trades)       |
//| 4. OnTester: Fixed equity curve and running max calculations     |
//| 5. OnTick: Fixed position opening timing and delay logic         |
//| 6. OnTick: Fixed trade counter updates to match Python behavior  |
//|                                                                  |
//| CRITICAL: These corrections ensure that equity curves, trade     |
//| counts, win/loss ratios, and final scores match exactly between  |
//| Python and MQL5 implementations.                                 |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Corrected OnTester function - matches Python evaluate_report    |
//+------------------------------------------------------------------+
double OnTester()
{
//── 1) Recoger trades cerrados ----------------------------------
   HistorySelect(0, TimeCurrent());
   int total_deals = HistoryDealsTotal();

   double profits[];
   long   times[];
   int    n_trades = 0;

   for(int i = 0; i < total_deals; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT)
         continue;

      ArrayResize(profits, n_trades + 1);
      ArrayResize(times, n_trades + 1);

      profits[n_trades] = HistoryDealGetDouble(ticket, DEAL_PROFIT) +
                         HistoryDealGetDouble(ticket, DEAL_SWAP) +
                         HistoryDealGetDouble(ticket, DEAL_COMMISSION);

      times[n_trades] = HistoryDealGetInteger(ticket, DEAL_TIME);
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
   {
      running_max[i] = (running_max[i-1] > equity[i]) ? running_max[i-1] : equity[i];
   }
   
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
   {
      rdd = 0.0;
   }
   else
   {
      rdd = total_ret / max_dd;
   }
   
   if(rdd < rdd_floor)
      return -1.0;
      
   double rdd_nl = 1.0 / (1.0 + MathExp(-(rdd - rdd_floor) / (rdd_floor * 5.0)));

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
   
   // Check for negative slope
   if(slope < 0.0)
      return -1.0;

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
   
   double r2;
   if(MathAbs(ss_tot) < 1e-12)
   {
      r2 = 0.0;
   }
   else
   {
      r2 = 1.0 - (ss_res / ss_tot);
   }

   // Normalize slope - MATCH PYTHON
   double slope_nl = 1.0 / (1.0 + MathExp(-(MathLog(1.0 + slope) / 5.0)));

//── 6) Walk-Forward Analysis - CORRECTED TO MATCH PYTHON -----------
   const int window = 5;
   const int step = 1;
   int wins = 0;
   int total_windows = 0;
   double win_ratios_sum = 0.0;
   int win_ratios_count = 0;

   if(n_trades >= window)
   {
      for(int start = 0; start <= n_trades - window; start += step)
      {
         int end = start + window;
         
         // 1) Rentabilidad de la ventana (equity) - MATCH PYTHON
         double r = equity[end] - equity[start];
         if(r > 0.0)
            wins++;
         total_windows++;
         
         // 2) Ratio de ganadoras/perdedoras en la ventana - MATCH PYTHON
         int n_winners = 0;
         for(int j = start; j < end; j++)
         {
            if(profit_ordered[j] > 0.0)
               n_winners++;
         }
         double win_ratio = (double)n_winners / window;
         win_ratios_sum += win_ratio;
         win_ratios_count++;
      }
   }

   double wf_nl = 0.0;
   if(total_windows > 0 && win_ratios_count > 0)
   {
      double prop_ventanas_rentables = (double)wins / total_windows;
      double avg_win_ratio = win_ratios_sum / win_ratios_count;
      wf_nl = prop_ventanas_rentables * avg_win_ratio;
   }

//── 7) Calcular score final - EXACT PYTHON WEIGHTS ----------------
   double score = 0.12 * r2 +
                  0.15 * slope_nl +
                  0.24 * rdd_nl +
                  0.19 * trade_nl +
                  0.30 * wf_nl;

   return score;
}

//+------------------------------------------------------------------+
//| Corrected backtest logic in OnTick to match Python behavior    |
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
   OnnxRun(ExtHandle_main, ONNX_DEFAULT, features_main, main_out, main_out2);
   OnnxRun(ExtHandle_meta, ONNX_DEFAULT, features_meta, meta_out, meta_out2);

   double main_prob = main_out2[0].proba[1];
   double meta_sig  = meta_out2[0].proba[1];

   // EXACT PYTHON LOGIC FOR DIRECTION HANDLING
   double prob_buy, prob_sell;
   int dir_flag;

   if(DIRECTION == "buy")
   {
      prob_buy = main_prob;
      prob_sell = 0.0;
      dir_flag = 0;
   }
   else if(DIRECTION == "sell")
   {
      prob_buy = 0.0;
      prob_sell = main_prob;
      dir_flag = 1;
   }
   else // "both"
   {
      prob_buy = main_prob;
      prob_sell = 1.0 - main_prob;
      dir_flag = 2;
   }

   // EXACT PYTHON SIGNAL LOGIC
   bool buy_sig  = (dir_flag != 1) ? (prob_buy > main_threshold) : false;
   bool sell_sig = (dir_flag != 0) ? (prob_sell > main_threshold) : false;
   bool meta_ok  = (meta_sig > meta_threshold);

   int label = ((buy_sig || sell_sig) && meta_ok) ? 1 : 0;
   if(debug)
      print_features_debug(features_main, features_meta, label);

   //--- 1) CERRAR posiciones cuyas señales hayan desaparecido - EXACT PYTHON LOGIC
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      if(PositionGetSymbol(i) != _Symbol)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) != MAGIC_NUMBER)
         continue;

      ENUM_POSITION_TYPE ptype = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

      bool must_close = false;
      // MATCH PYTHON: Close LONG if !buy_sig, Close SHORT if !sell_sig
      if(ptype == POSITION_TYPE_BUY && !buy_sig)
      {
         must_close = true;
         if(debug)
            Print("🔍 DEBUG - Cerrando BUY: buy_sig=", buy_sig, " prob_buy=", prob_buy, " threshold=", main_threshold);
      }
      else if(ptype == POSITION_TYPE_SELL && !sell_sig)
      {
         must_close = true;
         if(debug)
            Print("🔍 DEBUG - Cerrando SELL: sell_sig=", sell_sig, " prob_sell=", prob_sell, " threshold=", main_threshold);
      }

      if(must_close)
      {
         if(debug)
            Print("🔍 DEBUG - CERRANDO POSICIÓN tipo: ", EnumToString(ptype));
         m_trade.PositionClose(PositionGetString(POSITION_SYMBOL));
         last_trade_bar_index = bar_counter;
      }
   }

   //--- 2) ABRIR nuevas posiciones - EXACT PYTHON LOGIC
   int live_pos = countOrders(MAGIC_NUMBER);
   bool delay_ok = (bar_counter - last_trade_bar_index) >= delay_bars;
   bool pool_ok = (max_orders == 0 || live_pos < max_orders);

   if(meta_ok && delay_ok && pool_ok)
   {
      bool trade_opened_this_bar = false;
      
      // BUY - MATCH PYTHON EXACTLY
      if(buy_sig && (max_orders == 0 || live_pos < max_orders))
      {
         if(debug)
            Print("🔍 DEBUG - ABRIENDO BUY");
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
         live_pos++;
      }

      // SELL - MATCH PYTHON: Check position limit again after potential BUY opening
      if(sell_sig && (max_orders == 0 || live_pos < max_orders))
      {
         if(debug)
            Print("🔍 DEBUG - ABRIENDO SELL");
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
         live_pos++;
      }
      
      // Update last_trade_bar only once per bar, regardless of how many positions opened
      if(trade_opened_this_bar)
         last_trade_bar_index = bar_counter;
   }
}