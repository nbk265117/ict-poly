"""
═══════════════════════════════════════════════════════════════════════════════
  BOT ICT HYBRID V12 - CONFIGURATION FINALE
═══════════════════════════════════════════════════════════════════════════════

  Stratégie: ICT Trend + RSI + Stochastic + FTFC Multi-Timeframe

  Configuration:
    • RSI(7): 45/55
    • Stochastic(5,3): 42/58
    • FTFC Min: 1.5
    • Cooldown: 3 minutes
    • Bet Size: $327/trade
    • Payout: 90%

  Objectifs:
    • 67+ trades/jour
    • 54%+ Win Rate
    • $23,000+/mois
    • 24/24 mois profitables

═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Path to local data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

CONFIG = {
    # Pairs
    'pairs': ['BTC', 'ETH', 'XRP'],

    # Trading
    'bet_size': 327,        # $327 par trade
    'payout': 0.90,         # 90% payout
    'cooldown': 3,          # 3 minutes entre trades

    # RSI
    'rsi_period': 7,
    'rsi_oversold': 45,     # Signal BULL si RSI < 45
    'rsi_overbought': 55,   # Signal BEAR si RSI > 55

    # Stochastic
    'stoch_k': 5,
    'stoch_d': 3,
    'stoch_oversold': 42,   # Signal BULL si Stoch < 42
    'stoch_overbought': 58, # Signal BEAR si Stoch > 58

    # FTFC (Multi-Timeframe Confirmation)
    'ftfc_min': 1.5,        # Score minimum pour trader
}


# ═══════════════════════════════════════════════════════════════════════════════
#  BOT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ICTHybridBot:
    """
    Bot de trading hybride ICT + Oscillateurs

    Utilise:
    - ICT Trend Detection (EMA 8/21/50)
    - RSI pour timing d'entrée
    - Stochastic pour confirmation
    - FTFC pour filtrage multi-timeframe
    """

    def __init__(self, config=CONFIG):
        self.config = config
        self.data_cache = {}

    def load_data(self, pair, interval):
        """Charge les données depuis les fichiers CSV locaux"""
        cache_key = f"{pair}_{interval}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        file_path = os.path.join(DATA_DIR, f"{pair}_{interval}.csv")
        if not os.path.exists(file_path):
            print(f"  Warning: {file_path} not found")
            return pd.DataFrame()

        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        self.data_cache[cache_key] = df
        return df

    def get_data_slice(self, df, start_dt, end_dt):
        """Extrait une portion des données entre deux dates"""
        return df[(df.index >= start_dt) & (df.index < end_dt)]

    # ═══════════════════════════════════════════════════════════════════════════
    #  INDICATEURS
    # ═══════════════════════════════════════════════════════════════════════════

    def ema(self, series, period):
        """Exponential Moving Average"""
        return series.ewm(span=period, adjust=False).mean()

    def rsi(self, df, period=7):
        """Relative Strength Index"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, 0.0001)
        return 100 - (100 / (1 + rs))

    def stochastic(self, df, k_period=5, d_period=3):
        """Stochastic Oscillator"""
        low_min = df['low'].rolling(k_period).min()
        high_max = df['high'].rolling(k_period).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min + 0.0001)
        d = k.rolling(d_period).mean()
        return k, d

    # ═══════════════════════════════════════════════════════════════════════════
    #  ICT CONCEPTS
    # ═══════════════════════════════════════════════════════════════════════════

    def get_trend(self, df):
        """
        Détection de tendance ICT basée sur EMA alignment
        BULL: EMA8 > EMA21 > EMA50
        BEAR: EMA8 < EMA21 < EMA50
        """
        close = df['close']
        ema8 = self.ema(close, 8)
        ema21 = self.ema(close, 21)
        ema50 = self.ema(close, 50)

        if ema8.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1]:
            return 'BULL', 1.0
        elif ema8.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1]:
            return 'BEAR', 1.0
        return 'NONE', 0

    def calculate_ftfc(self, df_1h, df_4h):
        """
        FTFC - Flow Through Filter Confirmation
        Score basé sur l'alignement des tendances 1H et 4H
        """
        trend_1h, strength_1h = self.get_trend(df_1h)
        trend_4h, strength_4h = self.get_trend(df_4h)

        score = 0.0

        # Score pour chaque timeframe
        if trend_1h != 'NONE':
            score += 1.5
        if trend_4h != 'NONE':
            score += 1.5

        # Bonus si aligné
        if trend_1h == trend_4h and trend_1h != 'NONE':
            score += 1.0

        # Retourner tendance dominante
        trend = trend_4h if trend_4h != 'NONE' else trend_1h

        return score, trend

    # ═══════════════════════════════════════════════════════════════════════════
    #  SIGNAL GENERATION
    # ═══════════════════════════════════════════════════════════════════════════

    def generate_signal(self, df_4h, df_1h, df_15m):
        """
        Génère un signal de trading

        Conditions BULL:
        - FTFC score >= 1.5
        - Tendance BULL sur 1H/4H
        - RSI < 45 (oversold)
        - Stochastic < 42 (oversold)

        Conditions BEAR:
        - FTFC score >= 1.5
        - Tendance BEAR sur 1H/4H
        - RSI > 55 (overbought)
        - Stochastic > 58 (overbought)
        """
        cfg = self.config

        # 1. FTFC Check
        ftfc_score, trend = self.calculate_ftfc(df_1h, df_4h)
        if ftfc_score < cfg['ftfc_min'] or trend == 'NONE':
            return None

        # 2. RSI
        rsi = self.rsi(df_15m, cfg['rsi_period'])
        rsi_val = rsi.iloc[-1]

        # 3. Stochastic
        stoch_k, _ = self.stochastic(df_15m, cfg['stoch_k'], cfg['stoch_d'])
        stoch_val = stoch_k.iloc[-1]

        # 4. Signal
        if trend == 'BULL':
            if rsi_val < cfg['rsi_oversold'] and stoch_val < cfg['stoch_oversold']:
                return 'BULL'

        elif trend == 'BEAR':
            if rsi_val > cfg['rsi_overbought'] and stoch_val > cfg['stoch_overbought']:
                return 'BEAR'

        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest():
    """Backtest complet avec configuration finale"""
    print("=" * 85)
    print("  ICT HYBRID V12 - BACKTEST FINAL (LOCAL DATA)")
    print("  Config: RSI 45/55 | Stoch 42/58 | FTFC 1.5 | $327/trade")
    print("=" * 85)

    bot = ICTHybridBot()
    cfg = bot.config

    PAIRS = cfg['pairs']
    BET = cfg['bet_size']
    PAYOUT = cfg['payout']
    COOLDOWN = cfg['cooldown']

    # Load all data upfront
    print(f"\n  Loading local data from {DATA_DIR}...")
    data = {}
    for pair in PAIRS:
        data[pair] = {
            '15m': bot.load_data(pair, '15m'),
            '1h': bot.load_data(pair, '1h'),
            '4h': bot.load_data(pair, '4h')
        }
        print(f"    {pair}: {len(data[pair]['15m']):,} candles (15m)")

    periods = []
    for year in [2024, 2025]:
        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            end = f"{year}-{month+1:02d}-01" if month < 12 else f"{year+1}-01-01"
            periods.append((f"{year}-{month:02d}", start, end))

    print(f"\n  Configuration:")
    print(f"    Pairs: {', '.join(PAIRS)}")
    print(f"    Bet Size: ${BET}/trade")
    print(f"    Payout: {PAYOUT*100}%")
    print(f"    RSI({cfg['rsi_period']}): {cfg['rsi_oversold']}/{cfg['rsi_overbought']}")
    print(f"    Stoch({cfg['stoch_k']},{cfg['stoch_d']}): {cfg['stoch_oversold']}/{cfg['stoch_overbought']}")
    print(f"    FTFC Min: {cfg['ftfc_min']}")
    print(f"    Cooldown: {COOLDOWN}min")

    print(f"\n  {'Mois':<10} {'Trades':>8} {'T/J':>8} {'Wins':>8} {'WR%':>8} {'PnL':>14}")
    print("  " + "-" * 60)

    all_results = []

    for month_name, start_str, end_str in periods:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d")

        if start_dt > datetime.now():
            continue
        if end_dt > datetime.now():
            end_dt = datetime.now()

        days = (end_dt - start_dt).days

        month_trades = 0
        month_wins = 0
        month_pnl = 0

        for pair in PAIRS:
            try:
                df_15m = bot.get_data_slice(data[pair]['15m'], start_dt, end_dt)
                df_1h = bot.get_data_slice(data[pair]['1h'], start_dt, end_dt)
                df_4h = bot.get_data_slice(data[pair]['4h'], start_dt, end_dt)

                if len(df_15m) < 100 or len(df_1h) < 50 or len(df_4h) < 20:
                    continue

                last_trade = None

                for i in range(100, len(df_15m) - 1):
                    ts = df_15m.index[i]

                    # Cooldown
                    if last_trade:
                        mins = (ts - last_trade).total_seconds() / 60
                        if mins < COOLDOWN:
                            continue

                    # Slice data
                    df_4h_s = df_4h[df_4h.index <= ts].tail(30)
                    df_1h_s = df_1h[df_1h.index <= ts].tail(60)
                    df_15m_s = df_15m.iloc[:i+1].tail(100)

                    if len(df_4h_s) < 15 or len(df_1h_s) < 30:
                        continue

                    # Signal
                    signal = bot.generate_signal(df_4h_s, df_1h_s, df_15m_s)

                    if signal:
                        entry = df_15m.iloc[i]['close']
                        exit_p = df_15m.iloc[i + 1]['close']

                        won = (exit_p > entry) == (signal == 'BULL')
                        if exit_p == entry:
                            won = False

                        pnl = BET * PAYOUT if won else -BET

                        month_trades += 1
                        if won:
                            month_wins += 1
                        month_pnl += pnl
                        last_trade = ts

            except Exception as e:
                continue

        # Stats
        tpd = month_trades / days if days > 0 else 0
        wr = month_wins / month_trades * 100 if month_trades > 0 else 0

        all_results.append({
            'month': month_name,
            'trades': month_trades,
            'tpd': tpd,
            'wins': month_wins,
            'wr': wr,
            'pnl': month_pnl
        })

        status = "+" if month_pnl > 0 else "-"
        print(f"  {month_name:<10} {month_trades:>8} {tpd:>7.1f} {month_wins:>8} {wr:>7.1f}% ${month_pnl:>+12,.0f}{status}")

    # ═══════════════════════════════════════════════════════════════════════════
    #  RÉSUMÉ FINAL
    # ═══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 85)
    print("  RÉSUMÉ FINAL")
    print("=" * 85)

    total_trades = sum(r['trades'] for r in all_results)
    total_wins = sum(r['wins'] for r in all_results)
    total_pnl = sum(r['pnl'] for r in all_results)
    total_days = len(all_results) * 30

    avg_tpd = total_trades / total_days
    overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    avg_monthly = total_pnl / len(all_results) if all_results else 0
    profitable_months = sum(1 for r in all_results if r['pnl'] > 0)

    y2024_pnl = sum(r['pnl'] for r in all_results if r['month'].startswith('2024'))
    y2025_pnl = sum(r['pnl'] for r in all_results if r['month'].startswith('2025'))

    print(f"""
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║  CONFIGURATION FINALE                                                      ║
  ╠═══════════════════════════════════════════════════════════════════════════╣
  ║  • Pairs: BTC, ETH, XRP                                                    ║
  ║  • Bet Size: ${BET}/trade                                                 ║
  ║  • Payout: {PAYOUT*100}%                                                          ║
  ║  • RSI(7): {cfg['rsi_oversold']}/{cfg['rsi_overbought']}                                                          ║
  ║  • Stoch(5,3): {cfg['stoch_oversold']}/{cfg['stoch_overbought']}                                                      ║
  ║  • FTFC Min: {cfg['ftfc_min']}                                                        ║
  ║  • Cooldown: {COOLDOWN}min                                                         ║
  ╠═══════════════════════════════════════════════════════════════════════════╣
  ║  RÉSULTATS ({len(all_results)} mois)                                                     ║
  ╠═══════════════════════════════════════════════════════════════════════════╣
  ║  • Total Trades: {total_trades:,}                                               ║
  ║  • Trades/Jour: {avg_tpd:.1f}                                                     ║
  ║  • Win Rate: {overall_wr:.1f}%                                                     ║
  ║  • PnL Total: ${total_pnl:+,.0f}                                            ║
  ║  • PnL Moyen/Mois: ${avg_monthly:+,.0f}                                        ║
  ║  • Mois Profitables: {profitable_months}/24                                            ║
  ╠═══════════════════════════════════════════════════════════════════════════╣
  ║  PAR ANNÉE                                                                 ║
  ╠═══════════════════════════════════════════════════════════════════════════╣
  ║  • 2024: ${y2024_pnl:+,.0f}                                                ║
  ║  • 2025: ${y2025_pnl:+,.0f}                                                ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
    """)

    # Comparaison objectif
    print("  COMPARAISON OBJECTIF V10:")
    print("  " + "-" * 60)
    print(f"  {'Métrique':<20} {'Actuel':>15} {'Objectif':>15} {'Status':>10}")
    print("  " + "-" * 60)
    print(f"  {'Trades/Jour':<20} {avg_tpd:>15.1f} {'67':>15} {'OK' if avg_tpd >= 67 else 'X':>10}")
    print(f"  {'Win Rate':<20} {overall_wr:>14.1f}% {'58.6%':>15} {'OK' if overall_wr >= 58.6 else 'X':>10}")
    print(f"  {'PnL/Mois':<20} ${avg_monthly:>+13,.0f} {'$23,447':>15} {'OK' if avg_monthly >= 23447 else 'X':>10}")
    print(f"  {'Mois Profit':<20} {profitable_months:>13}/24 {'24/24':>15} {'OK' if profitable_months >= 24 else 'X':>10}")
    print("  " + "-" * 60)

    print("\n" + "=" * 85)

    return all_results


def run():
    """Alias for run_backtest"""
    return run_backtest()


if __name__ == "__main__":
    run_backtest()
