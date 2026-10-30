import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from numpy.lib.stride_tricks import sliding_window_view

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="5-Day SMA Streak & Connors RSI Trading System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Default Universe: Major Indices, Major ETFs, and Liquid Benchmark Equities
DEFAULT_UNIVERSE = {
    # Major Indices
    "^GSPC": "S&P 500 Index",
    "^DJI": "Dow Jones Industrial Average",
    "^RUT": "Russell 2000 Index",
    "^NDX": "Nasdaq 100 Index",
    # Core Index ETFs
    "SPY": "SPDR S&P 500 ETF",
    "DIA": "SPDR Dow Jones ETF",
    "IWM": "iShares Russell 2000 ETF",
    "QQQ": "Invesco QQQ Trust",
    "MDY": "SPDR S&P MidCap 400 ETF",
    # Major Sector & Asset ETFs
    "XLK": "Technology Select Sector SPDR",
    "XLF": "Financial Select Sector SPDR",
    "XLE": "Energy Select Sector SPDR",
    "XLV": "Health Care Select Sector SPDR",
    "XLI": "Industrial Select Sector SPDR",
    "XLY": "Consumer Discretionary SPDR",
    "XLP": "Consumer Staples SPDR",
    "XLU": "Utilities Select Sector SPDR",
    "XLB": "Materials Select Sector SPDR",
    "SMH": "VanEck Semiconductor ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "GLD": "SPDR Gold Shares",
    # Major Liquid Stocks
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "NVDA": "NVIDIA Corp.",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
    "META": "Meta Platforms Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "XOM": "Exxon Mobil Corp.",
}


# -----------------------------------------------------------------------------
# INDICATOR CALCULATIONS
# -----------------------------------------------------------------------------
def wilder_rsi(series: pd.Series, period: int) -> pd.Series:
    """Calculates J. Welles Wilder's RSI using exponential smoothing (alpha = 1/period)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Handle zero loss (100 RSI) and zero gain/loss (50 RSI)
    rsi = np.where(avg_loss == 0, np.where(avg_gain == 0, 50.0, 100.0), rsi)
    return pd.Series(rsi, index=series.index)


def compute_close_to_close_streak(close: pd.Series) -> pd.Series:
    """Calculates consecutive close-to-close up/down streak for Connors RSI."""
    vals = close.to_numpy(dtype=float)
    n = len(vals)
    streak = np.zeros(n, dtype=float)
    for i in range(1, n):
        if np.isnan(vals[i]) or np.isnan(vals[i - 1]):
            streak[i] = 0.0
        elif vals[i] > vals[i - 1]:
            streak[i] = max(0.0, streak[i - 1]) + 1.0
        elif vals[i] < vals[i - 1]:
            streak[i] = min(0.0, streak[i - 1]) - 1.0
        else:
            streak[i] = 0.0
    return pd.Series(streak, index=close.index)


def compute_percent_rank(series: pd.Series, lookback: int = 100) -> pd.Series:
    """Vectorized 100-bar PercentRank of 1-day returns using NumPy sliding windows."""
    roc = series.pct_change().to_numpy(dtype=float)
    n = len(roc)
    out = np.full(n, np.nan)
    if n <= lookback:
        return pd.Series(out, index=series.index)

    windows = sliding_window_view(roc, lookback + 1)
    prev_returns = windows[:, :-1]
    curr_return = windows[:, -1:]
    valid_mask = ~np.isnan(prev_returns).any(axis=1) & ~np.isnan(curr_return[:, 0])

    ranks = np.full(len(windows), np.nan)
    ranks[valid_mask] = (
        np.sum(prev_returns[valid_mask] < curr_return[valid_mask], axis=1)
        / float(lookback)
        * 100.0
    )
    out[lookback:] = ranks
    return pd.Series(out, index=series.index)


def compute_sma_streak(
    close: pd.Series, sma: pd.Series, cap_at_window: bool = False, sma_len: int = 5
) -> pd.Series:
    """
    Computes consecutive closes above (+1, +2, +3, +4, +5...) or below (-1, -2, -3, -4, -5...)
    the N-day Simple Moving Average.
    """
    c_vals = close.to_numpy(dtype=float)
    s_vals = sma.to_numpy(dtype=float)
    n = len(c_vals)
    streak = np.zeros(n, dtype=float)

    for i in range(n):
        if np.isnan(c_vals[i]) or np.isnan(s_vals[i]):
            streak[i] = np.nan
            continue
        prev = 0.0 if (i == 0 or np.isnan(streak[i - 1])) else streak[i - 1]
        if c_vals[i] > s_vals[i]:
            val = max(0.0, prev) + 1.0
            streak[i] = min(val, float(sma_len)) if cap_at_window else val
        elif c_vals[i] < s_vals[i]:
            val = min(0.0, prev) - 1.0
            streak[i] = max(val, -float(sma_len)) if cap_at_window else val
        else:
            streak[i] = 0.0

    return pd.Series(streak, index=close.index)


def enrich_indicators(
    df: pd.DataFrame, sma_len: int = 5, cap_streak: bool = False
) -> pd.DataFrame:
    """Adds 5-day SMA, SMA Streak, Connors RSI (3,2,100), 2-Day Avg CRSI, and Rolling High/Low Closes."""
    df = df.copy()
    df["SMA"] = df["Close"].rolling(sma_len).mean()
    df["SMA_Dist_Pct"] = ((df["Close"] / df["SMA"]) - 1.0) * 100.0
    df["SMA_Streak"] = compute_sma_streak(
        df["Close"], df["SMA"], cap_at_window=cap_streak, sma_len=sma_len
    )

    # Connors RSI (3, 2, 100) and its 2-Day Average
    df["Close_Streak"] = compute_close_to_close_streak(df["Close"])
    rsi_3 = wilder_rsi(df["Close"], 3)
    streak_rsi_2 = wilder_rsi(df["Close_Streak"], 2)
    pct_rank_100 = compute_percent_rank(df["Close"], 100)

    df["CRSI"] = (rsi_3 + streak_rsi_2 + pct_rank_100) / 3.0
    df["CRSI_2D_Avg"] = df["CRSI"].rolling(2).mean()

    # Rolling Highest / Lowest Closes for Exit Rules
    df["Lowest_Close_3"] = df["Close"].rolling(3).min()
    df["Lowest_Close_5"] = df["Close"].rolling(5).min()
    df["Highest_Close_3"] = df["Close"].rolling(3).max()
    df["Highest_Close_5"] = df["Close"].rolling(5).max()

    return df


# -----------------------------------------------------------------------------
# DATA FETCHING (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker: str, years: int = 21) -> pd.DataFrame:
    """Downloads daily historical OHLCV data via yfinance and normalizes columns."""
    end_date = dt.date.today() + dt.timedelta(days=1)
    start_date = end_date - dt.timedelta(days=int(years * 365.25) + 150)

    raw = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    if not all(col in raw.columns for col in required):
        return pd.DataFrame()

    df = raw[required + (["Volume"] if "Volume" in raw.columns else [])].dropna(
        subset=["Open", "High", "Low", "Close"]
    )
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# -----------------------------------------------------------------------------
# BACKTEST ENGINE
# -----------------------------------------------------------------------------
EXIT_OPTIONS = [
    "1st Close Below 5-Day SMA (Short: 1st Close Above)",
    "3-Day Lowest Close (Short: 3-Day Highest Close)",
    "5-Day Lowest Close (Short: 5-Day Highest Close)",
    "1st Close Above 5-Day SMA (Short: 1st Close Below)",
    "3-Day Highest Close (Short: 3-Day Lowest Close)",
    "5-Day Highest Close (Short: 5-Day Lowest Close)",
    "Fixed Holding Period (N Days)",
]


def run_backtest(
    df: pd.DataFrame,
    backtest_years: int = 20,
    trade_side: str = "Both (Long & Short)",
    entry_style: str = "Momentum / Breakout (Long >= +N, Short <= -N)",
    long_streak_target: int = 5,
    short_streak_target: int = -5,
    use_crsi_filter: bool = False,
    crsi_long_filter: float = 30.0,
    crsi_short_filter: float = 70.0,
    exit_rule: str = EXIT_OPTIONS[0],
    fixed_hold_days: int = 5,
    execution_timing: str = "Close on Signal Day",
    initial_capital: float = 100_000.0,
):
    """
    Runs a bar-by-bar backtest over the specified lookback window.
    Supports both Momentum/Trend and Mean-Reversion streak triggers plus symmetrical Long/Short exits.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    cutoff_date = df.index.max() - pd.DateOffset(years=backtest_years)
    bt_df = df[df.index >= cutoff_date].copy()
    if len(bt_df) < 30:
        return pd.DataFrame(), pd.DataFrame(), {}

    dates = bt_df.index
    opens = bt_df["Open"].to_numpy(dtype=float)
    closes = bt_df["Close"].to_numpy(dtype=float)
    smas = bt_df["SMA"].to_numpy(dtype=float)
    streaks = bt_df["SMA_Streak"].to_numpy(dtype=float)
    crsi_2d = bt_df["CRSI_2D_Avg"].to_numpy(dtype=float)
    low_3 = bt_df["Lowest_Close_3"].to_numpy(dtype=float)
    low_5 = bt_df["Lowest_Close_5"].to_numpy(dtype=float)
    high_3 = bt_df["Highest_Close_3"].to_numpy(dtype=float)
    high_5 = bt_df["Highest_Close_5"].to_numpy(dtype=float)

    n = len(bt_df)
    pos = 0  # +1 for Long, -1 for Short, 0 for Flat
    entry_price = 0.0
    entry_idx = 0
    entry_date = None
    entry_streak = 0.0
    entry_crsi = np.nan

    equity = np.full(n, initial_capital, dtype=float)
    position_series = np.zeros(n, dtype=int)
    trades = []
    cur_equity = initial_capital

    allow_long = trade_side in ("Long Only", "Both (Long & Short)")
    allow_short = trade_side in ("Short Only", "Both (Long & Short)")
    is_momentum = entry_style.startswith("Momentum")

    def check_exit_condition(side: int, i: int, bars_held: int) -> bool:
        if bars_held < 1:
            return False
        c = closes[i]
        s = smas[i]
        if side == 1:  # LONG EXIT RULES
            if exit_rule.startswith("1st Close Below 5-Day SMA"):
                return c < s
            elif exit_rule.startswith("3-Day Lowest Close"):
                return c <= low_3[i]
            elif exit_rule.startswith("5-Day Lowest Close"):
                return c <= low_5[i]
            elif exit_rule.startswith("1st Close Above 5-Day SMA"):
                return c > s
            elif exit_rule.startswith("3-Day Highest Close"):
                return c >= high_3[i]
            elif exit_rule.startswith("5-Day Highest Close"):
                return c >= high_5[i]
            elif exit_rule.startswith("Fixed Holding Period"):
                return bars_held >= fixed_hold_days
        elif side == -1:  # SHORT EXIT RULES (Vice Versa)
            if exit_rule.startswith("1st Close Below 5-Day SMA"):
                return c > s  # Vice versa: 1st Close Above 5-Day SMA
            elif exit_rule.startswith("3-Day Lowest Close"):
                return c >= high_3[i]  # Vice versa: 3-Day Highest Close
            elif exit_rule.startswith("5-Day Lowest Close"):
                return c >= high_5[i]  # Vice versa: 5-Day Highest Close
            elif exit_rule.startswith("1st Close Above 5-Day SMA"):
                return c < s  # Vice versa: 1st Close Below 5-Day SMA
            elif exit_rule.startswith("3-Day Highest Close"):
                return c <= low_3[i]  # Vice versa: 3-Day Lowest Close
            elif exit_rule.startswith("5-Day Highest Close"):
                return c <= low_5[i]  # Vice versa: 5-Day Lowest Close
            elif exit_rule.startswith("Fixed Holding Period"):
                return bars_held >= fixed_hold_days
        return False

    def check_entry_condition(i: int) -> int:
        st_val = streaks[i]
        if np.isnan(st_val) or np.isnan(smas[i]):
            return 0
        c_val = crsi_2d[i]

        if is_momentum:
            long_ok = allow_long and (st_val >= long_streak_target)
            short_ok = allow_short and (st_val <= short_streak_target)
            if use_crsi_filter:
                long_ok = long_ok and (not np.isnan(c_val) and c_val >= crsi_long_filter)
                short_ok = short_ok and (not np.isnan(c_val) and c_val <= crsi_short_filter)
        else:
            # Mean Reversion: Buy negative streak pullback, Short positive streak extension
            long_ok = allow_long and (st_val <= -abs(long_streak_target))
            short_ok = allow_short and (st_val >= abs(short_streak_target))
            if use_crsi_filter:
                long_ok = long_ok and (not np.isnan(c_val) and c_val <= crsi_long_filter)
                short_ok = short_ok and (not np.isnan(c_val) and c_val >= crsi_short_filter)

        if long_ok:
            return 1
        if short_ok:
            return -1
        return 0

    for i in range(1, n):
        # Mark-to-market daily return if holding position from prior bar
        if pos != 0:
            bar_ret = pos * ((closes[i] / closes[i - 1]) - 1.0)
            cur_equity *= 1.0 + bar_ret

        equity[i] = cur_equity
        position_series[i] = pos

        # Check exit if currently in a trade
        if pos != 0:
            bars_held = i - entry_idx
            if check_exit_condition(pos, i, bars_held):
                if execution_timing == "Next Bar Open" and (i + 1) < n:
                    exit_px = opens[i + 1]
                    exit_dt = dates[i + 1]
                    # Adjust equity from today's close to next open
                    gap_ret = pos * ((exit_px / closes[i]) - 1.0)
                    cur_equity *= 1.0 + gap_ret
                    equity[i] = cur_equity
                else:
                    exit_px = closes[i]
                    exit_dt = dates[i]

                trade_ret = pos * ((exit_px / entry_price) - 1.0) * 100.0
                trades.append(
                    {
                        "Direction": "Long" if pos == 1 else "Short",
                        "Entry Date": entry_date.strftime("%Y-%m-%d"),
                        "Entry Price": round(entry_price, 2),
                        "Entry Streak": int(entry_streak),
                        "Entry 2D CRSI": round(entry_crsi, 1) if not np.isnan(entry_crsi) else np.nan,
                        "Exit Date": exit_dt.strftime("%Y-%m-%d"),
                        "Exit Price": round(exit_px, 2),
                        "Bars Held": bars_held,
                        "Return (%)": round(trade_ret, 2),
                    }
                )
                pos = 0
                continue

        # Check entry if flat
        if pos == 0:
            sig = check_entry_condition(i)
            if sig != 0:
                if execution_timing == "Next Bar Open" and (i + 1) < n:
                    entry_price = opens[i + 1]
                    entry_date = dates[i + 1]
                    entry_idx = i + 1
                    # Capture intraday move on bar i+1 during next iteration
                else:
                    entry_price = closes[i]
                    entry_date = dates[i]
                    entry_idx = i
                entry_streak = streaks[i]
                entry_crsi = crsi_2d[i]
                pos = sig
                position_series[i] = pos

    bt_df["Strategy_Equity"] = equity
    bt_df["BuyHold_Equity"] = initial_capital * (bt_df["Close"] / bt_df["Close"].iloc[0])
    bt_df["Position"] = position_series

    trades_df = pd.DataFrame(trades)
    metrics = compute_performance_metrics(bt_df, trades_df, initial_capital)
    return bt_df, trades_df, metrics


def compute_performance_metrics(
    bt_df: pd.DataFrame, trades_df: pd.DataFrame, initial_capital: float
) -> dict:
    """Computes CAGR, Max Drawdown, Win Rate, Profit Factor, Sharpe Ratio, and Expectancy."""
    eq = bt_df["Strategy_Equity"]
    bh = bt_df["BuyHold_Equity"]
    total_days = (bt_df.index[-1] - bt_df.index[0]).days
    years = max(total_days / 365.25, 0.1)

    total_ret = ((eq.iloc[-1] / initial_capital) - 1.0) * 100.0
    bh_ret = ((bh.iloc[-1] / initial_capital) - 1.0) * 100.0
    cagr = (((eq.iloc[-1] / initial_capital) ** (1.0 / years)) - 1.0) * 100.0
    bh_cagr = (((bh.iloc[-1] / initial_capital) ** (1.0 / years)) - 1.0) * 100.0

    # Drawdown
    rolling_max = eq.cummax()
    dd = ((eq - rolling_max) / rolling_max) * 100.0
    max_dd = dd.min()

    # Daily Sharpe (rf = 0 for simplicity)
    daily_rets = eq.pct_change().dropna()
    sharpe = (
        (daily_rets.mean() / daily_rets.std()) * np.sqrt(252)
        if len(daily_rets) > 2 and daily_rets.std() > 0
        else 0.0
    )

    num_trades = len(trades_df)
    if num_trades > 0:
        wins = trades_df[trades_df["Return (%)"] > 0]
        losses = trades_df[trades_df["Return (%)"] <= 0]
        win_rate = (len(wins) / num_trades) * 100.0
        avg_trade = trades_df["Return (%)"].mean()
        avg_win = wins["Return (%)"].mean() if not wins.empty else 0.0
        avg_loss = losses["Return (%)"].mean() if not losses.empty else 0.0
        gross_profit = wins["Return (%)"].sum()
        gross_loss = abs(losses["Return (%)"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        avg_bars = trades_df["Bars Held"].mean()
    else:
        win_rate = avg_trade = avg_win = avg_loss = profit_factor = avg_bars = 0.0

    return {
        "Total Return (%)": round(total_ret, 2),
        "Buy & Hold Return (%)": round(bh_ret, 2),
        "CAGR (%)": round(cagr, 2),
        "Buy & Hold CAGR (%)": round(bh_cagr, 2),
        "Max Drawdown (%)": round(max_dd, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Total Trades": int(num_trades),
        "Win Rate (%)": round(win_rate, 1),
        "Profit Factor": round(profit_factor, 2) if np.isfinite(profit_factor) else 99.9,
        "Avg Trade (%)": round(avg_trade, 2),
        "Avg Win (%)": round(avg_win, 2),
        "Avg Loss (%)": round(avg_loss, 2),
        "Avg Bars Held": round(avg_bars, 1),
    }


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ System Parameters")

selected_ticker = st.sidebar.selectbox(
    "Primary Symbol (Chart & Backtest)",
    options=list(DEFAULT_UNIVERSE.keys()),
    format_func=lambda x: f"{x} — {DEFAULT_UNIVERSE.get(x, '')}",
    index=4,  # Default to SPY
)

custom_ticker = st.sidebar.text_input(
    "Or Enter Custom Ticker", value="", placeholder="e.g., QQQ, TSLA, ^GSPC"
).strip().upper()

active_ticker = custom_ticker if custom_ticker else selected_ticker

st.sidebar.markdown("---")
st.sidebar.subheader("1. Streak & Indicator Settings")
sma_period = st.sidebar.number_input("SMA Period (Days)", min_value=2, max_value=50, value=5)
cap_streak_mode = st.sidebar.checkbox(
    "Cap Streak at ±5 (Rolling 5-Bar Window)",
    value=False,
    help="Unchecked = consecutive streak (+1, +2, +5, +8...). Checked = caps maximum streak at ±SMA Period.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Backtest Entry Rules")
backtest_years = st.sidebar.slider("Backtest Horizon (Years)", min_value=5, max_value=25, value=20)
trade_side = st.sidebar.selectbox(
    "System Direction",
    options=["Long Only", "Short Only", "Both (Long & Short)"],
    index=2,
)
entry_style = st.sidebar.radio(
    "Streak Trigger Mode",
    options=[
        "Mean Reversion (Long <= -N, Short >= +N)",
        "Momentum / Breakout (Long >= +N, Short <= -N)",
    ],
    index=0,
)

col_l, col_s = st.sidebar.columns(2)
with col_l:
    long_streak_target = st.number_input("Long Streak |N|", min_value=1, max_value=15, value=5)
with col_s:
    short_streak_target = st.number_input("Short Streak |N|", min_value=1, max_value=15, value=5)

use_crsi_filter = st.sidebar.checkbox("Filter Entries with 2-Day Avg Connors RSI", value=False)
if use_crsi_filter:
    c1, c2 = st.sidebar.columns(2)
    with c1:
        crsi_long_thresh = st.number_input("Long 2D CRSIThresh", 1.0, 99.0, 25.0)
    with c2:
        crsi_short_thresh = st.number_input("Short 2D CRSIThresh", 1.0, 99.0, 75.0)
else:
    crsi_long_thresh, crsi_short_thresh = 25.0, 75.0

st.sidebar.markdown("---")
st.sidebar.subheader("3. Backtest Exit Rules")
selected_exit = st.sidebar.selectbox("Primary Exit Rule", options=EXIT_OPTIONS, index=0)
fixed_hold_days = 5
if selected_exit.startswith("Fixed Holding Period"):
    fixed_hold_days = st.sidebar.slider("Holding Period (Bars)", 1, 20, 5)

execution_timing = st.sidebar.selectbox(
    "Trade Execution Timing",
    options=["Close on Signal Day", "Next Bar Open"],
    index=0,
)

# -----------------------------------------------------------------------------
# MAIN HEADER & DATA LOAD
# -----------------------------------------------------------------------------
st.title("📊 5-Day SMA Streak & Connors RSI Trading System")
st.caption(
    f"Active Symbol: **{active_ticker}** | 20-Year Backtest Engine • Multi-Exit Matrix • "
    f"5-Year Candlestick & Streak Oscillator • Universe Scanner"
)

with st.spinner(f"Loading {backtest_years}-year daily history for {active_ticker}..."):
    raw_df = fetch_history(active_ticker, years=max(backtest_years + 2, 22))

if raw_df.empty:
    st.error(f"Could not load historical data for `{active_ticker}`. Please check the ticker symbol.")
    st.stop()

full_df = enrich_indicators(raw_df, sma_len=sma_period, cap_streak=cap_streak_mode)
bt_df, trades_df, metrics = run_backtest(
    full_df,
    backtest_years=backtest_years,
    trade_side=trade_side,
    entry_style=entry_style,
    long_streak_target=long_streak_target,
    short_streak_target=-abs(short_streak_target),
    use_crsi_filter=use_crsi_filter,
    crsi_long_filter=crsi_long_thresh,
    crsi_short_filter=crsi_short_thresh,
    exit_rule=selected_exit,
    fixed_hold_days=fixed_hold_days,
    execution_timing=execution_timing,
)

# Current Snapshot Banner for Active Symbol
latest = full_df.iloc[-1]
prev_bar = full_df.iloc[-2] if len(full_df) > 1 else latest
day_chg = ((latest["Close"] / prev_bar["Close"]) - 1.0) * 100.0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Last Close", f"${latest['Close']:,.2f}", f"{day_chg:+.2f}%")
m2.metric(f"{sma_period}-Day SMA", f"${latest['SMA']:,.2f}", f"{latest['SMA_Dist_Pct']:+.2f}% vs SMA")
m3.metric(
    f"Current {sma_period}-SMA Streak",
    f"{int(latest['SMA_Streak']):+d}" if not np.isnan(latest["SMA_Streak"]) else "0",
)
m4.metric("Connors RSI (3,2,100)", f"{latest['CRSI']:.1f}")
m5.metric("2-Day Avg Connors RSI", f"{latest['CRSI_2D_Avg']:.1f}")

# -----------------------------------------------------------------------------
# NAVIGATION TABS
# -----------------------------------------------------------------------------
tab_chart, tab_bt, tab_scanner = st.tabs(
    [
        "📈 5-Year Chart & Streak Oscillator",
        f"🧪 {backtest_years}-Year Backtest & All-Exits Matrix",
        "🌐 Universe Scanner (Streaks & 2-Day CRSI)",
    ]
)

# =============================================================================
# TAB 1: 5-YEAR DAILY CANDLESTICK CHART + STREAK OSCILLATOR SUBPLOT
# =============================================================================
with tab_chart:
    c_top1, c_top2, c_top3 = st.columns([2, 2, 2])
    with c_top1:
        chart_years = st.slider("Chart Lookback (Years)", min_value=1, max_value=10, value=5)
    with c_top2:
        show_trade_markers = st.checkbox("Overlay Backtest Entry/Exit Markers", value=True)
    with c_top3:
        show_crsi_subplot = st.checkbox("Include 2-Day Avg Connors RSI Panel", value=False)

    chart_cutoff = full_df.index.max() - pd.DateOffset(years=chart_years)
    c_df = full_df[full_df.index >= chart_cutoff].copy()

    num_rows = 3 if show_crsi_subplot else 2
    row_heights = [0.58, 0.24, 0.18] if show_crsi_subplot else [0.68, 0.32]
    subplot_titles = [
        f"{active_ticker} — {chart_years}-Year Daily Candlesticks & {sma_period}-Day SMA",
        f"{sma_period}-Day SMA Streak Oscillator",
    ]
    if show_crsi_subplot:
        subplot_titles.append("2-Day Average Connors RSI (3, 2, 100)")

    fig = make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Row 1: Daily Candlesticks + 5-Day SMA
    fig.add_trace(
        go.Candlestick(
            x=c_df.index,
            open=c_df["Open"],
            high=c_df["High"],
            low=c_df["Low"],
            close=c_df["Close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=c_df.index,
            y=c_df["SMA"],
            mode="lines",
            name=f"{sma_period}-Day SMA",
            line=dict(color="#ff9800", width=1.6),
        ),
        row=1,
        col=1,
    )

    # Optional Entry/Exit Trade Markers
    if show_trade_markers and not trades_df.empty:
        t_sub = trades_df.copy()
        t_sub["Entry_DT"] = pd.to_datetime(t_sub["Entry Date"])
        t_sub["Exit_DT"] = pd.to_datetime(t_sub["Exit Date"])
        t_sub = t_sub[t_sub["Entry_DT"] >= chart_cutoff]

        if not t_sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=t_sub["Entry_DT"],
                    y=t_sub["Entry Price"],
                    mode="markers",
                    name="Trade Entry",
                    marker=dict(symbol="triangle-up", size=10, color="#00e676", line=dict(width=1, color="black")),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_sub["Exit_DT"],
                    y=t_sub["Exit Price"],
                    mode="markers",
                    name="Trade Exit",
                    marker=dict(symbol="triangle-down", size=10, color="#ffeb3b", line=dict(width=1, color="black")),
                ),
                row=1,
                col=1,
            )

    # Row 2: Streak Oscillator (Histogram Bars + Oscillator Line)
    streak_colors = np.where(c_df["SMA_Streak"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(
        go.Bar(
            x=c_df.index,
            y=c_df["SMA_Streak"],
            name="SMA Streak Histogram",
            marker_color=streak_colors,
            opacity=0.65,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=c_df.index,
            y=c_df["SMA_Streak"],
            mode="lines",
            name="Streak Oscillator",
            line=dict(color="#90caf9", width=1.2),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=long_streak_target, line_dash="dash", line_color="#26a69a", row=2, col=1)
    fig.add_hline(y=-abs(short_streak_target), line_dash="dash", line_color="#ef5350", row=2, col=1)
    fig.add_hline(y=0, line_color="#78909c", line_width=1, row=2, col=1)

    # Optional Row 3: 2-Day Average Connors RSI
    if show_crsi_subplot:
        fig.add_trace(
            go.Scatter(
                x=c_df.index,
                y=c_df["CRSI_2D_Avg"],
                mode="lines",
                name="2-Day Avg CRSI",
                line=dict(color="#ce93d8", width=1.5),
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=80, line_dash="dot", line_color="#ef5350", row=3, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="#26a69a", row=3, col=1)

    fig.update_layout(
        height=760 if show_crsi_subplot else 660,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Streak (Bars)", row=2, col=1)
    if show_crsi_subplot:
        fig.update_yaxes(title_text="2D CRSI", range=[0, 100], row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: 20-YEAR BACKTEST & MULTI-EXIT COMPARISON MATRIX
# =============================================================================
with tab_bt:
    st.subheader(f"Performance Summary ({active_ticker} — {backtest_years} Years)")
    st.markdown(f"**Active Exit Rule:** `{selected_exit}` | **Entry Style:** `{entry_style}`")

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Strategy CAGR", f"{metrics.get('CAGR (%)', 0):.2f}%", f"B&H: {metrics.get('Buy & Hold CAGR (%)', 0):.2f}%")
    k2.metric("Total Return", f"{metrics.get('Total Return (%)', 0):,.1f}%", f"B&H: {metrics.get('Buy & Hold Return (%)', 0):,.1f}%")
    k3.metric("Win Rate", f"{metrics.get('Win Rate (%)', 0):.1f}%", f"{metrics.get('Total Trades', 0)} Trades")
    k4.metric("Profit Factor", f"{metrics.get('Profit Factor', 0):.2f}", f"Avg Trade: {metrics.get('Avg Trade (%)', 0):+.2f}%")
    k5.metric("Max Drawdown", f"{metrics.get('Max Drawdown (%)', 0):.2f}%")
    k6.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}", f"Avg Hold: {metrics.get('Avg Bars Held', 0):.1f}d")

    # Equity Curve Plot
    eq_fig = go.Figure()
    eq_fig.add_trace(
        go.Scatter(
            x=bt_df.index,
            y=bt_df["Strategy_Equity"],
            mode="lines",
            name=f"Strategy ({selected_exit.split(' (')[0]})",
            line=dict(color="#00e676", width=2.2),
        )
    )
    eq_fig.add_trace(
        go.Scatter(
            x=bt_df.index,
            y=bt_df["BuyHold_Equity"],
            mode="lines",
            name=f"Buy & Hold ({active_ticker})",
            line=dict(color="#90a4ae", width=1.5, dash="dot"),
        )
    )
    eq_fig.update_layout(
        title=f"{backtest_years}-Year Equity Growth ($100,000 Initial Capital)",
        height=420,
        template="plotly_dark",
        hovermode="x unified",
        yaxis_title="Portfolio Equity ($)",
        margin=dict(l=40, r=40, t=50, b=30),
    )
    st.plotly_chart(eq_fig, use_container_width=True)

    # Side-by-Side Comparison of All Exit Rules
    st.subheader("🔄 Side-by-Side Exit Strategy Comparison (20-Year Matrix)")
    st.caption(
        "Automatically benchmarks every exit variation (3-Day Lowest Close, 5-Day Lowest Close, "
        "1st Close Below/Above 5-Day SMA, 3/5-Day Highest Close, Fixed N-Day) using your current entry settings."
    )

    exit_comparison_rows = []
    for ex_opt in EXIT_OPTIONS:
        _, _, ex_m = run_backtest(
            full_df,
            backtest_years=backtest_years,
            trade_side=trade_side,
            entry_style=entry_style,
            long_streak_target=long_streak_target,
            short_streak_target=-abs(short_streak_target),
            use_crsi_filter=use_crsi_filter,
            crsi_long_filter=crsi_long_thresh,
            crsi_short_filter=crsi_short_thresh,
            exit_rule=ex_opt,
            fixed_hold_days=fixed_hold_days,
            execution_timing=execution_timing,
        )
        if ex_m:
            row_data = {"Exit System": ex_opt}
            row_data.update(ex_m)
            exit_comparison_rows.append(row_data)

    if exit_comparison_rows:
        comp_df = pd.DataFrame(exit_comparison_rows)
        st.dataframe(
            comp_df[
                [
                    "Exit System",
                    "Total Trades",
                    "Win Rate (%)",
                    "Avg Trade (%)",
                    "Profit Factor",
                    "CAGR (%)",
                    "Total Return (%)",
                    "Max Drawdown (%)",
                    "Sharpe Ratio",
                    "Avg Bars Held",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("📋 Trade-by-Trade Execution Log")
    if not trades_df.empty:
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    else:
        st.info("No trades triggered under the current streak and filter thresholds.")

# =============================================================================
# TAB 3: UNIVERSE SCANNER (SORTABLE STREAK & 2-DAY AVG CONNORS RSI TABLE)
# =============================================================================
with tab_scanner:
    st.subheader("🌐 Multi-Asset Universe Scanner")
    st.caption(
        "Click any column header to sort by Current 5-Day SMA Streak, 2-Day Average Connors RSI, "
        "or Distance from the 5-Day SMA."
    )

    default_tickers_str = ", ".join(DEFAULT_UNIVERSE.keys())
    universe_input = st.text_area(
        "Universe Tickers (Comma-Separated — Add or Remove Symbols freely)",
        value=default_tickers_str,
        height=70,
    )
    scan_tickers = [t.strip().upper() for t in universe_input.split(",") if t.strip()]

    @st.cache_data(ttl=1800, show_spinner=False)
    def run_universe_scan(tickers: list, sma_p: int, cap_s: bool) -> pd.DataFrame:
        end_dt = dt.date.today() + dt.timedelta(days=1)
        start_dt = end_dt - dt.timedelta(days=260)  # ~180 trading bars (>100 needed for CRSI)
        raw_batch = yf.download(
            tickers,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            auto_adjust=True,
            group_by="ticker",
            progress=False,
        )
        rows = []
        for tk in tickers:
            try:
                if len(tickers) == 1:
                    t_df = raw_batch.copy()
                else:
                    if tk not in raw_batch.columns.get_level_values(0):
                        continue
                    t_df = raw_batch[tk].copy()

                t_df = t_df.dropna(subset=["Close"])
                if len(t_df) < 105:
                    continue

                ind_df = enrich_indicators(t_df, sma_len=sma_p, cap_streak=cap_s)
                last = ind_df.iloc[-1]
                prev = ind_df.iloc[-2]
                ret_1d = ((last["Close"] / prev["Close"]) - 1.0) * 100.0
                st_val = int(last["SMA_Streak"]) if not np.isnan(last["SMA_Streak"]) else 0
                crsi_2d_val = float(last["CRSI_2D_Avg"])

                if st_val <= -5 and crsi_2d_val <= 25:
                    status = "🟢 Strong Oversold (Streak <= -5 & 2D CRSI <= 25)"
                elif st_val >= 5 and crsi_2d_val >= 75:
                    status = "🔴 Strong Overbought (Streak >= +5 & 2D CRSI >= 75)"
                elif st_val <= -5:
                    status = "🟢 Negative Streak <= -5"
                elif st_val >= 5:
                    status = "🟠 Positive Streak >= +5"
                else:
                    status = "⚪ Neutral"

                rows.append(
                    {
                        "Ticker": tk,
                        "Name": DEFAULT_UNIVERSE.get(tk, "Custom Symbol"),
                        "Last Close": round(float(last["Close"]), 2),
                        "1D Chg (%)": round(ret_1d, 2),
                        f"{sma_p}-Day SMA": round(float(last["SMA"]), 2),
                        "% vs SMA": round(float(last["SMA_Dist_Pct"]), 2),
                        f"{sma_p}-SMA Streak": st_val,
                        "2-Day Avg CRSI": round(crsi_2d_val, 2),
                        "Connors RSI (Today)": round(float(last["CRSI"]), 2),
                        "Close Up/Down Streak": int(last["Close_Streak"]),
                        "Signal Watch": status,
                    }
                )
            except Exception:
                continue

        return pd.DataFrame(rows)

    with st.spinner(f"Scanning {len(scan_tickers)} symbols for 5-Day SMA Streaks & 2-Day Avg Connors RSI..."):
        scan_df = run_universe_scan(scan_tickers, sma_period, cap_streak_mode)

    if not scan_df.empty:
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            sort_col = st.selectbox(
                "Default Sort Column",
                options=[
                    "2-Day Avg CRSI",
                    f"{sma_period}-SMA Streak",
                    "% vs SMA",
                    "1D Chg (%)",
                    "Ticker",
                ],
                index=1,
            )
        with f_col2:
            sort_asc = st.radio("Sort Order", options=["Ascending (Lowest First)", "Descending (Highest First)"], horizontal=True)

        scan_df = scan_df.sort_values(
            by=sort_col, ascending=sort_asc.startswith("Ascending")
        ).reset_index(drop=True)

        st.dataframe(
            scan_df,
            use_container_width=True,
            hide_index=True,
            height=600,
            column_config={
                "2-Day Avg CRSI": st.column_config.ProgressColumn(
                    "2-Day Avg CRSI",
                    help="2-Day Simple Average of Connors RSI (3, 2, 100)",
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
                f"{sma_period}-SMA Streak": st.column_config.NumberColumn(
                    f"{sma_period}-SMA Streak",
                    help="Consecutive daily closes above (+) or below (-) the 5-Day SMA",
                    format="%+d",
                ),
            },
        )
    else:
        st.warning("No valid symbols returned from Yahoo Finance.")
