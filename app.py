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
    """Calculates Wilder's RSI using exponential smoothing (alpha = 1/period)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
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
    """Vectorized 100-bar PercentRank of 1-day returns."""
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
    df: pd.DataFrame, sma_len: int = 5, exit_lookback_x: int = 5, cap_streak: bool = False
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

    # Rolling Highest / Lowest Closes for Exit Rules (3-day, 5-day, and custom X-day)
    df["Highest_Close_X"] = df["Close"].rolling(exit_lookback_x).max()
    df["Lowest_Close_X"] = df["Close"].rolling(exit_lookback_x).min()
    df["Highest_Close_3"] = df["Close"].rolling(3).max()
    df["Lowest_Close_3"] = df["Close"].rolling(3).min()
    df["Highest_Close_5"] = df["Close"].rolling(5).max()
    df["Lowest_Close_5"] = df["Close"].rolling(5).min()

    return df


# -----------------------------------------------------------------------------
# DATA FETCHING (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker: str, years: int = 21) -> pd.DataFrame:
    """Downloads daily historical OHLCV data via yfinance."""
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
# BACKTEST ENGINE (EXACT NEXT-DAY OPEN EXECUTION)
# -----------------------------------------------------------------------------
EXIT_OPTIONS = [
    "1st Close Above SMA (Short: 1st Close Below SMA)",
    "X-Day Highest Close (Short: X-Day Lowest Close)",
    "1st Close Above SMA OR X-Day Highest Close (Short: Vice-Versa)",
    "3-Day Highest Close (Short: 3-Day Lowest Close)",
    "5-Day Highest Close (Short: 5-Day Lowest Close)",
    "1st Close Below SMA (Short: 1st Close Above SMA)",
    "3-Day Lowest Close (Short: 3-Day Highest Close)",
    "5-Day Lowest Close (Short: 5-Day Highest Close)",
    "Fixed Holding Period (X Days)",
]


def run_backtest(
    df: pd.DataFrame,
    backtest_years: int = 20,
    trade_side: str = "Both (Long & Short)",
    entry_style: str = "Mean Reversion (Long: Streak <= -X, Short: Streak >= +X)",
    streak_match_mode: str = "At Least X Days (|Streak| >= X)",
    streak_x: int = 5,
    exit_lookback_x: int = 5,
    use_crsi_filter: bool = False,
    crsi_long_filter: float = 25.0,
    crsi_short_filter: float = 75.0,
    exit_rule: str = EXIT_OPTIONS[0],
    initial_capital: float = 100_000.0,
):
    """
    Exact Next-Bar Open State Machine:
    1. Signal evaluated at Day t Close.
    2. Entry executed at Day t+1 Open.
    3. Exit condition evaluated at Day k Close (for k >= t+1).
    4. Exit executed at Day k+1 Open.
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

    high_x = bt_df["Highest_Close_X"].to_numpy(dtype=float)
    low_x = bt_df["Lowest_Close_X"].to_numpy(dtype=float)
    high_3 = bt_df["Highest_Close_3"].to_numpy(dtype=float)
    low_3 = bt_df["Lowest_Close_3"].to_numpy(dtype=float)
    high_5 = bt_df["Highest_Close_5"].to_numpy(dtype=float)
    low_5 = bt_df["Lowest_Close_5"].to_numpy(dtype=float)

    n = len(bt_df)
    equity = np.full(n, initial_capital, dtype=float)
    position_series = np.zeros(n, dtype=int)
    trades = []

    allow_long = trade_side in ("Long Only", "Both (Long & Short)")
    allow_short = trade_side in ("Short Only", "Both (Long & Short)")
    is_mean_rev = entry_style.startswith("Mean Reversion")
    exact_match = streak_match_mode.startswith("Exact")

    def check_entry_signal_at_close(i: int) -> int:
        st_val = streaks[i]
        if np.isnan(st_val) or np.isnan(smas[i]):
            return 0
        c_val = crsi_2d[i]

        if is_mean_rev:
            # Long when streak is negative (-X), Short when streak is positive (+X)
            long_cond = (st_val == -streak_x) if exact_match else (st_val <= -streak_x)
            short_cond = (st_val == streak_x) if exact_match else (st_val >= streak_x)
            long_ok = allow_long and long_cond
            short_ok = allow_short and short_cond
            if use_crsi_filter:
                long_ok = long_ok and (not np.isnan(c_val) and c_val <= crsi_long_filter)
                short_ok = short_ok and (not np.isnan(c_val) and c_val >= crsi_short_filter)
        else:
            # Momentum: Long when streak is +X, Short when streak is -X
            long_cond = (st_val == streak_x) if exact_match else (st_val >= streak_x)
            short_cond = (st_val == -streak_x) if exact_match else (st_val <= -streak_x)
            long_ok = allow_long and long_cond
            short_ok = allow_short and short_cond
            if use_crsi_filter:
                long_ok = long_ok and (not np.isnan(c_val) and c_val >= crsi_long_filter)
                short_ok = short_ok and (not np.isnan(c_val) and c_val <= crsi_short_filter)

        if long_ok:
            return 1
        if short_ok:
            return -1
        return 0

    def check_exit_signal_at_close(side: int, i: int, bars_held: int) -> bool:
        c = closes[i]
        s = smas[i]
        if side == 1:  # LONG EXIT AT CLOSE (To Exit at Tomorrow's Open)
            if exit_rule.startswith("1st Close Above SMA ("):
                return c > s
            elif exit_rule.startswith("X-Day Highest Close"):
                return c >= high_x[i]
            elif exit_rule.startswith("1st Close Above SMA OR X-Day Highest Close"):
                return (c > s) or (c >= high_x[i])
            elif exit_rule.startswith("3-Day Highest Close"):
                return c >= high_3[i]
            elif exit_rule.startswith("5-Day Highest Close"):
                return c >= high_5[i]
            elif exit_rule.startswith("1st Close Below SMA"):
                return c < s
            elif exit_rule.startswith("3-Day Lowest Close"):
                return c <= low_3[i]
            elif exit_rule.startswith("5-Day Lowest Close"):
                return c <= low_5[i]
            elif exit_rule.startswith("Fixed Holding Period"):
                return bars_held >= exit_lookback_x
        elif side == -1:  # SHORT EXIT AT CLOSE (Vice-Versa, To Exit at Tomorrow's Open)
            if exit_rule.startswith("1st Close Above SMA ("):
                return c < s  # Vice-versa: 1st Close Below SMA
            elif exit_rule.startswith("X-Day Highest Close"):
                return c <= low_x[i]  # Vice-versa: X-Day Lowest Close
            elif exit_rule.startswith("1st Close Above SMA OR X-Day Highest Close"):
                return (c < s) or (c <= low_x[i])
            elif exit_rule.startswith("3-Day Highest Close"):
                return c <= low_3[i]  # Vice-versa: 3-Day Lowest Close
            elif exit_rule.startswith("5-Day Highest Close"):
                return c <= low_5[i]  # Vice-versa: 5-Day Lowest Close
            elif exit_rule.startswith("1st Close Below SMA"):
                return c > s  # Vice-versa: 1st Close Above SMA
            elif exit_rule.startswith("3-Day Lowest Close"):
                return c >= high_3[i]  # Vice-versa: 3-Day Highest Close
            elif exit_rule.startswith("5-Day Lowest Close"):
                return c >= high_5[i]  # Vice-versa: 5-Day Highest Close
            elif exit_rule.startswith("Fixed Holding Period"):
                return bars_held >= exit_lookback_x
        return False

    # State variables
    pos = 0
    pending_entry_side = 0
    pending_exit = False

    entry_price = 0.0
    entry_idx = 0
    entry_date = None
    signal_date = None
    entry_streak = 0.0
    entry_crsi = np.nan
    realized_equity = initial_capital

    for i in range(n):
        # 1. PROCESS ORDERS AT TODAY'S OPEN (Bar i Open)
        if pos != 0 and pending_exit:
            exit_px = opens[i]
            exit_dt = dates[i]
            trade_ret_frac = pos * ((exit_px - entry_price) / entry_price)
            realized_equity = realized_equity * (1.0 + trade_ret_frac)

            trades.append(
                {
                    "Direction": "Long" if pos == 1 else "Short",
                    "Signal Date (Close)": signal_date.strftime("%Y-%m-%d"),
                    "Entry Date (Open)": entry_date.strftime("%Y-%m-%d"),
                    "Entry Open ($)": round(entry_price, 2),
                    "Signal Streak": int(entry_streak),
                    "Signal 2D CRSI": round(entry_crsi, 1) if not np.isnan(entry_crsi) else np.nan,
                    "Exit Date (Open)": exit_dt.strftime("%Y-%m-%d"),
                    "Exit Open ($)": round(exit_px, 2),
                    "Trading Days Held": i - entry_idx,
                    "Return (%)": round(trade_ret_frac * 100.0, 2),
                }
            )
            pos = 0
            pending_exit = False

        if pos == 0 and pending_entry_side != 0:
            pos = pending_entry_side
            pending_entry_side = 0
            entry_price = opens[i]
            entry_date = dates[i]
            entry_idx = i

        # 2. UPDATE MARK-TO-MARKET EQUITY AT TODAY'S CLOSE (Bar i Close)
        if pos != 0:
            unrealized_ret_frac = pos * ((closes[i] - entry_price) / entry_price)
            equity[i] = realized_equity * (1.0 + unrealized_ret_frac)
        else:
            equity[i] = realized_equity

        position_series[i] = pos

        # 3. EVALUATE SIGNALS AT TODAY'S CLOSE (For Execution at Tomorrow's Open, i+1)
        if pos != 0:
            bars_held = (i - entry_idx) + 1  # 1 at the close of entry day
            if check_exit_signal_at_close(pos, i, bars_held):
                pending_exit = True
        else:
            sig = check_entry_signal_at_close(i)
            if sig != 0:
                pending_entry_side = sig
                signal_date = dates[i]
                entry_streak = streaks[i]
                entry_crsi = crsi_2d[i]

    bt_df["Strategy_Equity"] = equity
    bt_df["BuyHold_Equity"] = initial_capital * (bt_df["Close"] / bt_df["Close"].iloc[0])
    bt_df["Position"] = position_series

    trades_df = pd.DataFrame(trades)
    metrics = compute_performance_metrics(bt_df, trades_df, initial_capital)

    # Capture any live signal triggered on the most recent bar for Tomorrow's Open
    if pending_exit and pos != 0:
        metrics["Tomorrow_Action"] = f"EXIT {'LONG' if pos == 1 else 'SHORT'} AT TOMORROW'S OPEN"
    elif pending_entry_side == 1:
        metrics["Tomorrow_Action"] = "ENTER LONG AT TOMORROW'S OPEN"
    elif pending_entry_side == -1:
        metrics["Tomorrow_Action"] = "ENTER SHORT AT TOMORROW'S OPEN"
    elif pos != 0:
        metrics["Tomorrow_Action"] = f"HOLD {'LONG' if pos == 1 else 'SHORT'} (Entered {entry_date.strftime('%Y-%m-%d')} @ ${entry_price:.2f})"
    else:
        metrics["Tomorrow_Action"] = "FLAT / NO SIGNAL FOR TOMORROW'S OPEN"

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

    rolling_max = eq.cummax()
    dd = ((eq - rolling_max) / rolling_max) * 100.0
    max_dd = dd.min()

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
        avg_bars = trades_df["Trading Days Held"].mean()
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
        "Avg Days Held": round(avg_bars, 1),
    }


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ Strategy Parameters")

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
st.sidebar.subheader("1. Entry Logic (Close Signal → Next Open)")
sma_period = st.sidebar.number_input("SMA Period (Days)", min_value=2, max_value=50, value=5)
streak_x = st.sidebar.number_input(
    "Entry Streak Threshold (X Days)",
    min_value=1,
    max_value=20,
    value=5,
    help="If Streak reaches -X (for Long) or +X (for Short) at today's close, enter at tomorrow's open.",
)
streak_match_mode = st.sidebar.selectbox(
    "Streak Trigger Condition",
    options=[
        "At Least X Days (|Streak| >= X)",
        "Exact Xth Day Only (|Streak| == X)",
    ],
    index=0,
)
cap_streak_mode = st.sidebar.checkbox(
    "Cap Streak Oscillator at ±SMA Period",
    value=False,
)

backtest_years = st.sidebar.slider("Backtest Horizon (Years)", min_value=5, max_value=25, value=20)
trade_side = st.sidebar.selectbox(
    "System Direction",
    options=["Long Only", "Short Only", "Both (Long & Short)"],
    index=2,
)
entry_style = st.sidebar.radio(
    "Strategy Mode",
    options=[
        "Mean Reversion (Long: Streak <= -X, Short: Streak >= +X)",
        "Momentum / Trend (Long: Streak >= +X, Short: Streak <= -X)",
    ],
    index=0,
)

use_crsi_filter = st.sidebar.checkbox("Filter Entries with 2-Day Avg Connors RSI", value=False)
if use_crsi_filter:
    c1, c2 = st.sidebar.columns(2)
    with c1:
        crsi_long_thresh = st.number_input("Long 2D CRSI <=", 1.0, 99.0, 25.0)
    with c2:
        crsi_short_thresh = st.number_input("Short 2D CRSI >=", 1.0, 99.0, 75.0)
else:
    crsi_long_thresh, crsi_short_thresh = 25.0, 75.0

st.sidebar.markdown("---")
st.sidebar.subheader("2. Exit Logic (Close Signal → Next Open)")
selected_exit = st.sidebar.selectbox("Primary Exit Rule (Executed at Next Open)", options=EXIT_OPTIONS, index=0)
exit_lookback_x = st.sidebar.number_input(
    "Exit Lookback (X Days for X-Day High/Low Close)",
    min_value=2,
    max_value=30,
    value=int(streak_x),
    help="Used for 'X-Day Highest Close (Short: X-Day Lowest Close)' and Fixed Holding Period.",
)

# -----------------------------------------------------------------------------
# MAIN HEADER & DATA LOAD
# -----------------------------------------------------------------------------
st.title("📊 5-Day SMA Streak & Connors RSI Trading System")
st.caption(
    f"Active Symbol: **{active_ticker}** | **Execution:** Enter at Tomorrow's Open after {streak_x}-Day Streak • "
    f"Exit at Next Open following Exit Signal Close"
)

with st.spinner(f"Loading {backtest_years}-year daily history for {active_ticker}..."):
    raw_df = fetch_history(active_ticker, years=max(backtest_years + 2, 22))

if raw_df.empty:
    st.error(f"Could not load historical data for `{active_ticker}`. Please check the ticker symbol.")
    st.stop()

full_df = enrich_indicators(
    raw_df, sma_len=sma_period, exit_lookback_x=exit_lookback_x, cap_streak=cap_streak_mode
)
bt_df, trades_df, metrics = run_backtest(
    full_df,
    backtest_years=backtest_years,
    trade_side=trade_side,
    entry_style=entry_style,
    streak_match_mode=streak_match_mode,
    streak_x=streak_x,
    exit_lookback_x=exit_lookback_x,
    use_crsi_filter=use_crsi_filter,
    crsi_long_filter=crsi_long_thresh,
    crsi_short_filter=crsi_short_thresh,
    exit_rule=selected_exit,
)

# Live Snapshot Banner
latest = full_df.iloc[-1]
prev_bar = full_df.iloc[-2] if len(full_df) > 1 else latest
day_chg = ((latest["Close"] / prev_bar["Close"]) - 1.0) * 100.0

m1, m2, m3, m4, m5, m6 = st.columns([1.1, 1.1, 1.1, 1.1, 1.1, 2.0])
m1.metric("Last Close", f"${latest['Close']:,.2f}", f"{day_chg:+.2f}%")
m2.metric(f"{sma_period}-Day SMA", f"${latest['SMA']:,.2f}", f"{latest['SMA_Dist_Pct']:+.2f}% vs SMA")
m3.metric(
    f"Current {sma_period}-SMA Streak",
    f"{int(latest['SMA_Streak']):+d}" if not np.isnan(latest["SMA_Streak"]) else "0",
)
m4.metric(f"{exit_lookback_x}D High / Low Close", f"${latest['Highest_Close_X']:,.2f}", f"Low: ${latest['Lowest_Close_X']:,.2f}")
m5.metric("2-Day Avg CRSI", f"{latest['CRSI_2D_Avg']:.1f}", f"Today: {latest['CRSI']:.1f}")
m6.metric("Next Open Status", metrics.get("Tomorrow_Action", "FLAT"))

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
        show_trade_markers = st.checkbox("Overlay Next-Open Entry/Exit Fills", value=True)
    with c_top3:
        show_crsi_subplot = st.checkbox("Include 2-Day Avg Connors RSI Panel", value=False)

    chart_cutoff = full_df.index.max() - pd.DateOffset(years=chart_years)
    c_df = full_df[full_df.index >= chart_cutoff].copy()

    num_rows = 3 if show_crsi_subplot else 2
    row_heights = [0.58, 0.24, 0.18] if show_crsi_subplot else [0.68, 0.32]
    subplot_titles = [
        f"{active_ticker} — {chart_years}-Year Daily Candlesticks & {sma_period}-Day SMA",
        f"{sma_period}-Day SMA Streak Oscillator (±{streak_x} Trigger Thresholds)",
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

    # Optional Entry/Exit Trade Markers (Plotted at Actual Next-Open Fill Dates & Prices)
    if show_trade_markers and not trades_df.empty:
        t_sub = trades_df.copy()
        t_sub["Entry_DT"] = pd.to_datetime(t_sub["Entry Date (Open)"])
        t_sub["Exit_DT"] = pd.to_datetime(t_sub["Exit Date (Open)"])
        t_sub = t_sub[t_sub["Entry_DT"] >= chart_cutoff]

        if not t_sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=t_sub["Entry_DT"],
                    y=t_sub["Entry Open ($)"],
                    mode="markers",
                    name="Entry Fill (Open)",
                    marker=dict(symbol="triangle-up", size=10, color="#00e676", line=dict(width=1, color="black")),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=t_sub["Exit_DT"],
                    y=t_sub["Exit Open ($)"],
                    mode="markers",
                    name="Exit Fill (Open)",
                    marker=dict(symbol="triangle-down", size=10, color="#ffeb3b", line=dict(width=1, color="black")),
                ),
                row=1,
                col=1,
            )

    # Row 2: Streak Oscillator
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
    fig.add_hline(y=streak_x, line_dash="dash", line_color="#26a69a", row=2, col=1)
    fig.add_hline(y=-streak_x, line_dash="dash", line_color="#ef5350", row=2, col=1)
    fig.add_hline(y=0, line_color="#78909c", line_width=1, row=2, col=1)

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
    fig.update_yaxes(title_text="Streak (Days)", row=2, col=1)
    if show_crsi_subplot:
        fig.update_yaxes(title_text="2D CRSI", range=[0, 100], row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2: 20-YEAR BACKTEST & MULTI-EXIT COMPARISON MATRIX
# =============================================================================
with tab_bt:
    st.subheader(f"Performance Summary ({active_ticker} — {backtest_years} Years)")
    st.markdown(
        f"**Entry:** Streak = `±{streak_x}` at Close → **Enter at Tomorrow's Open** | "
        f"**Exit:** `{selected_exit}` at Close → **Exit at Next Open**"
    )

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Strategy CAGR", f"{metrics.get('CAGR (%)', 0):.2f}%", f"B&H: {metrics.get('Buy & Hold CAGR (%)', 0):.2f}%")
    k2.metric("Total Return", f"{metrics.get('Total Return (%)', 0):,.1f}%", f"B&H: {metrics.get('Buy & Hold Return (%)', 0):,.1f}%")
    k3.metric("Win Rate", f"{metrics.get('Win Rate (%)', 0):.1f}%", f"{metrics.get('Total Trades', 0)} Trades")
    k4.metric("Profit Factor", f"{metrics.get('Profit Factor', 0):.2f}", f"Avg Trade: {metrics.get('Avg Trade (%)', 0):+.2f}%")
    k5.metric("Max Drawdown", f"{metrics.get('Max Drawdown (%)', 0):.2f}%")
    k6.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}", f"Avg Hold: {metrics.get('Avg Days Held', 0):.1f}d")

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
        title=f"{backtest_years}-Year Equity Curve ($100,000 Initial Capital — Next-Open Fills)",
        height=420,
        template="plotly_dark",
        hovermode="x unified",
        yaxis_title="Portfolio Equity ($)",
        margin=dict(l=40, r=40, t=50, b=30),
    )
    st.plotly_chart(eq_fig, use_container_width=True)

    st.subheader("🔄 Side-by-Side Exit Strategy Comparison (All Exits Executed at Next Open)")
    exit_comparison_rows = []
    for ex_opt in EXIT_OPTIONS:
        _, _, ex_m = run_backtest(
            full_df,
            backtest_years=backtest_years,
            trade_side=trade_side,
            entry_style=entry_style,
            streak_match_mode=streak_match_mode,
            streak_x=streak_x,
            exit_lookback_x=exit_lookback_x,
            use_crsi_filter=use_crsi_filter,
            crsi_long_filter=crsi_long_thresh,
            crsi_short_filter=crsi_short_thresh,
            exit_rule=ex_opt,
        )
        if ex_m:
            row_data = {"Exit System (Next Open Fill)": ex_opt.replace("X-Day", f"{exit_lookback_x}-Day").replace("X Days", f"{exit_lookback_x} Days")}
            row_data.update(ex_m)
            exit_comparison_rows.append(row_data)

    if exit_comparison_rows:
        comp_df = pd.DataFrame(exit_comparison_rows)
        st.dataframe(
            comp_df[
                [
                    "Exit System (Next Open Fill)",
                    "Total Trades",
                    "Win Rate (%)",
                    "Avg Trade (%)",
                    "Avg Win (%)",
                    "Avg Loss (%)",
                    "Profit Factor",
                    "CAGR (%)",
                    "Total Return (%)",
                    "Max Drawdown (%)",
                    "Sharpe Ratio",
                    "Avg Days Held",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("📋 Trade-by-Trade Execution Log (Signal Close → Entry Open → Exit Open)")
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
        "or Tomorrow's Open Action Signal."
    )

    default_tickers_str = ", ".join(DEFAULT_UNIVERSE.keys())
    universe_input = st.text_area(
        "Universe Tickers (Comma-Separated — Add or Remove Symbols freely)",
        value=default_tickers_str,
        height=70,
    )
    scan_tickers = [t.strip().upper() for t in universe_input.split(",") if t.strip()]

    @st.cache_data(ttl=1800, show_spinner=False)
    def run_universe_scan(
        tickers: list, sma_p: int, st_x: int, ex_x: int, cap_s: bool
    ) -> pd.DataFrame:
        end_dt = dt.date.today() + dt.timedelta(days=1)
        start_dt = end_dt - dt.timedelta(days=260)
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

                ind_df = enrich_indicators(
                    t_df, sma_len=sma_p, exit_lookback_x=ex_x, cap_streak=cap_s
                )
                last = ind_df.iloc[-1]
                prev = ind_df.iloc[-2]
                ret_1d = ((last["Close"] / prev["Close"]) - 1.0) * 100.0
                st_val = int(last["SMA_Streak"]) if not np.isnan(last["SMA_Streak"]) else 0
                crsi_2d_val = float(last["CRSI_2D_Avg"])

                if st_val <= -st_x:
                    status = f"🟢 BUY AT TOMORROW'S OPEN (Streak = {st_val})"
                elif st_val >= st_x:
                    status = f"🔴 SHORT AT TOMORROW'S OPEN (Streak = +{st_val})"
                else:
                    status = "⚪ No Entry Signal"

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
                        f"{ex_x}D High Close": round(float(last["Highest_Close_X"]), 2),
                        f"{ex_x}D Low Close": round(float(last["Lowest_Close_X"]), 2),
                        "Signal for Tomorrow's Open": status,
                    }
                )
            except Exception:
                continue

        return pd.DataFrame(rows)

    with st.spinner(f"Scanning {len(scan_tickers)} symbols for {sma_period}-Day SMA Streaks & 2-Day Avg Connors RSI..."):
        scan_df = run_universe_scan(
            scan_tickers, sma_period, streak_x, exit_lookback_x, cap_streak_mode
        )

    if not scan_df.empty:
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            sort_col = st.selectbox(
                "Default Sort Column",
                options=[
                    f"{sma_period}-SMA Streak",
                    "2-Day Avg CRSI",
                    "% vs SMA",
                    "1D Chg (%)",
                    "Ticker",
                ],
                index=0,
            )
        with f_col2:
            sort_asc = st.radio(
                "Sort Order",
                options=["Ascending (Lowest First)", "Descending (Highest First)"],
                horizontal=True,
            )

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
                    help="Consecutive daily closes above (+) or below (-) the SMA",
                    format="%+d",
                ),
            },
        )
    else:
        st.warning("No valid symbols returned from Yahoo Finance.")
