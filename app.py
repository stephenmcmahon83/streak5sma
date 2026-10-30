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

PRESET_UNIVERSES = {
    "Major Indices & Core ETFs": [
        "SPY", "QQQ", "IWM", "DIA", "MDY",
        "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "SMH", "TLT", "GLD",
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM"
    ],
    "Nasdaq 100": [
        "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
        "AMZN", "ANSS", "ARM", "ASML", "AZN", "BAX", "BIIB", "BKNG", "BKR", "CDNS",
        "CEG", "CHTR", "CMCSA", "COST", "CRWD", "CSCO", "CSX", "CTAS", "CTSH", "DASH",
        "DDOG", "DLTR", "DXCM", "EA", "EXC", "FANG", "FAST", "FTNT", "GEHC", "GFS",
        "GILD", "GOOG", "GOOGL", "HON", "IDXX", "ILMN", "INTC", "INTU", "ISRG", "KDP",
        "KHC", "KLAC", "LCID", "LULU", "MAR", "MCHP", "MDLZ", "MELI", "META", "MNST",
        "MRNA", "MSFT", "MU", "NFLX", "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW",
        "PAYX", "PCAR", "PDD", "PEP", "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX",
        "SIRI", "SNPS", "SNOW", "SPLK", "STM", "STX", "TEAM", "TMUS", "TSLA", "TTD",
        "TTWO", "TXN", "VRSK", "VRTX", "WBA", "WBD", "WDAY", "XEL", "ZS"
    ],
    "S&P 500 (Full Universe)": [],
    "Russell 2000 (Sample Liquid)": [
        "IWM", "AA", "AMC", "ARRY", "BCPC", "BLBD", "CACI", "CALM", "CAR", "CBT",
        "CCK", "CONN", "CROX", "CVI", "CWEN", "DNOW", "EEX", "ENR", "EXLS", "FIBK",
        "FOSL", "GATX", "GEF", "GIII", "HAI", "HBT", "HELE", "HIFS", "HRB", "HURN",
        "IBTX", "IDCC", "JBT", "KAI", "KFY", "KW", "LAD", "MHO", "NWE", "OGS"
    ]
}


@st.cache_data(ttl=86400)
def fetch_sp500_tickers() -> list:
    try:
        url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
        df = pd.read_csv(url)
        if "Symbol" in df.columns:
            return df["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        pass
    return ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "V", "UNH", "XOM", "JNJ", "MA", "PG", "HD"]


# -----------------------------------------------------------------------------
# INDICATOR CALCULATIONS
# -----------------------------------------------------------------------------
def wilder_rsi(series: pd.Series, period: int) -> pd.Series:
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
    df = df.copy()
    df["SMA"] = df["Close"].rolling(sma_len).mean()
    df["SMA_Dist_Pct"] = ((df["Close"] / df["SMA"]) - 1.0) * 100.0
    df["SMA_Streak"] = compute_sma_streak(
        df["Close"], df["SMA"], cap_at_window=cap_streak, sma_len=sma_len
    )

    df["Close_Streak"] = compute_close_to_close_streak(df["Close"])
    rsi_3 = wilder_rsi(df["Close"], 3)
    streak_rsi_2 = wilder_rsi(df["Close_Streak"], 2)
    pct_rank_100 = compute_percent_rank(df["Close"], 100)

    df["CRSI"] = (rsi_3 + streak_rsi_2 + pct_rank_100) / 3.0
    df["CRSI_2D_Avg"] = df["CRSI"].rolling(2).mean()

    df["Highest_Close_X"] = df["Close"].rolling(exit_lookback_x).max()
    df["Lowest_Close_X"] = df["Close"].rolling(exit_lookback_x).min()
    df["Highest_Close_3"] = df["Close"].rolling(3).max()
    df["Lowest_Close_3"] = df["Close"].rolling(3).min()
    df["Highest_Close_5"] = df["Close"].rolling(5).max()
    df["Lowest_Close_5"] = df["Close"].rolling(5).min()

    next_open = df["Open"].shift(-1)
    future_open_5 = df["Open"].shift(-6)
    df["Fwd_5D_Return"] = ((future_open_5 / next_open) - 1.0) * 100.0

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker: str, years: int = 25) -> pd.DataFrame:
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

def compute_monthly_annual_returns_table(equity_series: pd.Series) -> pd.DataFrame:
    """Computes a stable table of monthly and annual returns (%) from an equity series."""
    equity_series = equity_series.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    m_eq = equity_series.resample("ME").last()
    m_ret = m_eq.pct_change().fillna(0.0)
    
    # Clip extreme outlier percentages to prevent table distortion
    m_ret = m_ret.clip(lower=-0.99, upper=5.0)

    df = pd.DataFrame({"ret": m_ret})
    df["year"] = df.index.year
    df["month"] = df.index.strftime("%b")

    pivot = df.pivot_table(index="year", columns="month", values="ret", aggfunc="first")
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    existing_cols = [m for m in month_order if m in pivot.columns]
    pivot = pivot[existing_cols] * 100.0

    annual_rets = []
    for yr in pivot.index:
        yr_subset = equity_series[equity_series.index.year == yr]
        if len(yr_subset) >= 2:
            start_val = yr_subset.iloc[0]
            end_val = yr_subset.iloc[-1]
            if start_val > 0:
                ann_ret = ((end_val / start_val) - 1.0) * 100.0
                annual_rets.append(max(min(ann_ret, 999.9), -99.9))
            else:
                annual_rets.append(0.0)
        else:
            annual_rets.append(0.0)

    pivot["Annual (%)"] = annual_rets
    return pivot.fillna(0.0).round(2)



# -----------------------------------------------------------------------------
# BACKTEST ENGINE (SINGLE SYMBOL)
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
    backtest_years: int = 25,
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
    commission_per_trade: float = 1.0,
):
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
            long_cond = (st_val == -streak_x) if exact_match else (st_val <= -streak_x)
            short_cond = (st_val == streak_x) if exact_match else (st_val >= streak_x)
            long_ok = allow_long and long_cond
            short_ok = allow_short and short_cond
            if use_crsi_filter:
                long_ok = long_ok and (not np.isnan(c_val) and c_val <= crsi_long_filter)
                short_ok = short_ok and (not np.isnan(c_val) and c_val >= crsi_short_filter)
        else:
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
        if side == 1:
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
        elif side == -1:
            if exit_rule.startswith("1st Close Above SMA ("):
                return c < s
            elif exit_rule.startswith("X-Day Highest Close"):
                return c <= low_x[i]
            elif exit_rule.startswith("1st Close Above SMA OR X-Day Highest Close"):
                return (c < s) or (c <= low_x[i])
            elif exit_rule.startswith("3-Day Highest Close"):
                return c <= low_3[i]
            elif exit_rule.startswith("5-Day Highest Close"):
                return c <= low_5[i]
            elif exit_rule.startswith("1st Close Below SMA"):
                return c > s
            elif exit_rule.startswith("3-Day Lowest Close"):
                return c >= high_3[i]
            elif exit_rule.startswith("5-Day Lowest Close"):
                return c >= high_5[i]
            elif exit_rule.startswith("Fixed Holding Period"):
                return bars_held >= exit_lookback_x
        return False

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
        if pos != 0 and pending_exit:
            exit_px = opens[i]
            exit_dt = dates[i]
            trade_ret_dollar = pos * (exit_px - entry_price) * (realized_equity / entry_price)
            realized_equity = realized_equity + trade_ret_dollar - (2.0 * commission_per_trade)
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
                    "Return (%)": round(pos * ((exit_px - entry_price) / entry_price) * 100.0, 2),
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

        if pos != 0:
            unrealized_ret_dollar = pos * (closes[i] - entry_price) * (realized_equity / entry_price)
            equity[i] = realized_equity + unrealized_ret_dollar
        else:
            equity[i] = realized_equity

        position_series[i] = pos

        if pos != 0:
            bars_held = (i - entry_idx) + 1
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
    return bt_df, trades_df, metrics


def compute_performance_metrics(bt_df: pd.DataFrame, trades_df: pd.DataFrame, initial_capital: float) -> dict:
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
    "Primary Symbol (Chart View)",
    options=["SPY", "QQQ", "IWM", "^GSPC", "^DJI", "^NDX", "AAPL", "MSFT", "NVDA", "TSLA"],
    index=0,
)

custom_ticker = st.sidebar.text_input(
    "Or Enter Custom Ticker", value="", placeholder="e.g., AAPL, TSLA"
).strip().upper()
active_ticker = custom_ticker if custom_ticker else selected_ticker

st.sidebar.markdown("---")
st.sidebar.subheader("1. Capital & Portfolio Sizing")
initial_capital = st.sidebar.number_input(
    "Initial Portfolio Capital ($)",
    min_value=10_000.0,
    max_value=10_000_000.0,
    value=100_000.0,
    step=10_000.0,
    help="Starting dollar size of the portfolio."
)
max_portfolio_slots = st.sidebar.number_input(
    "Max Portfolio Size (Max Concurrent Stocks)",
    min_value=1, max_value=50, value=5,
    help="Limits how many simultaneous stock positions can be held."
)
commission_per_trade = st.sidebar.number_input(
    "Commission ($ Per Trade Entry/Exit)",
    min_value=0.0, max_value=20.0, value=1.0,
    help="Fixed commission deducted in dollars per entry and exit fill."
)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Entry Logic")
sma_period = st.sidebar.number_input("SMA Period (Days)", min_value=2, max_value=50, value=5)
streak_x = st.sidebar.number_input("Entry Streak Threshold (X Days)", min_value=1, max_value=20, value=5)
streak_match_mode = st.sidebar.selectbox("Streak Trigger Condition", options=["At Least X Days (|Streak| >= X)", "Exact Xth Day Only (|Streak| == X)"], index=0)
cap_streak_mode = st.sidebar.checkbox("Cap Streak Oscillator at ±SMA Period", value=False)

backtest_years = st.sidebar.slider("Backtest Horizon (Years)", min_value=5, max_value=30, value=25)
trade_side = st.sidebar.selectbox("System Direction", options=["Long Only", "Short Only", "Both (Long & Short)"], index=2)
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
st.sidebar.subheader("3. Exit Logic")
selected_exit = st.sidebar.selectbox("Primary Exit Rule", options=EXIT_OPTIONS, index=0)
exit_lookback_x = st.sidebar.number_input("Exit Lookback (X Days)", min_value=2, max_value=30, value=int(streak_x))


# -----------------------------------------------------------------------------
# MAIN HEADER & DATA LOAD
# -----------------------------------------------------------------------------
st.title("📊 5-Day SMA Streak & Connors RSI Trading System")
st.caption(f"Active Symbol: **{active_ticker}** | Initial Capital: **${initial_capital:,.0f}** | Horizon: **{backtest_years} Years**")

with st.spinner(f"Loading daily history for {active_ticker}..."):
    raw_df = fetch_history(active_ticker, years=max(backtest_years + 2, 28))

if raw_df.empty:
    st.error(f"Could not load historical data for `{active_ticker}`.")
    st.stop()

full_df = enrich_indicators(raw_df, sma_len=sma_period, exit_lookback_x=exit_lookback_x, cap_streak=cap_streak_mode)
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
    initial_capital=initial_capital,
    commission_per_trade=commission_per_trade,
)

latest = full_df.iloc[-1]
prev_bar = full_df.iloc[-2] if len(full_df) > 1 else latest
day_chg = ((latest["Close"] / prev_bar["Close"]) - 1.0) * 100.0

m1, m2, m3, m4, m5, m6 = st.columns([1.1, 1.1, 1.1, 1.1, 1.1, 2.0])
m1.metric("Last Close", f"${latest['Close']:,.2f}", f"{day_chg:+.2f}%")
m2.metric(f"{sma_period}-Day SMA", f"${latest['SMA']:,.2f}", f"{latest['SMA_Dist_Pct']:+.2f}% vs SMA")
m3.metric(f"Current Streak", f"{int(latest['SMA_Streak']):+d}" if not np.isnan(latest["SMA_Streak"]) else "0")
m4.metric(f"{exit_lookback_x}D High / Low", f"${latest['Highest_Close_X']:,.2f}", f"Low: ${latest['Lowest_Close_X']:,.2f}")
m5.metric("2-Day Avg CRSI", f"{latest['CRSI_2D_Avg']:.1f}")
m6.metric("Next Open Status", metrics.get("Tomorrow_Action", "FLAT"))


# -----------------------------------------------------------------------------
# NAVIGATION TABS
# -----------------------------------------------------------------------------
tab_chart, tab_bt, tab_scanner = st.tabs(
    [
        "📈 Chart & Full Backtest Horizon",
        f"🧪 {backtest_years}-Year Portfolio & Universe Backtest",
        "🌐 Universe Scanner (Edge Analysis & Forward Returns)",
    ]
)

# =============================================================================
# TAB 1: DAILY CANDLESTICK CHART
# =============================================================================
with tab_chart:
    c_top1, c_top2, c_top3 = st.columns([2, 2, 2])
    with c_top1:
        chart_years = st.slider("Chart Lookback (Years)", min_value=1, max_value=backtest_years, value=min(5, backtest_years))
    with c_top2:
        show_trade_markers = st.checkbox("Overlay Trade Fills", value=True)
    with c_top3:
        show_crsi_subplot = st.checkbox("Include 2-Day Avg Connors RSI Panel", value=False)

    chart_cutoff = full_df.index.max() - pd.DateOffset(years=chart_years)
    c_df = full_df[full_df.index >= chart_cutoff].copy()

    num_rows = 3 if show_crsi_subplot else 2
    row_heights = [0.58, 0.24, 0.18] if show_crsi_subplot else [0.68, 0.32]
    subplot_titles = [f"{active_ticker} — Daily Candlesticks & SMA", f"Streak Oscillator"]
    if show_crsi_subplot:
        subplot_titles.append("2-Day Average Connors RSI")

    fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=row_heights, subplot_titles=subplot_titles)

    fig.add_trace(go.Candlestick(x=c_df.index, open=c_df["Open"], high=c_df["High"], low=c_df["Low"], close=c_df["Close"], name="OHLC", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
    fig.add_trace(go.Scatter(x=c_df.index, y=c_df["SMA"], mode="lines", name=f"{sma_period}-SMA", line=dict(color="#ff9800", width=1.6)), row=1, col=1)

    if show_trade_markers and not trades_df.empty:
        t_sub = trades_df.copy()
        t_sub["Entry_DT"] = pd.to_datetime(t_sub["Entry Date (Open)"])
        t_sub["Exit_DT"] = pd.to_datetime(t_sub["Exit Date (Open)"])
        t_sub = t_sub[t_sub["Entry_DT"] >= chart_cutoff]
        if not t_sub.empty:
            fig.add_trace(go.Scatter(x=t_sub["Entry_DT"], y=t_sub["Entry Open ($)"], mode="markers", name="Entry Fill", marker=dict(symbol="triangle-up", size=10, color="#00e676")), row=1, col=1)
            fig.add_trace(go.Scatter(x=t_sub["Exit_DT"], y=t_sub["Exit Open ($)"], mode="markers", name="Exit Fill", marker=dict(symbol="triangle-down", size=10, color="#ffeb3b")), row=1, col=1)

    streak_colors = np.where(c_df["SMA_Streak"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=c_df.index, y=c_df["SMA_Streak"], name="Streak Histogram", marker_color=streak_colors, opacity=0.65), row=2, col=1)
    fig.add_trace(go.Scatter(x=c_df.index, y=c_df["SMA_Streak"], mode="lines", name="Streak Line", line=dict(color="#90caf9", width=1.2)), row=2, col=1)
    fig.add_hline(y=streak_x, line_dash="dash", line_color="#26a69a", row=2, col=1)
    fig.add_hline(y=-streak_x, line_dash="dash", line_color="#ef5350", row=2, col=1)
    fig.add_hline(y=0, line_color="#78909c", line_width=1, row=2, col=1)

    if show_crsi_subplot:
        fig.add_trace(go.Scatter(x=c_df.index, y=c_df["CRSI_2D_Avg"], mode="lines", name="2D CRSI", line=dict(color="#ce93d8", width=1.5)), row=3, col=1)

    fig.update_layout(height=720, template="plotly_dark", xaxis_rangeslider_visible=False, hovermode="x unified", margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TAB 2: PORTFOLIO BACKTEST & MONTHLY/ANNUAL RETURNS TABLE
# =============================================================================
with tab_bt:
    st.subheader(f"🧪 {backtest_years}-Year Backtest & Performance Matrix")
    st.markdown(
        f"Initial Capital: **${initial_capital:,.0f}** | Max Slots: **{max_portfolio_slots}** | Commission: **${commission_per_trade}/trade**"
    )

    bt_mode = st.radio("Backtest Mode", options=["Single Symbol Details", "Full Universe Multi-Stock Portfolio Backtest"], horizontal=True)

    if bt_mode.startswith("Single Symbol"):
        st.markdown(f"**Active Symbol:** `{active_ticker}` | **Exit Rule:** `{selected_exit}`")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Strategy CAGR", f"{metrics.get('CAGR (%)', 0):.2f}%", f"B&H: {metrics.get('Buy & Hold CAGR (%)', 0):.2f}%")
        k2.metric("Ending Equity", f"${bt_df['Strategy_Equity'].iloc[-1]:,.2f}")
        k3.metric("Win Rate", f"{metrics.get('Win Rate (%)', 0):.1f}%", f"{metrics.get('Total Trades', 0)} Trades")
        k4.metric("Profit Factor", f"{metrics.get('Profit Factor', 0):.2f}")
        k5.metric("Max Drawdown", f"{metrics.get('Max Drawdown (%)', 0):.2f}%")
        k6.metric("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', 0):.2f}")

        eq_fig = go.Figure()
        eq_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["Strategy_Equity"], mode="lines", name="Strategy", line=dict(color="#00e676", width=2)))
        eq_fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df["BuyHold_Equity"], mode="lines", name="Buy & Hold", line=dict(color="#90a4ae", width=1.5, dash="dot")))
        eq_fig.update_layout(height=420, template="plotly_dark", hovermode="x unified", title=f"Full {backtest_years}-Year Equity Growth (${initial_capital:,.0f} Initial Capital)", yaxis_title="Equity ($)")
        st.plotly_chart(eq_fig, use_container_width=True)

        st.subheader("📅 Monthly & Annual Returns Matrix (%)")
        returns_matrix = compute_monthly_annual_returns_table(bt_df["Strategy_Equity"])
        if not returns_matrix.empty:
            st.dataframe(returns_matrix, use_container_width=True)
        else:
            st.info("Insufficient history for monthly returns breakdown.")

        st.subheader("📋 Trade Execution Log")
        if not trades_df.empty:
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades triggered.")

    else:
        uc1, uc2 = st.columns(2)
        with uc1:
            univ_preset = st.selectbox("Select Universe for Portfolio Simulation", options=list(PRESET_UNIVERSES.keys()), index=0)
        with uc2:
            univ_cap = st.slider("Max Equities in Portfolio Simulation (0 = All)", min_value=0, max_value=550, value=30)

        if univ_preset == "S&P 500 (Full Universe)":
            uni_tickers = fetch_sp500_tickers()
        else:
            uni_tickers = PRESET_UNIVERSES[univ_preset]

        uni_tickers = uni_tickers[:univ_cap] if univ_cap > 0 else uni_tickers
        st.write(f"Universe Loaded: **{len(uni_tickers)} symbols** | Portfolio Slots: **{max_portfolio_slots}** | Capital: **${initial_capital:,.0f}**")

        @st.cache_data(ttl=3600, show_spinner=False)
        def run_portfolio_simulation(tickers: tuple, bt_yrs: int, t_side: str, e_style: str, s_mode: str, s_x: int, ex_x: int, crsi_flt: bool, c_l: float, c_s: float, ex_rule: str, max_slots: int, comm: float, init_cap: float) -> tuple:
            end_dt = dt.date.today() + dt.timedelta(days=1)
            start_dt = end_dt - dt.timedelta(days=int(bt_yrs * 365.25) + 180)
            try:
                raw_batch = yf.download(list(tickers), start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), auto_adjust=True, group_by="ticker", progress=False, threads=True)
            except Exception:
                return pd.DataFrame(), pd.DataFrame(), {}

            stock_dfs = {}
            for tk in tickers:
                try:
                    t_df = raw_batch.copy() if len(tickers) == 1 else raw_batch[tk].copy()
                    t_df = t_df.dropna(subset=["Close", "Open"])
                    if len(t_df) >= 225:
                        stock_dfs[tk] = enrich_indicators(t_df, sma_len=sma_period, exit_lookback_x=ex_x, cap_streak=cap_streak_mode)
                except Exception:
                    continue

            if not stock_dfs:
                return pd.DataFrame(), pd.DataFrame(), {}

            # Dynamic timeline: Use union of all timelines so shorter history stocks participate without truncating the 25y horizon
            master_index = None
            for tk, df in stock_dfs.items():
                sub = df[df.index >= (df.index.max() - pd.DateOffset(years=bt_yrs))]
                if master_index is None:
                    master_index = sub.index
                else:
                    master_index = master_index.union(sub.index)
            master_index = master_index.sort_values()

            if len(master_index) < 30:
                return pd.DataFrame(), pd.DataFrame(), {}

            individual_rows = []
            all_trades = []
            for tk, df in stock_dfs.items():
                b_df, t_df, m = run_backtest(
                    df, backtest_years=bt_yrs, trade_side=t_side, entry_style=e_style,
                    streak_match_mode=s_mode, streak_x=s_x, exit_lookback_x=ex_x,
                    use_crsi_filter=crsi_flt, crsi_long_filter=c_l, crsi_short_filter=c_s, exit_rule=ex_rule,
                    commission_per_trade=comm, initial_capital=init_cap
                )
                if m and m.get("Total Trades", 0) > 0:
                    individual_rows.append({
                        "Ticker": tk,
                        "Strategy CAGR (%)": m["CAGR (%)"],
                        "Win Rate (%)": m["Win Rate (%)"],
                        "Profit Factor": m["Profit Factor"],
                        "Total Trades": m["Total Trades"],
                        "Max Drawdown (%)": m["Max Drawdown (%)"],
                        "Sharpe Ratio": m["Sharpe Ratio"],
                    })
                    if not t_df.empty:
                        t_df["Ticker"] = tk
                        all_trades.append(t_df)

            ind_summary_df = pd.DataFrame(individual_rows)

            port_equity = np.full(len(master_index), init_cap, dtype=float)
            cash = init_cap
            active_positions = []

            for i in range(1, len(master_index)):
                dt_curr = master_index[i]
                remaining_positions = []
                for pos in active_positions:
                    tk = pos["Ticker"]
                    df = stock_dfs[tk]
                    if dt_curr not in df.index:
                        remaining_positions.append(pos)
                        continue

                    row = df.loc[dt_curr]
                    c_val = row["Close"]
                    s_val = row["SMA"]
                    h_x = row["Highest_Close_X"]
                    l_x = row["Lowest_Close_X"]

                    is_exit = False
                    if pos["Direction"] == 1:
                        if ex_rule.startswith("1st Close Above SMA") and c_val > s_val: is_exit = True
                        elif ex_rule.startswith("X-Day Highest Close") and c_val >= h_x: is_exit = True
                        elif ex_rule.startswith("1st Close Above SMA OR") and (c_val > s_val or c_val >= h_x): is_exit = True
                        elif ex_rule.startswith("Fixed Holding Period") and (i - pos["entry_idx"]) >= ex_x: is_exit = True
                    else:
                        if ex_rule.startswith("1st Close Above SMA") and c_val < s_val: is_exit = True
                        elif ex_rule.startswith("X-Day Highest Close") and c_val <= l_x: is_exit = True
                        elif ex_rule.startswith("1st Close Above SMA OR") and (c_val < s_val or c_val <= l_x): is_exit = True
                        elif ex_rule.startswith("Fixed Holding Period") and (i - pos["entry_idx"]) >= ex_x: is_exit = True

                    if is_exit and i + 1 < len(master_index):
                        next_dt = master_index[i+1]
                        if next_dt in df.index:
                            exit_open = df.loc[next_dt, "Open"]
                            ret_frac = pos["Direction"] * ((exit_open - pos["entry_price"]) / pos["entry_price"])
                            pnl = pos["allocated_capital"] * ret_frac - comm
                            cash += pos["allocated_capital"] + pnl
                        else:
                            remaining_positions.append(pos)
                    else:
                        remaining_positions.append(pos)

                active_positions = remaining_positions

                available_slots = max_slots - len(active_positions)
                if available_slots > 0 and cash > 5000:
                    candidates = []
                    for tk, df in stock_dfs.items():
                        if any(p["Ticker"] == tk for p in active_positions):
                            continue
                        if dt_curr not in df.index:
                            continue

                        row = df.loc[dt_curr]
                        st_val = row["SMA_Streak"]
                        crsi_val = row["CRSI_2D_Avg"]
                        if np.isnan(st_val):
                            continue

                        is_mean_rev = e_style.startswith("Mean Reversion")
                        if is_mean_rev:
                            long_ok = (t_side in ("Long Only", "Both (Long & Short)")) and (st_val <= -s_x)
                            short_ok = (t_side in ("Short Only", "Both (Long & Short)")) and (st_val >= s_x)
                        else:
                            long_ok = (t_side in ("Long Only", "Both (Long & Short)")) and (st_val >= s_x)
                            short_ok = (t_side in ("Short Only", "Both (Long & Short)")) and (st_val <= -s_x)

                        if crsi_flt:
                            if long_ok: long_ok = (not np.isnan(crsi_val) and crsi_val <= c_l)
                            if short_ok: short_ok = (not np.isnan(crsi_val) and crsi_val >= c_s)

                        if long_ok:
                            candidates.append({"Ticker": tk, "direction": 1, "score": crsi_val})
                        elif short_ok:
                            candidates.append({"Ticker": tk, "direction": -1, "score": crsi_val})

                    candidates = sorted(candidates, key=lambda x: x["score"] if x["direction"] == 1 else -x["score"])
                    to_enter = candidates[:available_slots]

                    slot_capital = cash / max(1, len(active_positions) + len(to_enter))
                    for cand in to_enter:
                        tk = cand["Ticker"]
                        if i + 1 < len(master_index):
                            next_dt = master_index[i+1]
                            df = stock_dfs[tk]
                            if next_dt in df.index:
                                entry_open = df.loc[next_dt, "Open"]
                                alloc = min(slot_capital, cash)
                                if alloc > comm:
                                    cash -= (alloc + comm)
                                    active_positions.append({
                                        "Ticker": tk,
                                        "entry_price": entry_open,
                                        "Direction": cand["direction"],
                                        "allocated_capital": alloc,
                                        "entry_idx": i+1
                                    })

                current_market_value = cash
                for pos in active_positions:
                    tk = pos["Ticker"]
                    df = stock_dfs[tk]
                    if dt_curr in df.index:
                        curr_px = df.loc[dt_curr, "Close"]
                        pos_val = pos["allocated_capital"] * (1.0 + pos["Direction"] * ((curr_px - pos["entry_price"]) / pos["entry_price"]))
                        current_market_value += max(0.0, pos_val)
                    else:
                        current_market_value += pos["allocated_capital"]

                port_equity[i] = current_market_value

            port_df = pd.DataFrame({"Portfolio_Equity": port_equity}, index=master_index)
            bench_ticker = "SPY" if "SPY" in stock_dfs else list(stock_dfs.keys())[0]
            bench_close = stock_dfs[bench_ticker]["Close"].reindex(master_index).ffill().bfill()
            port_df["BuyHold_Equity"] = init_cap * (bench_close / bench_close.iloc[0])

            years_sim = max((master_index[-1] - master_index[0]).days / 365.25, 0.1)
            port_cagr = (((port_df["Portfolio_Equity"].iloc[-1] / init_cap) ** (1.0 / years_sim)) - 1.0) * 100.0
            bh_cagr = (((port_df["BuyHold_Equity"].iloc[-1] / init_cap) ** (1.0 / years_sim)) - 1.0) * 100.0
            max_dd = (((port_df["Portfolio_Equity"] - port_df["Portfolio_Equity"].cummax()) / port_df["Portfolio_Equity"].cummax()) * 100.0).min()

            port_metrics = {
                "Portfolio CAGR (%)": round(port_cagr, 2),
                "Benchmark CAGR (%)": round(bh_cagr, 2),
                "Max Drawdown (%)": round(max_dd, 2),
                "Final Portfolio Value ($)": round(port_df["Portfolio_Equity"].iloc[-1], 2),
            }

            return ind_summary_df, port_df, port_metrics

        if st.button("🚀 Run Portfolio Simulation", type="primary"):
            with st.spinner(f"Simulating multi-stock portfolio across {len(uni_tickers)} symbols (dynamic timelines)..."):
                ind_df, port_df, p_metrics = run_portfolio_simulation(
                    tuple(uni_tickers), backtest_years, trade_side, entry_style, streak_match_mode,
                    streak_x, exit_lookback_x, use_crsi_filter, crsi_long_thresh, crsi_short_thresh, selected_exit,
                    int(max_portfolio_slots), float(commission_per_trade), float(initial_capital)
                )
            if not port_df.empty:
                st.session_state["cached_port_res"] = (ind_df, port_df, p_metrics)
            else:
                st.warning("Portfolio simulation returned no records.")

        if "cached_port_res" in st.session_state:
            ind_df, port_df, p_metrics = st.session_state["cached_port_res"]

            st.markdown("### 🏆 Portfolio-Level Performance Summary")
            pk1, pk2, pk3, pk4 = st.columns(4)
            pk1.metric("Portfolio CAGR", f"{p_metrics.get('Portfolio CAGR (%)', 0):.2f}%", f"Benchmark: {p_metrics.get('Benchmark CAGR (%)', 0):.2f}%")
            pk2.metric("Final Portfolio Value", f"${p_metrics.get('Final Portfolio Value ($)', 0):,.2f}")
            pk3.metric("Max Portfolio Drawdown", f"{p_metrics.get('Max Drawdown (%)', 0):.2f}%")
            pk4.metric("Starting Capital", f"${initial_capital:,.0f}")

            peq_fig = go.Figure()
            peq_fig.add_trace(go.Scatter(x=port_df.index, y=port_df["Portfolio_Equity"], mode="lines", name="Portfolio Strategy", line=dict(color="#00e676", width=2.2)))
            peq_fig.add_trace(go.Scatter(x=port_df.index, y=port_df["BuyHold_Equity"], mode="lines", name="Benchmark (SPY/First)", line=dict(color="#90a4ae", width=1.5, dash="dot")))
            peq_fig.update_layout(height=420, template="plotly_dark", hovermode="x unified", title=f"Portfolio Equity Growth (${initial_capital:,.0f} Initial Capital)", yaxis_title="Equity ($)")
            st.plotly_chart(peq_fig, use_container_width=True)

            st.subheader("📅 Portfolio Monthly & Annual Returns Matrix (%)")
            port_returns_matrix = compute_monthly_annual_returns_table(port_df["Portfolio_Equity"])
            if not port_returns_matrix.empty:
                st.dataframe(port_returns_matrix, use_container_width=True)

            st.markdown("### 📋 Individual Stock CAGRs & Performance Table")
            st.dataframe(ind_df.sort_values(by="Strategy CAGR (%)", ascending=False), use_container_width=True, hide_index=True, height=450)
        else:
            st.info("Click **Run Portfolio Simulation** above to compute portfolio-level CAGR, equity curves, and returns tables.")


# =============================================================================
# TAB 3: UNIVERSE SCANNER (EDGE ANALYSIS & FORWARD RETURNS)
# =============================================================================
with tab_scanner:
    st.subheader("🌐 Full Stock Universe Scanner & Edge Analysis")
    st.caption("Compare the **Average 5D Forward Return following the Current Streak** against the **Baseline Average 5D Forward Return (All Days)**.")

    col_u1, col_u2 = st.columns([2, 2])
    with col_u1:
        selected_universe_preset = st.selectbox("Select Stock Universe", options=list(PRESET_UNIVERSES.keys()), index=2, key="scan_univ")
    with col_u2:
        max_symbols_cap = st.slider("Max Symbols to Batch Scan (0 = All)", min_value=0, max_value=550, value=505, key="scan_cap")

    if selected_universe_preset == "S&P 500 (Full Universe)":
        base_tickers = fetch_sp500_tickers()
    else:
        base_tickers = PRESET_UNIVERSES[selected_universe_preset]

    scan_tickers = base_tickers[:max_symbols_cap] if max_symbols_cap > 0 else base_tickers
    st.write(f"Loaded **{len(scan_tickers)} symbols** in universe `{selected_universe_preset}`.")

    custom_universe_txt = st.text_area("Or Override with Custom Tickers (Comma-Separated)", value="", height=60, key="scan_custom")
    if custom_universe_txt.strip():
        scan_tickers = [t.strip().upper() for t in custom_universe_txt.split(",") if t.strip()]

    @st.cache_data(ttl=3600, show_spinner=False)
    def run_full_universe_scan(tickers: tuple, sma_p: int, st_x: int, ex_x: int, cap_s: bool) -> pd.DataFrame:
        end_dt = dt.date.today() + dt.timedelta(days=1)
        start_dt = end_dt - dt.timedelta(days=365 * 3)
        try:
            raw_batch = yf.download(list(tickers), start=start_dt.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"), auto_adjust=True, group_by="ticker", progress=False, threads=True)
        except Exception:
            return pd.DataFrame()

        rows = []
        for tk in tickers:
            try:
                if len(tickers) == 1:
                    t_df = raw_batch.copy()
                else:
                    if tk not in raw_batch.columns.get_level_values(0):
                        continue
                    t_df = raw_batch[tk].copy()

                t_df = t_df.dropna(subset=["Close", "Open"])
                if len(t_df) < 120:
                    continue

                ind_df = enrich_indicators(t_df, sma_len=sma_p, exit_lookback_x=ex_x, cap_streak=cap_s)
                last = ind_df.iloc[-1]
                prev = ind_df.iloc[-2]
                ret_1d = ((last["Close"] / prev["Close"]) - 1.0) * 100.0
                st_val = int(last["SMA_Streak"]) if not np.isnan(last["SMA_Streak"]) else 0
                crsi_2d_val = float(last["CRSI_2D_Avg"])

                baseline_fwd_ret = float(ind_df["Fwd_5D_Return"].mean()) if "Fwd_5D_Return" in ind_df else np.nan
                streak_history = ind_df[ind_df["SMA_Streak"] == st_val]
                if not streak_history.empty and "Fwd_5D_Return" in streak_history:
                    streak_fwd_ret = float(streak_history["Fwd_5D_Return"].mean())
                    streak_fwd_display = f"{streak_fwd_ret:+.2f}%" if not np.isnan(streak_fwd_ret) else "no history"
                else:
                    streak_fwd_ret = np.nan
                    streak_fwd_display = "no history"

                if st_val <= -st_x:
                    status = f"🟢 BUY (Streak = {st_val})"
                elif st_val >= st_x:
                    status = f"🔴 SHORT (Streak = +{st_val})"
                else:
                    status = "⚪ Neutral"

                rows.append({
                    "Ticker": tk,
                    "Last Close": round(float(last["Close"]), 2),
                    "1D Chg (%)": round(ret_1d, 2),
                    f"{sma_p}-SMA Streak": st_val,
                    "2-Day Avg CRSI": round(crsi_2d_val, 2),
                    "Avg 5D Fwd Ret [Current Streak]": streak_fwd_display,
                    "_raw_streak_fwd": streak_fwd_ret if not np.isnan(streak_fwd_ret) else -999.0,
                    "Baseline Avg 5D Fwd Ret (All Days)": f"{baseline_fwd_ret:+.2f}%" if not np.isnan(baseline_fwd_ret) else "no history",
                    "Signal for Tomorrow's Open": status,
                })
            except Exception:
                continue
        return pd.DataFrame(rows)

    if st.button("🚀 Run Universe Batch Scan Now", type="primary"):
        with st.spinner(f"Batch scanning {len(scan_tickers)} equities across Yahoo Finance..."):
            scan_result_df = run_full_universe_scan(tuple(scan_tickers), sma_period, streak_x, exit_lookback_x, cap_streak_mode)
        if not scan_result_df.empty:
            st.session_state["cached_scan_df"] = scan_result_df
        else:
            st.warning("Batch scan returned no records.")

    if "cached_scan_df" in st.session_state:
        df_display = st.session_state["cached_scan_df"]
        sc1, sc2 = st.columns(2)
        with sc1:
            sort_metric = st.selectbox("Sort Table By", options=["_raw_streak_fwd", f"{sma_period}-SMA Streak", "2-Day Avg CRSI", "1D Chg (%)", "Ticker"], index=0, format_func=lambda x: "Avg 5D Fwd Ret [Current Streak]" if x == "_raw_streak_fwd" else x)
        with sc2:
            sort_dir = st.radio("Order", options=["Ascending", "Descending"], horizontal=True, index=1, key="scan_ord")

        df_display = df_display.sort_values(by=sort_metric, ascending=(sort_dir == "Ascending")).reset_index(drop=True)
        render_df = df_display.drop(columns=["_raw_streak_fwd"])

        st.dataframe(
            render_df, use_container_width=True, hide_index=True, height=620,
            column_config={"2-Day Avg CRSI": st.column_config.ProgressColumn("2-Day Avg CRSI", format="%.1f", min_value=0, max_value=100)}
        )
    else:
        st.info("Click **Run Universe Batch Scan Now** above to generate rankings across the S&P 500 / universe.")
