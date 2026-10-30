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
    """Computes a stable table of monthly and annual returns (%) using 'ME' (Month End)."""
    equity_series = equity_series.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    m_eq = equity_series.resample("ME").last()
    m_ret = m_eq.pct_change().fillna(0.0)
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
                long_ok = long_ok and (not np.isnan(c_val) and c_val <= crsi_long_filter)
                short_ok = short_ok and (not np.isnan(c_val) and c_val >= crsi_short_filter)

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
    total_days = (bt_df.
