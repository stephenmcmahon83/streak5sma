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
                    # Allow any stock with at least 1 year of history (approx 225 bars)
                    if len(t_df) >= 225:
                        stock_dfs[tk] = enrich_indicators(t_df, sma_len=sma_period, exit_lookback_x=ex_x, cap_streak=cap_streak_mode)
                except Exception:
                    continue

            if not stock_dfs:
                return pd.DataFrame(), pd.DataFrame(), {}

            # Build a unified master trading calendar covering the full requested backtest window from all available data
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

            # Portfolio Simulation over master calendar
            port_equity = np.full(len(master_index), init_cap, dtype=float)
            cash = init_cap
            active_positions = []

            for i in range(1, len(master_index)):
                dt_curr = master_index[i]
                remaining_positions = []
                
                # 1. Manage exits for active positions
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

                # 2. Check new entries for stocks active on this date
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

                # 3. Mark to market portfolio equity
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
            
            # Use SPY or the first valid ticker as benchmark
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
