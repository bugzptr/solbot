# At the top of solusdt_bot.py
PAPER_TRADING_STATE_FILE = Path("results/paper_trading_state.json")
PAPER_TRADE_LOG_FILE = Path("results/paper_trading_log.csv")

def load_paper_trading_state() -> Dict:
    if PAPER_TRADING_STATE_FILE.exists():
        try:
            with open(PAPER_TRADING_STATE_FILE, 'r') as f:
                state = json.load(f)
                # Convert relevant timestamps back to datetime if stored as strings
                if state.get("current_position") and state["current_position"].get("entry_time"):
                    state["current_position"]["entry_time"] = pd.to_datetime(state["current_position"]["entry_time"])
                return state
        except Exception as e:
            logger.error(f"Error loading paper trading state: {e}. Starting fresh.")
    return {
        "current_paper_equity": strategy_config_global.get("paper_trading.initial_equity", 10000),
        "current_position": None, # {'symbol': '', 'type': '', 'entry_price': 0, ...}
        "last_processed_candle_ts": None # Store timestamp of last fully processed candle
    }

def save_paper_trading_state(state: Dict):
    try:
        # Convert datetime objects to ISO format string for JSON serialization
        if state.get("current_position") and isinstance(state["current_position"].get("entry_time"), pd.Timestamp):
            state["current_position"]["entry_time"] = state["current_position"]["entry_time"].isoformat()
        if isinstance(state.get("last_processed_candle_ts"), pd.Timestamp):
            state["last_processed_candle_ts"] = state["last_processed_candle_ts"].isoformat()
            
        # Atomic save: write to temp file then rename
        temp_file = PAPER_TRADING_STATE_FILE.with_suffix(".tmp")
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=4)
        os.replace(temp_file, PAPER_TRADING_STATE_FILE)
        logger.debug("Paper trading state saved.")
    except Exception as e:
        logger.error(f"Error saving paper trading state: {e}")

def log_paper_trade(trade_details: Dict):
    # Append to CSV
    log_df = pd.DataFrame([trade_details])
    file_exists = PAPER_TRADE_LOG_FILE.exists()
    log_df.to_csv(PAPER_TRADE_LOG_FILE, mode='a', header=not file_exists, index=False)
    logger.info(f"Paper trade logged: {trade_details.get('type')} {trade_details.get('symbol')} exited. PnL $: {trade_details.get('pnl_dollar',0):.2f}")


def run_paper_trader(api_config_dict: Dict, base_strategy_config: StrategyConfig):
    logger_paper = logging.getLogger("PaperTrader")
    logger_paper.info("--- Starting Paper Trading Mode ---")

    # Use optimized parameters (load from best_params.json or use base_config if WFA used it)
    # For simplicity, assume base_strategy_config already holds the desired params
    # (e.g., after Optuna, user updates solusdt_strategy_base.json with best params)
    current_config = base_strategy_config 

    api = BitgetAPI(**api_config_dict)
    system = DualNNFXSystem(api, current_config) # System uses current_config
    
    state = load_paper_trading_state()
    # Convert loaded string timestamps back to pd.Timestamp if necessary
    if state.get("last_processed_candle_ts"):
        try: state["last_processed_candle_ts"] = pd.to_datetime(state["last_processed_candle_ts"])
        except: state["last_processed_candle_ts"] = None
    if state.get("current_position") and state["current_position"].get("entry_time"):
        try: state["current_position"]["entry_time"] = pd.to_datetime(state["current_position"]["entry_time"])
        except: # Handle case where position is invalid or timestamp conversion fails
             logger.warning("Could not convert entry_time for existing position, resetting position.")
             state["current_position"] = None


    symbol = current_config.get("symbol", "SOLUSDT")
    granularity = current_config.get("granularity", "4H")
    # Convert granularity like "4H" to timedelta
    granularity_td = pd.to_timedelta(granularity.replace('H', 'h').replace('D', 'd').replace('W','w').replace('M','m'))


    check_interval_seconds = current_config.get("paper_trading.check_interval_seconds", 300)
    risk_per_trade = current_config.get("risk_per_trade", 0.015)
    sl_atr_mult = current_config.get("stop_loss_atr_multiplier", 2.0)
    tp_atr_mult = current_config.get("take_profit_atr_multiplier", 3.0)


    while True:
        try:
            logger_paper.info(f"Fetching latest klines for {symbol}...")
            # Fetch enough candles for indicators + a few extra for checks
            # e.g., min_data_after_indicators (100) + chandelier_period (22) + buffer (10) ~ 132
            # Let's use a fixed moderate number like 200-300 for live signal generation
            kline_fetch_limit = current_config.get("backtest_min_data_after_get_klines", 200) # Reuse this config
            
            latest_klines_df = api.get_klines(symbol, granularity, limit=kline_fetch_limit)

            if latest_klines_df.empty or len(latest_klines_df) < current_config.get("backtest_min_data_after_indicators", 50):
                logger_paper.warning(f"Not enough kline data fetched ({len(latest_klines_df)} candles). Retrying after interval.")
                time.sleep(check_interval_seconds)
                continue

            # Identify the latest fully closed candle
            # API usually returns candles up to the current one which might be forming.
            # If the last candle's timestamp + granularity_td > now, it's forming.
            # A simpler way: assume the second to last is always closed if data is fresh.
            if len(latest_klines_df) < 2:
                logger_paper.warning("Less than 2 klines fetched, cannot determine closed candle. Retrying.")
                time.sleep(check_interval_seconds)
                continue
            
            closed_candle = latest_klines_df.iloc[-2] # Second to last candle
            closed_candle_ts = closed_candle.name # This is a pd.Timestamp

            if state.get("last_processed_candle_ts") and closed_candle_ts <= state["last_processed_candle_ts"]:
                logger_paper.debug(f"No new closed candle since {state['last_processed_candle_ts']}. Current closed: {closed_candle_ts}. Sleeping.")
                time.sleep(check_interval_seconds)
                continue
            
            logger_paper.info(f"New closed candle detected: {closed_candle_ts}")

            # Process this new closed candle
            # We need enough history for indicators ending at this closed_candle
            # So, pass latest_klines_df.loc[:closed_candle_ts] (inclusive)
            df_for_indicators = latest_klines_df.loc[:closed_candle_ts].copy()
            if len(df_for_indicators) < current_config.get("backtest_min_data_after_indicators", 50):
                 logger_paper.warning(f"Not enough historical data ({len(df_for_indicators)}) leading up to {closed_candle_ts} for indicators. Sleeping.")
                 state["last_processed_candle_ts"] = closed_candle_ts # Mark as processed to avoid re-evaluating this candle
                 save_paper_trading_state(state)
                 time.sleep(check_interval_seconds)
                 continue


            df_indicators = system.calculate_indicators(df_for_indicators, symbol)
            # DropNa for only the last row needed for signal generation
            # No, generate_signals needs history. The dropna in backtest_pair is more aggressive.
            # For live, we only care about the signal on the *latest closed candle*.
            # Indicators might have NaNs at the start, but should be valid for the latest rows.
            
            # Get signals for the latest closed candle
            df_signals = system.generate_signals(df_indicators)
            if df_signals.empty or closed_candle_ts not in df_signals.index:
                logger_paper.warning(f"Could not generate signals for candle {closed_candle_ts}. Indicator data might be all NaN up to this point.")
                state["last_processed_candle_ts"] = closed_candle_ts
                save_paper_trading_state(state)
                time.sleep(check_interval_seconds)
                continue

            latest_signals = df_signals.loc[closed_candle_ts] # Signals for the specific closed candle

            current_price_for_action = latest_signals['close'] # Use close of the signal candle
            current_atr = latest_signals.get('atr', np.nan)

            # --- Position Management ---
            if state["current_position"] is None: # If no open position
                if latest_signals.get('long_signal', False) and pd.notna(current_atr) and current_atr > 1e-9:
                    sl = current_price_for_action - (sl_atr_mult * current_atr)
                    tp = current_price_for_action + (tp_atr_mult * current_atr)
                    stop_distance_price = sl_atr_mult * current_atr
                    if stop_distance_price > 1e-9:
                        position_size = (state["current_paper_equity"] * risk_per_trade) / stop_distance_price
                        state["current_position"] = {
                            "symbol": symbol, "type": "long", "entry_price": current_price_for_action,
                            "entry_time": closed_candle_ts, "position_size": position_size,
                            "stop_loss_price": sl, "take_profit_price": tp, "atr_at_entry": current_atr
                        }
                        logger_paper.info(f"PAPER ENTRY (LONG): {symbol} at {current_price_for_action:.4f}, Size: {position_size:.4f}, SL: {sl:.4f}, TP: {tp:.4f}")
                    else: logger_paper.warning(f"[{symbol}] Could not enter long: ATR too small for valid stop loss.")

                elif latest_signals.get('short_signal', False) and pd.notna(current_atr) and current_atr > 1e-9:
                    sl = current_price_for_action + (sl_atr_mult * current_atr)
                    tp = current_price_for_action - (tp_atr_mult * current_atr)
                    stop_distance_price = sl_atr_mult * current_atr
                    if stop_distance_price > 1e-9:
                        position_size = (state["current_paper_equity"] * risk_per_trade) / stop_distance_price
                        state["current_position"] = {
                            "symbol": symbol, "type": "short", "entry_price": current_price_for_action,
                            "entry_time": closed_candle_ts, "position_size": position_size,
                            "stop_loss_price": sl, "take_profit_price": tp, "atr_at_entry": current_atr
                        }
                        logger_paper.info(f"PAPER ENTRY (SHORT): {symbol} at {current_price_for_action:.4f}, Size: {position_size:.4f}, SL: {sl:.4f}, TP: {tp:.4f}")
                    else: logger_paper.warning(f"[{symbol}] Could not enter short: ATR too small for valid stop loss.")

            elif state["current_position"] is not None: # If position exists, check for exit
                pos = state["current_position"]
                exit_paper_trade = False
                exit_reason_paper = ""

                if pos["type"] == "long":
                    if current_price_for_action <= pos["stop_loss_price"]: exit_paper_trade, exit_reason_paper = True, "Stop Loss"
                    elif current_price_for_action >= pos["take_profit_price"]: exit_paper_trade, exit_reason_paper = True, "Take Profit"
                    elif latest_signals.get('long_exit', False): exit_paper_trade, exit_reason_paper = True, "Strategy Exit Signal"
                
                elif pos["type"] == "short":
                    if current_price_for_action >= pos["stop_loss_price"]: exit_paper_trade, exit_reason_paper = True, "Stop Loss"
                    elif current_price_for_action <= pos["take_profit_price"]: exit_paper_trade, exit_reason_paper = True, "Take Profit"
                    elif latest_signals.get('short_exit', False): exit_paper_trade, exit_reason_paper = True, "Strategy Exit Signal"

                if exit_paper_trade:
                    pnl_pips_paper = (current_price_for_action - pos['entry_price']) if pos['type'] == 'long' else (pos['entry_price'] - current_price_for_action)
                    pnl_dollar_paper = pnl_pips_paper * pos['position_size']
                    state["current_paper_equity"] += pnl_dollar_paper
                    
                    trade_log_entry = {
                        "symbol": pos["symbol"], "type": pos["type"], 
                        "entry_time": pos["entry_time"].isoformat(), "exit_time": closed_candle_ts.isoformat(),
                        "entry_price": pos["entry_price"], "exit_price": current_price_for_action,
                        "pnl_dollar": pnl_dollar_paper, "exit_reason": exit_reason_paper,
                        "equity_after_trade": state["current_paper_equity"], "position_size": pos["position_size"]
                    }
                    log_paper_trade(trade_log_entry)
                    state["current_position"] = None
            
            state["last_processed_candle_ts"] = closed_candle_ts
            save_paper_trading_state(state) # Save state after all actions for this candle

        except requests.exceptions.RequestException as e_api:
            logger_paper.error(f"API Request Exception during paper trading loop: {e_api}")
        except Exception as e_loop:
            logger_paper.error(f"Unhandled exception in paper trading loop: {e_loop}", exc_info=True)
        
        logger_paper.debug(f"Loop finished. Current paper equity: {state.get('current_paper_equity', 'N/A'):.2f}. Sleeping for {check_interval_seconds}s.")
        time.sleep(check_interval_seconds)

# Modify __main__ to select action
if __name__ == "__main__":
    # ... (ts_run, h_file_log, logger setup as before) ...
    ts_run = datetime.now().strftime('%Y%m%d_%H%M%S')
    h_file_log = None
    for dir_p_str in ["config","data","logs","results/optuna_studies","results/walk_forward_reports"]: Path(dir_p_str).mkdir(parents=True,exist_ok=True)
    
    try:
        log_f_path = Path("logs")/f"solusdt_bot_run_{ts_run}.log"
        h_file_log=logging.FileHandler(log_f_path); h_file_log.setLevel(logging.INFO)
        h_file_log.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(h_file_log) 
        
        strategy_config_global = StrategyConfig() 
        logger.info(f"SOLUSDT NNFX Bot Run ID: {ts_run}")
        logger.info(f"Using Base Strategy Config: {strategy_config_global.config_path}")

        api_cfg = {}; path_api=Path("config/api_config.json")
        if path_api.exists():
            try: 
                with open(path_api,"r") as f: api_cfg=json.load(f)
                logger.info("API credentials loaded.")
            except Exception as e: logger.error(f"API cfg load err {path_api}: {e}")
        else: 
            logger.warning(f"API config {path_api} not found. Creating dummy. Public API access only.")
            try:
                with open(path_api,"w") as f_dum: json.dump({"api_key":"", "secret_key":"", "passphrase":"", "sandbox":True},f_dum,indent=4)
                logger.info(f"Dummy API cfg created: {path_api}. Please update.")
            except Exception as e_dum: logger.error(f"Dummy API cfg creation err: {e_dum}")

        ACTION = strategy_config_global.get("run_action", "OPTIMIZE") 
        SYMBOL = strategy_config_global.get("symbol", "SOLUSDT")
        N_OPT_TRIALS = strategy_config_global.get("optuna_trials", 30) 
        
        logger.info(f"Script Action: {ACTION}, Symbol: {SYMBOL}, Optuna Trials (if any): {N_OPT_TRIALS}")

        best_params_from_optuna_run = None 

        if ACTION == "PAPER_TRADE": # New action
            run_paper_trader(api_cfg, strategy_config_global) # Pass strategy_config_global (base or user-modded)
        
        elif ACTION in ["OPTIMIZE", "BOTH"]: # Existing Optuna logic
            # ... (Optuna logic from previous full file) ...
            logger.info(f"--- Starting Optuna Parameter Optimization for {SYMBOL} ({N_OPT_TRIALS} trials) ---")
            study_name_opt = f"{SYMBOL.lower()}_opt_{ts_run}"
            storage_opt = f"sqlite:///results/optuna_studies/{study_name_opt}.db"
            study = optuna.create_study(study_name=study_name_opt,storage=storage_opt,load_if_exists=False,direction="maximize")
            obj_func = lambda trial: optuna_objective_solusdt(trial, api_cfg, strategy_config_global) 
            study.optimize(obj_func, n_trials=N_OPT_TRIALS)
            logger.info(f"Optuna study {study_name_opt} complete. Best value: {study.best_value:.4f}")
            logger.info(f"Best params: {study.best_params}")
            best_params_from_optuna_run = study.best_params 
            with open(Path(f"results/optuna_studies/{study_name_opt}_best_params.json"),'w') as fbp: json.dump(best_params_from_optuna_run,fbp,indent=4)
            logger.info(f"Best params saved.")


        if ACTION == "BACKTEST_ONLY": 
            # ... (BACKTEST_ONLY logic from previous full file) ...
            logger.info(f"--- Running Single Backtest for {SYMBOL} with Base Config ---")
            cfg_bt_only = strategy_config_global 
            api_bt_only = BitgetAPI(**api_cfg)
            system_bt_only = DualNNFXSystem(api_bt_only, cfg_bt_only)
            result_bt_only = system_bt_only.backtest_pair(SYMBOL)
            logger.info(f"Backtest Result for {SYMBOL} (Base Config): Score={system_bt_only._calculate_score(result_bt_only):.2f}, Trades={result_bt_only.get('total_trades',0)}, PnL%={result_bt_only.get('total_return_pct',0):.2f}%")
            if 'error' not in result_bt_only or result_bt_only['error'] == 'No trades':
                res_file_bo = Path(f"results/backtest_only_{SYMBOL}_{ts_run}.json")
                try:
                    with open(res_file_bo, 'w') as f_bo:
                        def datetime_converter(o): 
                            if isinstance(o, (datetime, pd.Timestamp)): return o.isoformat()
                            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
                        json.dump(result_bt_only, f_bo, indent=4, default=datetime_converter) 
                    logger.info(f"BACKTEST_ONLY result saved to {res_file_bo}.")
                except Exception as e_json_save:
                    logger.error(f"Error saving BACKTEST_ONLY result to JSON: {e_json_save}")


        if ACTION in ["WALK_FORWARD", "BOTH"]:
            # ... (Walk-Forward logic from previous full file, ensuring params_for_wfa_flat is correctly built from best_params_from_optuna_run or loaded) ...
            params_for_wfa_flat = None 
            if best_params_from_optuna_run:
                logger.info("Using parameters from the current Optuna run for WFA.")
                params_for_wfa_flat = best_params_from_optuna_run 
            else:
                study_dir = Path("results/optuna_studies")
                param_files = sorted(study_dir.glob(f"{SYMBOL.lower()}_opt_*_best_params.json"), key=os.path.getmtime, reverse=True)
                if param_files:
                    logger.info(f"Loading best params for WFA from: {param_files[0]}")
                    with open(param_files[0], 'r') as f_latest_best: params_for_wfa_flat = json.load(f_latest_best) 
                else:
                    logger.warning("No optimized params from current Optuna run or saved files. WFA will use parameters from solusdt_strategy_base.json.")
                    temp_base_cfg_for_wfa = strategy_config_global 
                    params_for_wfa_flat = {} 
                    opt_ranges_from_base = temp_base_cfg_for_wfa.get("optuna_parameter_ranges", {})
                    # Define the map from Optuna/JSON keys to actual config keys
                    parameter_key_map_for_base_to_wfa = { 
                        "tema_length": ("indicators", "tema_period"), "cci_length": ("indicators", "cci_period"),
                        "efi_length": ("indicators", "elder_fi_period"), "kijun_sen_length": ("indicators", "kijun_sen_period"),
                        "williams_r_length": ("indicators", "williams_r_period"), "cmf_length": ("indicators", "cmf_window"),
                        "williams_r_threshold_opt": ("indicators", "williams_r_threshold"),
                        "stop_loss_atr_multiplier_opt": (None, "stop_loss_atr_multiplier"),
                        "take_profit_atr_multiplier_opt": (None, "take_profit_atr_multiplier"),
                        "risk_per_trade_opt": (None, "risk_per_trade")
                    }
                    for opt_key in opt_ranges_from_base.keys():
                        target_path_parts = parameter_key_map_for_base_to_wfa.get(opt_key)
                        if target_path_parts:
                            section, actual_key = target_path_parts
                            if section == "indicators":
                                params_for_wfa_flat[opt_key] = temp_base_cfg_for_wfa.get(f"indicators.{actual_key}")
                            elif section is None:
                                params_for_wfa_flat[opt_key] = temp_base_cfg_for_wfa.get(actual_key)
                        else: 
                             params_for_wfa_flat[opt_key] = temp_base_cfg_for_wfa.get(f"indicators.{opt_key}", temp_base_cfg_for_wfa.get(opt_key))
                    logger.info(f"Constructed WFA params from base config (flat dict): {params_for_wfa_flat}")

            if params_for_wfa_flat: 
                 run_walk_forward_analysis(SYMBOL, params_for_wfa_flat, api_cfg, strategy_config_global)
            else:
                 logger.error("Cannot run WFA: No parameters available (neither from current opt nor loaded).")


    except Exception as e_main: logger.critical("MAIN SCRIPT ERROR:", exc_info=True)
    finally:
        logger.info(f"Run {ts_run} finished.")
        if h_file_log: 
            h_file_log.close() 
            logging.getLogger().removeHandler(h_file_log)