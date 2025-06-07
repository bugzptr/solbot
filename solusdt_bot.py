import pandas as pd
import numpy as np
import requests
import hmac
import hashlib
import base64
import time
import json
import os
from datetime import datetime, timedelta, timezone
import ta # Make sure 'ta' is installed: pip install ta
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import logging
from pathlib import Path
import sys
import io
import optuna # pip install optuna sqlalchemy

# --- Global Setup ---
warnings.filterwarnings('ignore', category=UserWarning, module='ta')
warnings.filterwarnings('ignore', category=RuntimeWarning) 

# Configure logging
if not logging.getLogger().handlers: 
    logging.basicConfig(
        level=logging.INFO, # INFO for general, DEBUG for dev
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)] 
    )
logger = logging.getLogger("SOLUSDT_NNFX_Bot")

BASE_CONFIG_PATH = Path("config/solusdt_strategy_base.json")
API_CONFIG_PATH = Path("config/api_config.json")

# --- Default Configuration (Fallback) ---
default_config_for_fallback = {
    "major_bases_for_filtering": ["BTC", "ETH", "SOL", "AVAX", "LINK", "ADA", "DOT", "MATIC", "BNB", "XRP", "DOGE", "TRX", "LTC"], # Not used in single-asset bot
    "filter_by_major_bases_for_top_volume": True, # Not used
    "top_n_pairs_by_volume_to_scan": 1, # Not used
    "symbol": "SOLUSDT", # Specific to this bot
    "granularity": "4H",
    "backtest_kline_limit": 1500, 
    "backtest_min_data_after_get_klines": 200,
    "backtest_min_data_after_indicators": 100,
    "backtest_min_trades_for_ranking": 3, # Used by Optuna if checking trades
    "risk_per_trade": 0.015,
    "stop_loss_atr_multiplier": 2.0,
    "take_profit_atr_multiplier": 3.0,
    "indicators": { 
        "tema_period": 21, 
        "cci_period": 14, 
        "elder_fi_period": 13,
        "chandelier_period": 22, "chandelier_multiplier": 3.0,
        "kijun_sen_period": 26, 
        "williams_r_period": 14, "williams_r_threshold": -50,
        "cmf_window": 20, 
        "psar_step": 0.02, "psar_max_step": 0.2,
        "atr_period_risk": 14, "atr_period_chandelier": 22
    },
    "scoring_weights": { # Used by _calculate_score, which Optuna uses
        "win_rate_weight": 0.20, "profit_factor_weight": 0.25, "total_return_pct_weight": 0.20,
        "sharpe_ratio_weight": 0.25, "max_drawdown_pct_penalty_weight": 0.05, 
        "total_trades_factor_weight": 0.05 
    },
    "optuna_parameter_ranges": { # For Optuna suggestions
        "tema_period": [15, 40, 1], "cci_period": [10, 30, 1],
        "kijun_sen_period": [20, 52, 2], "cmf_window": [14, 30, 1],
        "williams_r_threshold": [-70, -30, 5],
        "stop_loss_atr_multiplier": [1.5, 3.5, 0.1],
        "take_profit_atr_multiplier": [1.5, 5.0, 0.1],
        "risk_per_trade": [0.005, 0.025, 0.001]
    },
    "walk_forward": { # Settings for Walk-Forward Analysis
        "enabled": False, "full_data_start_date": "2021-01-01", 
        "num_oos_periods": 4, "oos_period_days": 90, 
        "is_period_days_multiple_of_oos": 3 
    },
    "cleanup": {"cache_days_old": 7, "max_results_to_keep": 20, "cache_klines_freshness_hours": 4}
}

# --- Configuration Class ---
class StrategyConfig: 
    def __init__(self, params_override: Optional[Dict[str, Any]] = None):
        self.params = self._load_base_config()
        if params_override:
            self._deep_update(self.params, params_override)
        # logger.debug("StrategyConfig initialized.") # Keep this debug if helpful

    def _load_base_config(self) -> Dict[str, Any]:
        current_logger_cfg = logging.getLogger("SOLUSDT_NNFX_Bot.Config")
        if not BASE_CONFIG_PATH.exists():
            if current_logger_cfg.hasHandlers() and any(isinstance(h, logging.StreamHandler) for h in current_logger_cfg.handlers if h.stream == sys.stdout):
                 current_logger_cfg.warning(f"Base strategy config file not found: {BASE_CONFIG_PATH}. Using hardcoded fallback defaults.")
            else: 
                print(f"PRINT WARNING (logger not fully set): Base strategy config file not found: {BASE_CONFIG_PATH}. Using fallback defaults.", file=sys.stderr)
            try:
                BASE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(BASE_CONFIG_PATH, "w") as f_cfg: json.dump(default_config_for_fallback, f_cfg, indent=4)
                msg = f"Created dummy strategy config at {BASE_CONFIG_PATH}. Please review and customize."
                if current_logger_cfg.hasHandlers() and any(isinstance(h, logging.StreamHandler) for h in current_logger_cfg.handlers if h.stream == sys.stdout): current_logger_cfg.info(msg)
                else: print(f"PRINT INFO: {msg}", file=sys.stderr)
            except Exception as e_create_cfg: 
                err_msg = f"Could not create dummy strategy config: {e_create_cfg}"
                if current_logger_cfg.hasHandlers() and any(isinstance(h, logging.StreamHandler) for h in current_logger_cfg.handlers if h.stream == sys.stdout): current_logger_cfg.error(err_msg)
                else: print(f"PRINT ERROR: {err_msg}", file=sys.stderr)
            return default_config_for_fallback.copy() 
        try:
            with open(BASE_CONFIG_PATH, "r") as f: return json.load(f)
        except json.JSONDecodeError as e:
            # Log errors using the main bot logger if StrategyConfig is instantiated after logger setup
            logger.error(f"Error decoding JSON from {BASE_CONFIG_PATH}: {e}. Using fallback defaults.")
            return default_config_for_fallback.copy()
        except Exception as e:
            logger.error(f"Error loading base config {BASE_CONFIG_PATH}: {e}. Using fallback defaults.")
            return default_config_for_fallback.copy()

    def _deep_update(self, d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def get(self, key_path: str, default_val_override: Any = None) -> Any:
        keys = key_path.split('.')
        val = self.params
        try:
            for k in keys: val = val[k]
            return val
        except (KeyError, TypeError):
            current_level_fallback = default_config_for_fallback 
            try:
                for key_fb in keys: current_level_fallback = current_level_fallback[key_fb]
                return current_level_fallback
            except (KeyError, TypeError):
                return default_val_override

strategy_config_global = StrategyConfig()

class BitgetAPI: 
    def __init__(self):
        self.api_key = ""
        self.secret_key = ""
        self.passphrase = ""
        self.base_url = "https://api.bitget.com"
        self.session = requests.Session()
        self.rate_limit_delay = 0.25

        if API_CONFIG_PATH.exists():
            try:
                with open(API_CONFIG_PATH, "r") as f: api_creds = json.load(f)
                self.api_key = api_creds.get("api_key", "")
                self.secret_key = api_creds.get("secret_key", "")
                self.passphrase = api_creds.get("passphrase", "")
                # logger.info(f"API credentials loaded from {API_CONFIG_PATH}.") # Logged in main
            except Exception as e: logger.error(f"Error loading API config {API_CONFIG_PATH}: {e}")
        # else: logger.warning(f"API config {API_CONFIG_PATH} not found. Public API access only.") # Logged in main
        # logger.info(f"Bitget API initialized (API Key present: {bool(self.api_key)})") # Logged in main

    def _generate_signature(self, timestamp: str, method: str, request_path: str, query_string: str = "", body_string: str = "") -> str:
        if not self.secret_key: return ""
        message_to_sign = timestamp + method.upper() + request_path
        if body_string: message_to_sign += body_string
        mac = hmac.new(self.secret_key.encode('utf-8'), message_to_sign.encode('utf-8'), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode('utf-8')

    def _get_headers(self, method: str, request_path: str, query_string: str = "", body_string: str = "") -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        sig_body_string = body_string if method.upper() != "GET" else ""
        headers = {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": self._generate_signature(timestamp, method, request_path, query_string="", body_string=sig_body_string),
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json;charset=UTF-8", "locale": "en-US"
        }
        return {k: v for k, v in headers.items() if v} 

    def get_klines(self, symbol: str, granularity: str = "4H", limit: int = 1000, 
                   start_time_ms: Optional[int] = None, end_time_ms: Optional[int] = None) -> pd.DataFrame:
        current_config = strategy_config_global 
        api_symbol_for_request = symbol if symbol.endswith('_SPBL') else symbol + '_SPBL'
        safe_symbol_fname = api_symbol_for_request.replace('/', '_')
        cache_file_suffix = ""
        if start_time_ms and end_time_ms:
            s_date = datetime.fromtimestamp(start_time_ms/1000, tz=timezone.utc).strftime('%Y%m%d')
            e_date = datetime.fromtimestamp(end_time_ms/1000, tz=timezone.utc).strftime('%Y%m%d')
            cache_file_suffix = f"_{s_date}_{e_date}"
        cache_file = Path(f"data/{safe_symbol_fname}_{granularity.lower()}_klines{cache_file_suffix}.csv")
        expected_cols = ['open', 'high', 'low', 'close', 'volume']
        cache_freshness_h = current_config.get("cleanup.cache_klines_freshness_hours", 4)

        if cache_file.exists(): 
            if (time.time() - cache_file.stat().st_mtime) < cache_freshness_h * 3600 and not (start_time_ms or end_time_ms):
                try:
                    df_cache = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    if not df_cache.empty and all(c in df_cache.columns for c in expected_cols) and \
                       not (df_cache[expected_cols].isnull().values.any() or np.isinf(df_cache[expected_cols].values).any()): # Corrected bool check
                        logger.debug(f"[{symbol}] Using valid cached klines.")
                        return df_cache
                except Exception as e_cache: logger.warning(f"[{symbol}] Kline cache read error: {e_cache}. Refetching.")
        
        time.sleep(self.rate_limit_delay)
        request_path = "/api/spot/v1/market/candles"
        params = {"symbol": api_symbol_for_request, "period": granularity.lower(), "limit": str(limit)}
        if start_time_ms: params["after"] = str(start_time_ms) 
        if end_time_ms: params["before"] = str(end_time_ms)   
        
        max_r, df_out = 3, pd.DataFrame()
        for attempt in range(max_r):
            headers = self._get_headers("GET", request_path) 
            try:
                resp = self.session.get(f"{self.base_url}{request_path}", params=params, headers=headers, timeout=20)
                logger.debug(f"[{symbol}] Kline API (att {attempt+1}): {resp.status_code}, Resp: {resp.text[:250]}")
                resp.raise_for_status()
                api_data = resp.json()
                if str(api_data.get('code')) == '00000' and isinstance(api_data.get('data'), list):
                    if not api_data['data']: logger.warning(f"[{symbol}] API success but no candle data for range/limit."); return df_out
                    df_temp = pd.DataFrame(api_data['data'], columns=['ts', 'open', 'high', 'low', 'close', 'baseVol', 'quoteVol'])
                    df_temp = df_temp.rename(columns={'ts': 'timestamp', 'baseVol': 'volume'})
                    df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'].astype(np.int64), unit='ms')
                    df_temp.set_index('timestamp', inplace=True)
                    for col_name in expected_cols:
                        if col_name in df_temp.columns: df_temp[col_name] = pd.to_numeric(df_temp[col_name], errors='coerce')
                        else: logger.error(f"[{symbol}] Critical kline col '{col_name}' missing!"); return df_out
                    df_out = df_temp[expected_cols].copy() 
                    df_out.sort_index(inplace=True)
                    df_out.replace([np.inf, -np.inf], np.nan, inplace=True); df_out.dropna(inplace=True)
                    if df_out.empty: logger.warning(f"[{symbol}] No valid klines after processing."); return df_out
                    try: 
                        if not (start_time_ms or end_time_ms): 
                            df_out.to_csv(cache_file); logger.debug(f"[{symbol}] Fetched & cached klines.")
                    except Exception as e_csv: logger.error(f"[{symbol}] Error caching klines: {e_csv}")
                    return df_out 
                else:
                    err_msg, err_code = api_data.get('msg', 'Unknown API error'), api_data.get('code', 'N/A')
                    logger.warning(f"[{symbol}] API error klines (Code {err_code}): {err_msg}")
                    if str(err_code) == '40309': return df_out 
                    if attempt < max_r - 1: time.sleep(1 + 2**attempt + np.random.rand()) 
            except requests.exceptions.RequestException as e_req:
                logger.warning(f"[{symbol}] Request failed klines (att {attempt+1}): {e_req}")
                if attempt < max_r - 1: time.sleep(1 + 2**attempt + np.random.rand())
            except json.JSONDecodeError as e_json:
                logger.error(f"[{symbol}] JSON decode error klines (att {attempt+1}): {e_json}. Resp: {resp.text[:200] if 'resp' in locals() else 'N/A'}")
                if attempt < max_r - 1: time.sleep(1 + 2**attempt + np.random.rand())
        logger.error(f"[{symbol}] Failed to fetch klines after {max_r} attempts.")
        return df_out

class NNFXIndicators: 
    def __init__(self, config: StrategyConfig): self.config_params = config.get("indicators", {})
    def _get_param(self, key: str, default: Any) -> Any: return self.config_params.get(key, default)
    def tema(self, data: pd.Series) -> pd.Series:
        p=self._get_param("tema_period",21); e1=data.ewm(span=p,adjust=False).mean(); e2=e1.ewm(span=p,adjust=False).mean(); e3=e2.ewm(span=p,adjust=False).mean(); return 3*e1-3*e2+e3
    def kijun_sen(self, high: pd.Series, low: pd.Series) -> pd.Series:
        p=self._get_param("kijun_sen_period",26); mp=max(1,min(p,p//2 if p>1 else 1)); return (high.rolling(p,mp).max()+low.rolling(p,mp).min())/2
    def cci(self, high:pd.Series,low:pd.Series,close:pd.Series)->pd.Series: return ta.trend.CCIIndicator(high,low,close,self._get_param("cci_period",14),fillna=False).cci()
    def williams_r(self,high:pd.Series,low:pd.Series,close:pd.Series)->pd.Series: return ta.momentum.WilliamsRIndicator(high,low,close,self._get_param("williams_r_period",14),fillna=False).williams_r()
    def elder_force_index(self,close:pd.Series,volume:pd.Series)->pd.Series: return ta.volume.ForceIndexIndicator(close,volume,self._get_param("elder_fi_period",13),fillna=False).force_index()
    def klinger_oscillator(self,high:pd.Series,low:pd.Series,close:pd.Series,volume:pd.Series)->Tuple[pd.Series,pd.Series]:
        cmf_w=self._get_param("cmf_window",20); logger.debug(f"Using CMF(w={cmf_w}) for System B Volume")
        try:
            cmf_l=ta.volume.ChaikinMoneyFlowIndicator(high,low,close,volume,cmf_w,fillna=False).chaikin_money_flow()
            return cmf_l, pd.Series(np.nan,index=volume.index,dtype=float)
        except Exception as e: logger.error(f"CMF err: {e}"); nan_s=pd.Series(np.nan,index=volume.index); return nan_s,nan_s
    def chandelier_exit(self,high:pd.Series,low:pd.Series,close:pd.Series)->Tuple[pd.Series,pd.Series]:
        p,m,ap=self._get_param("chandelier_period",22),self._get_param("chandelier_multiplier",3.0),self._get_param("atr_period_chandelier",22)
        atr_s=self.atr(high,low,close,ap);mp=max(1,min(p,p//2 if p>1 else 1));h_h=high.rolling(p,mp).max();l_l=low.rolling(p,mp).min(); return h_h-(m*atr_s),l_l+(m*atr_s)
    def parabolic_sar(self,high:pd.Series,low:pd.Series,close:pd.Series)->pd.Series: return ta.trend.PSARIndicator(high,low,close,self._get_param("psar_step",.02),self._get_param("psar_max_step",.2),fillna=False).psar()
    def atr(self,high:pd.Series,low:pd.Series,close:pd.Series,period:Optional[int]=None)->pd.Series: return ta.volatility.AverageTrueRange(high,low,close,period if period is not None else self._get_param("atr_period_risk",14),fillna=False).average_true_range()

class DualNNFXSystem: 
    def __init__(self, api: BitgetAPI, config: StrategyConfig):
        self.api = api; self.config = config; self.ind = NNFXIndicators(config)
    def _safe_calc(self,idx,name,func,*args): # Simplified for less verbose errors
        try: return func(*args)
        except Exception as e: logger.warning(f"Err in {name}: {str(e)[:100]}"); n_out=2 if "Tuple" in str(func.__annotations__.get('return','')) and name in ["CMF","Chandelier"] else 1; return (pd.Series(np.nan,index=idx),)*n_out if n_out>1 else pd.Series(np.nan,index=idx)
    def calculate_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        d=df.copy();idx=d.index; i=self.ind; s=symbol # Aliases
        d['tema']=self._safe_calc(idx,f"[{s}]TEMA",i.tema,d['close']); d['cci']=self._safe_calc(idx,f"[{s}]CCI",i.cci,d['high'],d['low'],d['close'])
        d['elder_fi']=self._safe_calc(idx,f"[{s}]ElderFI",i.elder_force_index,d['close'],d['volume'])
        cl,cs=self._safe_calc(idx,f"[{s}]Chandelier",i.chandelier_exit,d['high'],d['low'],d['close']); d['chandelier_long'],d['chandelier_short']=cl,cs
        d['kijun_sen']=self._safe_calc(idx,f"[{s}]Kijun",i.kijun_sen,d['high'],d['low'])
        d['williams_r']=self._safe_calc(idx,f"[{s}]W%R",i.williams_r,d['high'],d['low'],d['close'])
        cmf,dum_sig=self._safe_calc(idx,f"[{s}]CMF",i.klinger_oscillator,d['high'],d['low'],d['close'],d['volume']); d['klinger'],d['klinger_signal']=cmf,dum_sig
        d['psar']=self._safe_calc(idx,f"[{s}]PSAR",i.parabolic_sar,d['high'],d['low'],d['close']); d['atr']=self._safe_calc(idx,f"[{s}]ATR",i.atr,d['high'],d['low'],d['close'])
        return d
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame: # Largely same, uses CMF logic
        df=data.copy();ind_cols=['tema','cci','elder_fi','kijun_sen','williams_r','klinger','chandelier_long','chandelier_short','psar'];[df.setdefault(c,np.nan) for c in ind_cols if c not in df]
        df['s_a_base']=np.select([df['close']>df['tema'],df['close']<df['tema']],[1,-1],0); df['s_a_conf']=np.select([df['cci']>0,df['cci']<0],[1,-1],0)
        df['s_a_vol']=np.select([df['elder_fi']>0,df['elder_fi']<0],[1,-1],0); df['s_b_base']=np.select([df['close']>df['kijun_sen'],df['close']<df['kijun_sen']],[1,-1],0)
        wpr_t=self.config.get("indicators.williams_r_threshold",-50); df['s_b_conf']=np.select([df['williams_r']>wpr_t,df['williams_r']<wpr_t],[1,-1],0)
        df['s_b_vol']=np.select([df['klinger']>0,df['klinger']<0],[1,-1],0) # CMF > 0 or < 0
        df['long_signal']=((df.s_a_base==1)&(df.s_a_conf==1)&(df.s_a_vol==1)&(df.s_b_base==1)&(df.s_b_conf==1)&(df.s_b_vol==1)).astype(bool)
        df['short_signal']=((df.s_a_base==-1)&(df.s_a_conf==-1)&(df.s_a_vol==-1)&(df.s_b_base==-1)&(df.s_b_conf==-1)&(df.s_b_vol==-1)).astype(bool)
        df['long_exit']=((df.close<df.chandelier_long)|(df.close<df.psar)).astype(bool); df['short_exit']=((df.close>df.chandelier_short)|(df.close>df.psar)).astype(bool)
        return df.drop(columns=['s_a_base','s_a_conf','s_a_vol','s_b_base','s_b_conf','s_b_vol'], errors='ignore') # Clean up temp signal cols

    def backtest_pair(self, symbol: str, data_df_override: Optional[pd.DataFrame]=None, date_from: Optional[datetime]=None, date_to: Optional[datetime]=None) -> Dict:
        log = logging.getLogger(__name__); log.info(f"[{symbol}] Backtesting... Range: {date_from.date() if date_from else 'Full'} to {date_to.date() if date_to else 'Full'}")
        cfg = self.config
        if data_df_override is not None: df_k = data_df_override
        else:
            start_ms, end_ms = (int(d.timestamp()*1000) if d else None for d in [date_from, date_to])
            df_k = self.api.get_klines(symbol,cfg.get("granularity","4H"),cfg.get("backtest_kline_limit",1000),start_ms,end_ms)
        if df_k.empty or len(df_k)<cfg.get("backtest_min_data_after_get_klines",100): return {"symbol":symbol,"error":f"Kline data insufficient ({len(df_k)})"}
        
        df_i = self.calculate_indicators(df_k,symbol)
        cols_to_check = [c for c in df_i.columns if c!='klinger_signal'] # Exclude dummy signal
        df_i.dropna(subset=cols_to_check,inplace=True)
        if len(df_i)<cfg.get("backtest_min_data_after_indicators",50): return {"symbol":symbol,"error":f"Data insufficient post-indicators/dropna ({len(df_i)})"}
        
        df_s = self.generate_signals(df_i)
        trades,pos,eq_hist,equity = [],None,[],10000.0
        risk,sl_m,tp_m = cfg.get("risk_per_trade",.015),cfg.get("stop_loss_atr_multiplier",2.0),cfg.get("take_profit_atr_multiplier",3.0)

        for _,r in df_s.iterrows(): # r for row
            if pos is None:
                atr_v = r.get('atr',np.nan)
                if pd.notna(atr_v) and atr_v > 1e-9 : # Valid ATR
                    price = r['close']
                    pos_size_calc = (equity*risk_val)/(sl_m*atr_v) if (sl_m*atr_v)>1e-9 else 0
                    if r.get('long_signal',False) and pos_size_calc > 0:
                        pos={'type':'long','entry_price':price,'entry_time':r.name,'sl':price-sl_m*atr_v,'tp':price+tp_m*atr_v,'atr0':atr_v,'size':pos_size_calc}
                    elif r.get('short_signal',False) and pos_size_calc > 0:
                        pos={'type':'short','entry_price':price,'entry_time':r.name,'sl':price+sl_m*atr_v,'tp':price-tp_m*atr_v,'atr0':atr_v,'size':pos_size_calc}
            elif pos:
                exit_now,exit_reason=False,""
                price=r['close']; lex,sex=r.get('long_exit',False),r.get('short_exit',False)
                if pos['type']=='long':
                    if price<=pos['sl']:exit_now,exit_reason=True,"SL"
                    elif price>=pos['tp']:exit_now,exit_reason=True,"TP"
                    elif lex:exit_now,exit_reason=True,"Signal"
                elif pos['type']=='short':
                    if price>=pos['sl']:exit_now,exit_reason=True,"SL"
                    elif price<=pos['tp']:exit_now,exit_reason=True,"TP"
                    elif sex:exit_now,exit_reason=True,"Signal"
                if exit_now:
                    pnl_p=(price-pos['entry_price']) if pos['type']=='long' else (pos['entry_price']-price)
                    pnl_r=pnl_p/(sl_m*pos['atr0']) if pos['atr0']>1e-9 else 0.0
                    pnl_usd=pnl_p*pos['size']; equity=max(0,equity+pnl_usd)
                    trades.append({'symbol':symbol,'type':pos['type'],'entry_time':pos['entry_time'],'exit_time':r.name,
                                   'entry_price':pos['entry_price'],'exit_price':price,'pnl_pips':pnl_p,'pnl_r':pnl_r,
                                   'pnl_dollar':pnl_usd,'exit_reason':exit_reason,'atr_at_entry':pos['atr0'],'equity_after_trade':equity})
                    pos=None
            eq_hist.append({'timestamp':r.name,'equity':equity,'in_position':pos is not None})
        
        if not trades: return {'symbol':symbol,'total_trades':0,'final_equity':equity,'equity_curve':eq_hist,'trades':[],'error':'No trades'}
        df_t=pd.DataFrame(trades)
        if not df_t.empty: df_t['entry_time']=pd.to_datetime(df_t['entry_time']).dt.tz_localize(None); df_t['exit_time']=pd.to_datetime(df_t['exit_time']).dt.tz_localize(None)
        if not eq_hist: eq_hist.append({'timestamp':df_s.index[0] if not df_s.empty else pd.Timestamp.now(tz=None).normalize(),'equity':10000.0,'in_position':False})
        
        n_tr=len(df_t); wins=df_t[df_t.pnl_r>0]; loss=df_t[df_t.pnl_r<0]; wr=len(wins)/n_tr if n_tr>0 else 0.0
        aw,al=(d.pnl_r.mean() if not d.empty else 0.0 for d in [wins,loss])
        sum_pr,sum_lr_abs = wins.pnl_r.sum(), abs(loss.pnl_r.sum())
        pf = sum_pr/sum_lr_abs if sum_lr_abs >1e-9 else (np.inf if sum_pr>1e-9 else 0.0)
        log.info(f"[{symbol}] Backtest done. Trades: {n_tr}, Final Eq: {equity:.2f}, PF: {pf:.2f}")
        return {'symbol':symbol,'total_trades':n_tr,'win_rate':wr,'avg_win_r':aw,'avg_loss_r':al,'profit_factor':pf,
                'total_return_r':df_t.pnl_r.sum(),'total_return_pct':(equity/10000.0-1.0)*100.0,
                'max_consecutive_losses':self._calc_mcl(df_t),'max_drawdown_pct':self._calc_mdd(eq_hist),
                'sharpe_ratio':self._calc_sharpe(df_t,eq_hist),'sortino_ratio':self._calc_sortino(df_t,eq_hist),
                'var_95_r':np.percentile(df_t.pnl_r,5) if n_tr>0 else 0.0, 'max_loss_r':df_t.pnl_r.min() if n_tr>0 else 0.0,
                'final_equity':equity,'trades':df_t.to_dict('records'),'equity_curve':eq_hist}

    def _calc_mcl(self,df_t): c,mc=0,0; for r in df_t['pnl_r']: c=c+1 if r<0 else 0; mc=max(mc,c); return mc
    def _calc_mdd(self,eq_h):
        if not eq_h: return 0.0; eq_s=pd.Series([p['equity'] for p in eq_h]); pk=eq_s.expanding(1).max().replace(0,np.nan)
        dd_min=((eq_s-pk)/pk).min(); return abs(dd_min*100.0) if pd.notna(dd_min) else 0.0
    def _calc_ann_factor(self,df_t,eq_h):
        if df_t.empty or len(df_t)<2 or not eq_h or len(eq_h)<2: return 1.0
        st,et=(pd.to_datetime(d[0]['timestamp' if i==0 else 'timestamp']).tz_localize(None) for i,d in enumerate([eq_h,eq_h[-1:]]))
        dur_d=max(1.0,(et-st).total_seconds()/(24*3600.0)); tpy=(len(df_t)/dur_d)*252.0; return np.sqrt(tpy) if tpy>0 else 1.0
    def _calc_sharpe(self,df_t,eq_h):
        if df_t.empty or df_t['pnl_r'].isnull().all() or len(df_t['pnl_r'].dropna())<2: return 0.0
        ret=df_t.pnl_r.dropna();mr,sr=ret.mean(),ret.std();if sr==0 or pd.isna(sr):return np.inf if mr>0 else(0.0 if mr==0 else -np.inf)
        return (mr/sr)*self._calc_ann_factor(df_t,eq_h)
    def _calc_sortino(self,df_t,eq_h,target_r=0.0):
        if df_t.empty or df_t['pnl_r'].isnull().all() or len(df_t['pnl_r'].dropna())<2: return 0.0
        ret=df_t.pnl_r.dropna();mr=ret.mean();down_dev_sq=(target_r-ret[ret<target_r])**2
        if down_dev_sq.empty:return np.inf if mr>target_r else 0.0
        exp_ddr=np.sqrt(down_dev_sq.mean());if exp_ddr==0 or pd.isna(exp_ddr):return np.inf if mr>target_r else 0.0
        return ((mr-target_r)/exp_ddr)*self._calc_ann_factor(df_t,eq_h)

    def _calculate_score(self, result_dict: Dict) -> float: 
        cfg_s = self.config.get("scoring_weights",{}); cfg_p = self.config.get("scoring",{})
        min_tr = self.config.get("backtest_min_trades_for_ranking",3)
        if result_dict.get('total_trades',0) < min_tr: return -999.0
        
        wr,pf,ret,nt,sr,dd = (result_dict.get(k,d) for k,d in [
            ('win_rate',0.0),('profit_factor',0.0),('total_return_pct',0.0),('total_trades',0),
            ('sharpe_ratio',0.0),('max_drawdown_pct',100.0)])

        score = ( (wr*100 * cfg_s.get("win_rate_weight",0.2)) +
                  (min(pf*20,100) if np.isfinite(pf) else (cfg_p.get("profit_factor_inf_score",50.0) if pf>0 else 0)) * cfg_s.get("profit_factor_weight",0.25) +
                  (min(max(ret,-100),100) * cfg_s.get("total_return_pct_weight",0.20)) +
                  (min(max(sr*25,-100),100) if np.isfinite(sr) else (cfg_p.get("sharpe_ratio_inf_score",50.0) if sr>0 else -50)) * cfg_s.get("sharpe_ratio_weight",0.25) -
                  (dd * cfg_s.get("max_drawdown_pct_penalty_weight",0.05)) + # Penalty, so subtract if weight positive
                  (min(nt*0.1,10) * cfg_s.get("total_trades_factor_weight",0.05)) )
        return round(max(score,-1000.0),2)

# --- Optuna Objective & Walk-Forward (Conceptual Stubs) ---
def optuna_objective_solusdt(trial: optuna.trial.Trial, api_config: Dict, base_config: StrategyConfig) -> float:
    logger_opt = logging.getLogger("OptunaObjective"); logger_opt.info(f"Trial {trial.number} starting...")
    opt_ranges = base_config.get("optuna_parameter_ranges", {})
    
    trial_params = {"indicators": {}, "stop_loss_atr_multiplier": 0.0, "take_profit_atr_multiplier": 0.0, "risk_per_trade": 0.0}
    def _get_r(k,d): r=opt_ranges.get(k); return r if r and len(r)==3 else d
    
    ind_p = trial_params["indicators"]
    ind_p["tema_period"]=trial.suggest_int("tema_period",*_get_r("tema_period",[15,40,1]))
    ind_p["cci_period"]=trial.suggest_int("cci_period",*_get_r("cci_period",[10,30,1]))
    ind_p["kijun_sen_period"]=trial.suggest_int("kijun_sen_period",*_get_r("kijun_sen_period",[20,52,2]))
    ind_p["cmf_window"]=trial.suggest_int("cmf_window",*_get_r("cmf_window",[14,30,1]))
    ind_p["williams_r_threshold"]=trial.suggest_int("williams_r_threshold",*_get_r("williams_r_threshold",[-70,-30,5]))
    
    trial_params["stop_loss_atr_multiplier"]=trial.suggest_float("sl_atr_mult",*_get_r("stop_loss_atr_multiplier",[1.5,3.5,0.1]))
    trial_params["take_profit_atr_multiplier"]=trial.suggest_float("tp_atr_mult",*_get_r("take_profit_atr_multiplier",[1.5,5.0,0.1]))
    trial_params["risk_per_trade"]=trial.suggest_float("risk_per_trade",*_get_r("risk_per_trade",[0.005,0.025,0.001]))

    # Create a new config instance for this trial, merging with base
    trial_config_obj = StrategyConfig(params_override=trial_params) # This will use base_config_path from global StrategyConfig

    api_trial = BitgetAPI(**api_config)
    system_trial = DualNNFXSystem(api_trial, trial_config_obj)
    symbol_to_opt = trial_config_obj.get("symbol", "SOLUSDT") # Get symbol from this trial's config
    
    res = system_trial.backtest_pair(symbol=symbol_to_opt)
    if 'error' in res and res['error']!='No trades': logger_opt.warning(f"Trial {trial.number} err: {res['error']}"); return -float('inf')
    
    score = system_trial._calculate_score(res)
    logger_opt.info(f"Trial {trial.number}: Score={score:.2f}, Trd={res.get('total_trades',0)}, PnL%={res.get('total_return_pct',0):.2f}, Params={trial.params}")
    return float(score)

def run_walk_forward_analysis(symbol: str, optimized_params_dict: Dict, api_config: Dict, base_config_for_wfa: StrategyConfig):
    logger_wfa = logging.getLogger("WalkForward")
    wfa_cfg = base_config_for_wfa.get("walk_forward", {})
    if not wfa_cfg.get("enabled", False): logger_wfa.info(f"WFA for {symbol} disabled. Skipping."); return
    logger_wfa.info(f"--- Starting WFA for {symbol} ---")

    api = BitgetAPI(**api_config)
    full_hist_df = api.get_klines(symbol, base_config_for_wfa.get("granularity","4H"), limit=2000) # Fetch large history
    if full_hist_df.empty or len(full_hist_df) < base_config_for_wfa.get("backtest_min_data_after_indicators",50)*wfa_cfg.get("num_oos_periods",4):
        logger_wfa.error(f"Not enough history for {symbol} for WFA. Fetched {len(full_hist_df)}."); return
    full_hist_df.sort_index(inplace=True)
    logger_wfa.info(f"WFA: {len(full_hist_df)} candles for {symbol} ({full_hist_df.index.min()} to {full_hist_df.index.max()})")

    num_oos,oos_days,is_mult = wfa_cfg.get("num_oos_periods",4),wfa_cfg.get("oos_period_days",90),wfa_cfg.get("is_period_days_multiple_of_oos",3)
    is_days = oos_days*is_mult; all_oos_res=[]
    oos_end_dt = full_hist_df.index[-1]

    for i in range(num_oos):
        oos_e, oos_s = oos_end_dt, oos_end_dt - pd.Timedelta(days=oos_days)
        is_e, is_s = oos_s - pd.Timedelta(days=1), oos_s - pd.Timedelta(days=1+is_days)
        if is_s < full_hist_df.index[0] or oos_s < full_hist_df.index[0]: logger_wfa.warning(f"WFA period {i+1} extends beyond data. Stopping."); break
        logger_wfa.info(f"WFA {num_oos-i}: IS=[{is_s.date()}-{is_e.date()}], OOS=[{oos_s.date()}-{oos_e.date()}]")
        
        # Construct params for WFA run (using Optuna's best or base)
        wfa_params_override = {"indicators":{}} # Start fresh for clarity
        opt_ranges = base_config_for_wfa.get("optuna_parameter_ranges",{})
        for k_opt,v_opt_range in opt_ranges.items(): # Iterate over keys defined in optuna_parameter_ranges
            if k_opt in optimized_params_dict: # If this param was optimized by Optuna
                if k_opt in ["sl_atr_mult", "tp_atr_mult", "risk_per_trade"]: # Top level params
                    wfa_params_override[k_opt.replace('_atr_mult','_atr_multiplier')] = optimized_params_dict[k_opt]
                else: # Assumed to be an indicator param
                    wfa_params_override["indicators"][k_opt] = optimized_params_dict[k_opt]
        
        wfa_run_cfg = StrategyConfig(params_override=wfa_params_override) # Uses base and overrides
        wfa_system = DualNNFXSystem(api, wfa_run_cfg)
        oos_data = full_hist_df[oos_s:oos_e].copy()

        if oos_data.empty or len(oos_data) < base_config_for_wfa.get("backtest_min_data_after_indicators",50):
            logger_wfa.warning(f"WFA OOS {num_oos-i}: Not enough data ({len(oos_data)}). Skipping."); oos_end_dt = is_s - pd.Timedelta(days=1); continue
        
        oos_res = wfa_system.backtest_pair(symbol,data_df_override=oos_data,date_from=oos_s,date_to=oos_e)
        if 'error' in oos_res and oos_res['error']!='No trades': logger_wfa.warning(f"WFA OOS {num_oos-i} FAILED: {oos_res['error']}")
        else: logger_wfa.info(f"WFA OOS {num_oos-i}: Trades={oos_res.get('total_trades',0)}, PnL%={oos_res.get('total_return_pct',0):.2f}%")
        all_oos_res.append({"wfa_period":num_oos-i, "is_start":is_s, "is_end":is_e, "oos_start":oos_s, "oos_end":oos_e, **oos_res})
        oos_end_dt = is_s - pd.Timedelta(days=1)

    if all_oos_res:
        df_oos = pd.DataFrame(all_oos_res).sort_values("wfa_period"); res_dir=Path("results/walk_forward_reports"); res_dir.mkdir(exist_ok=True)
        fn_wfa=res_dir/f"wfa_report_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"; df_oos.to_csv(fn_wfa,index=False)
        logger_wfa.info(f"WFA report for {symbol}: {fn_wfa}")
        logger_wfa.info(f"Agg OOS for {symbol}: Avg PnL%={df_oos['total_return_pct'].mean():.2f}%, Avg Trades={df_oos['total_trades'].mean():.1f}")

# --- Main Execution ---
if __name__ == "__main__":
    if sys.platform.startswith('win'): pass 

    ts_run = datetime.now().strftime('%Y%m%d_%H%M%S')
    h_file_log = None
    for dir_p_str in ["config","data","logs","results/optuna_studies","results/walk_forward_reports"]: Path(dir_p_str).mkdir(parents=True,exist_ok=True)
    
    try:
        log_f_path = Path("logs")/f"solusdt_bot_run_{ts_run}.log"
        h_file_log=logging.FileHandler(log_f_path); h_file_log.setLevel(logging.INFO)
        h_file_log.setFormatter(logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s'))
        logging.getLogger().addHandler(h_file_log) # Add to root to get logs from StrategyConfig init
        
        # strategy_config_global is already initialized at module level
        logger.info(f"SOLUSDT NNFX Bot Run ID: {ts_run}")
        logger.info(f"Using Base Strategy Config: {strategy_config_global.config_path}")

        api_cfg = {}; path_api=Path("config/api_config.json")
        if path_api.exists():
            try: 
                with open(path_api,"r") as f: api_cfg=json.load(f)
                logger.info("API credentials loaded.")
            except Exception as e: logger.error(f"API cfg load err {path_api}: {e}")
        else: # Create dummy API config if not found
            logger.warning(f"API config {path_api} not found. Creating dummy. Public API access only.")
            try:
                with open(path_api,"w") as f_dum: json.dump({"api_key":"", "secret_key":"", "passphrase":"", "sandbox":True},f_dum,indent=4)
                logger.info(f"Dummy API cfg created: {path_api}. Please update.")
            except Exception as e_dum: logger.error(f"Dummy API cfg creation err: {e_dum}")

        ACTION = strategy_config_global.get("run_action", "OPTIMIZE") # "OPTIMIZE", "WALK_FORWARD", "BOTH", "BACKTEST_ONLY"
        SYMBOL = strategy_config_global.get("symbol", "SOLUSDT")
        N_OPT_TRIALS = strategy_config_global.get("optuna_trials", 30) # Get from config or use a default
        
        best_params = None
        if ACTION in ["OPTIMIZE", "BOTH"]:
            logger.info(f"--- Starting Optuna Optimization for {SYMBOL} ({N_OPT_TRIALS} trials) ---")
            study_name_opt = f"{SYMBOL.lower()}_opt_{ts_run}"
            storage_opt = f"sqlite:///results/optuna_studies/{study_name_opt}.db"
            study = optuna.create_study(study_name=study_name_opt,storage=storage_opt,load_if_exists=False,direction="maximize")
            obj_func = lambda trial: optuna_objective_solusdt(trial, api_cfg, strategy_config_global) # Pass base config for ranges
            study.optimize(obj_func, n_trials=N_OPT_TRIALS)
            logger.info(f"Optuna study {study_name_opt} complete. Best value: {study.best_value:.4f}")
            logger.info(f"Best params: {study.best_params}")
            best_params = study.best_params
            with open(Path(f"results/optuna_studies/{study_name_opt}_best_params.json"),'w') as fbp: json.dump(best_params,fbp,indent=4)
            logger.info(f"Best params saved.")

        if ACTION == "BACKTEST_ONLY": # Simple backtest with base config
            logger.info(f"--- Running Single Backtest for {SYMBOL} with Base Config ---")
            cfg_backtest_only = StrategyConfig() # Uses solusdt_strategy_base.json
            api_bt_only = BitgetAPI(**api_cfg)
            system_bt_only = DualNNFXSystem(api_bt_only, cfg_backtest_only)
            result_bt_only = system_bt_only.backtest_pair(SYMBOL)
            logger.info(f"Backtest Result for {SYMBOL}: Score={system_bt_only._calculate_score(result_bt_only):.2f}, Trades={result_bt_only.get('total_trades',0)}, PnL%={result_bt_only.get('total_return_pct',0):.2f}%")
            # Could save this result to a file too

        if ACTION in ["WALK_FORWARD", "BOTH"]:
            params_for_wfa_run = {}
            if best_params: params_for_wfa_run = best_params
            else: # Try to load latest best params if not optimized now
                study_dir = Path("results/optuna_studies")
                param_files = sorted(study_dir.glob("*_best_params.json"), key=os.path.getmtime, reverse=True)
                if param_files:
                    logger.info(f"Loading best params for WFA from: {param_files[0]}")
                    with open(param_files[0], 'r') as f_latest_best: params_for_wfa_run = json.load(f_latest_best)
                else:
                    logger.warning("No optimized params from current run or saved files. WFA will use base config.")
                    # Construct a flat dict from base config for WFA if optimized_params_dict is expected flat
                    # This part needs careful alignment with how run_walk_forward_analysis expects params
                    temp_base_cfg = StrategyConfig() # Loads from file
                    params_for_wfa_run = { # Reconstruct a flat-like dict
                        "tema_period": temp_base_cfg.get("indicators.tema_period"),
                        "cci_period": temp_base_cfg.get("indicators.cci_period"),
                        # ... add all other optimizable params from optuna_parameter_ranges keys ...
                         "sl_atr_mult": temp_base_cfg.get("stop_loss_atr_multiplier"),
                         "tp_atr_mult": temp_base_cfg.get("take_profit_atr_multiplier"),
                         "risk_per_trade": temp_base_cfg.get("risk_per_trade")
                    }


            if params_for_wfa_run: # Ensure we have some params
                 run_walk_forward_analysis(SYMBOL, params_for_wfa_run, api_cfg, strategy_config_global)
            else:
                 logger.error("Cannot run WFA: No parameters available (neither from current opt nor loaded).")

    except Exception as e_main: logger.critical("MAIN SCRIPT ERROR:", exc_info=True)
    finally:
        logger.info(f"Run {ts_run} finished.")
        if h_file_log: logging.getLogger().removeHandler(h_file_log); h_file_log.close()