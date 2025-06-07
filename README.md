# SOLUSDT NNFX Dual System Trading Bot & Optimizer

This project implements an algorithmic trading bot specifically tailored and optimized for the SOLUSDT trading pair on the Bitget exchange. It utilizes a dual-system confirmation strategy inspired by the No Nonsense Forex (NNFX) methodology, coupled with parameter optimization using Optuna and a framework for walk-forward analysis to assess strategy robustness.

## 🎯 Strategy Overview

The core strategy relies on the consensus of two independent systems before generating a trading signal, aiming to improve signal quality and reduce false positives:

*   **System A (Momentum-Based):**
    *   **Baseline:** Triple Exponential Moving Average (TEMA)
    *   **Confirmation:** Commodity Channel Index (CCI)
    *   **Volume:** Elder's Force Index
*   **System B (Trend-Following):**
    *   **Baseline:** Kijun-Sen (from Ichimoku Cloud)
    *   **Confirmation:** Williams %R
    *   **Volume:** Chaikin Money Flow (CMF) *(Note: This is used as a robust alternative to Klinger/PVO, with CMF > 0 for bullish volume and < 0 for bearish)*

Trade exits are managed by a combination of Chandelier Exit and Parabolic SAR.

## ✨ Key Features

*   **SOLUSDT Focused:** All analysis, optimization, and testing are streamlined for the SOLUSDT pair.
*   **Dual System Confirmation:** Enhances signal reliability by requiring agreement from both System A and B.
*   **Parameter Optimization with Optuna:** Systematically finds optimal strategy parameters for SOLUSDT using historical data to maximize a defined performance metric (e.g., custom score, Sharpe Ratio).
*   **Walk-Forward Analysis Framework:** Allows for testing optimized parameters on out-of-sample data segments to assess robustness and mitigate overfitting.
*   **Comprehensive Backtesting Engine:**
    *   Detailed simulation of trades based on historical data.
    *   Calculates key performance metrics: Win Rate, Profit Factor, Total Return (R-multiple & %), Max Drawdown, Sharpe Ratio, Sortino Ratio, Max Consecutive Losses.
*   **Configurable Strategy:**
    *   All strategy parameters, optimization ranges, and walk-forward settings are managed via `config/solusdt_strategy_base.json`.
    *   API keys are stored separately in `config/api_config.json`.
*   **Bitget API Integration:** Fetches k-line data for backtesting and analysis.
*   **Data Caching:** Reduces API calls for k-line data.
*   **Reporting:** Generates CSV and text summaries for backtests, Optuna study results (best parameters), and walk-forward analysis reports.

## 🛠️ Project Structure
## 🚀 Getting Started

### Prerequisites

*   Python 3.9+
*   Pip (Python package installer)
*   Virtual Environment (recommended, e.g., `venv`)

### Installation

1.  **Clone the repository (or set up your project directory):**
    ```bash
    # git clone https://github.com/bugzptr/solbot/
    # cd solusdt_nnfx_optimizer
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```

3.  **Install required Python packages:**
    ```bash
    pip install pandas numpy requests ta optuna sqlalchemy
    ```
    *(Ensure `ta` library is version 0.9.0 or higher for full indicator compatibility, though the current script uses CMF which is widely available).*

### Configuration

1.  **API Configuration (`config/api_config.json`):**
    *   If this file doesn't exist, the script will attempt to create a dummy `api_config.json`.
    *   Populate it with your Bitget API key, secret key, and passphrase if you need authenticated access (not strictly required for public market data fetching).
    ```json
    {
        "api_key": "YOUR_BITGET_API_KEY",
        "secret_key": "YOUR_BITGET_SECRET_KEY",
        "passphrase": "YOUR_BITGET_PASSPHRASE" 
    }
    ```

2.  **Strategy Base Configuration (`config/solusdt_strategy_base.json`):**
    *   This file controls all aspects of the strategy, optimization, and walk-forward analysis.
    *   If it doesn't exist, a dummy version with default values will be created by the script.
    *   **Review and customize this file carefully**, especially:
        *   `symbol` (should be "SOLUSDT").
        *   Indicator parameters under the `"indicators"` section.
        *   `"optuna_parameter_ranges"`: Define the min, max, and step for each parameter you want Optuna to tune.
        *   `"walk_forward"`: Configure settings if you plan to run walk-forward analysis.
        *   `"scoring_weights"`: Adjust how different metrics contribute to the overall score used in optimization.

### Running the Bot

The main script `solusdt_bot.py` can perform different actions based on the `SCRIPT_ACTION` variable set within its `if __name__ == "__main__":` block.

1.  **Parameter Optimization:**
    *   Set `SCRIPT_ACTION = "OPTIMIZE"` in `solusdt_bot.py`.
    *   Configure `OPTIMIZATION_TRIALS` (e.g., `50`, `100`, or more).
    *   Run: `python solusdt_bot.py`
    *   Results (SQLite DB, best parameters JSON) will be saved in `results/optuna_studies/`.

2.  **Walk-Forward Analysis:**
    *   Set `SCRIPT_ACTION = "WALK_FORWARD"` in `solusdt_bot.py`.
    *   Ensure the `walk_forward` section in `config/solusdt_strategy_base.json` is configured.
    *   This mode typically uses parameters found from a prior optimization run (either passed programmatically or by updating `solusdt_strategy_base.json` with the best Optuna params). The current script will use the base config parameters for WFA if optimization wasn't run in the same session or if best params aren't explicitly loaded.
    *   Run: `python solusdt_bot.py`
    *   WFA reports (CSV) will be saved in `results/walk_forward_reports/`.

3.  **Run Both Optimization and then Walk-Forward:**
    *   Set `SCRIPT_ACTION = "BOTH"` in `solusdt_bot.py`.
    *   The script will first run Optuna optimization and then use the best parameters found from that run for the walk-forward analysis.

## 📄 Output

*   **Logs:** Detailed execution logs are saved in the `logs/` directory.
*   **Optuna Studies:** Optimization trials and results are stored in SQLite databases in `results/optuna_studies/`. A JSON file with the best parameters for each study is also saved here.
*   **Walk-Forward Reports:** CSV files detailing the performance in each out-of-sample period are saved in `results/walk_forward_reports/`.
*   **General Analysis Files:** (If full scan/reporting features from the previous multi-asset bot were re-enabled) CSVs and Excel reports summarizing backtests might be found in `results/`.

## 💡 Future Enhancements / To-Do

*   [ ] More sophisticated data fetching for Walk-Forward Analysis to ensure sufficient historical data across all IS/OOS periods.
*   [ ] Option to load specific Optuna best parameters for Walk-Forward Analysis if not running optimization in the same session.
*   [ ] Advanced Optuna features (e.g., pruners, multivariate TPE samplers).
*   [ ] Detailed plotting and visualization of optimization studies and walk-forward results.
*   [ ] Integration of live trading capabilities (order execution, position management).
*   [ ] More granular error handling and reporting for individual backtest failures within Optuna trials.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss and is not suitable for every investor. The authors and contributors are not responsible for any financial losses incurred using this software. **Use at your own risk.** Always backtest thoroughly and understand the risks before deploying any trading bot with real capital. This is not financial advice.
