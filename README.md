# Quant Research Assistant

AI Agent skill for quantitative investment research — factor analysis, backtesting, and portfolio optimization.

## Quick Start

```bash
pip install -r requirements.txt
```

## Commands

### Factor Analysis

```bash
python quant.py factor --universe hs300 --factor momentum_20d --period 2022-01-01:2023-12-31
```

Computes Rank IC (Spearman), ICIR, t-statistics, and performs quintile group backtest.

### Strategy Backtest

```bash
python quant.py backtest --strategy ma_cross --symbol 000300 --start 2020-01-01 --end 2024-12-31
```

Implements MA crossover with full performance metrics: annual return, Sharpe ratio, max drawdown, Calmar ratio, win rate.

### Portfolio Optimization

```bash
python quant.py portfolio --symbols 000300,000905,NH01535 --optimizer risk_parity
python quant.py portfolio --symbols 000300,000905,NH01535 --optimizer mean_variance
```

Risk parity and mean-variance optimization with efficient frontier visualization.

### Report Generation

```bash
python quant.py report --strategy ma_cross --symbol 000300 --output report.html
```

Generates a standalone HTML report with interactive charts.

## Features

- Real market data via akshare
- Spearman rank IC computation
- Quintile group backtest with return curves
- MA crossover strategy engine
- Risk parity and Markowitz optimization
- scipy.optimize.minimize for portfolio construction
- matplotlib-based visualization
- Standalone HTML report output

## Requirements

- Python 3.8+
- numpy, scipy, pandas, matplotlib
- akshare

## License

MIT
