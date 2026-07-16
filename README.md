# Quant Research Assistant

[![CI](https://github.com/wzx11223344/skill-quant-research/actions/workflows/ci.yml/badge.svg)](https://github.com/wzx11223344/skill-quant-research/actions/workflows/ci.yml)

AI Agent skill for quantitative investment research — factor analysis, backtesting, and portfolio optimization.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/wzx11223344/skill-quant-research.git
cd skill-quant-research

# Install dependencies
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

## Testing

Run unit tests:

```bash
pip install pytest
pytest tests/ -v
```

Run linting:

```bash
pip install flake8
flake8 quant.py tests/ --max-line-length=120
```

## CI/CD

This project uses GitHub Actions for continuous integration, running on every push and PR to the main branch:

- **Python 3.10 / 3.11 / 3.12** matrix testing
- **pytest** unit test suite
- **flake8** code linting (max-line-length=120, permissive config)

## Project Structure

```
skill-quant-research/
├── .github/workflows/ci.yml  # CI/CD configuration
├── pyproject.toml             # Project config (pytest + flake8)
├── SKILL.md                   # ClawHub skill definition
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── quant.py                   # CLI entry point
└── tests/
    └── test_main.py           # Unit tests
```

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

- Python 3.10+
- numpy, scipy, pandas, matplotlib
- akshare

## License

MIT
