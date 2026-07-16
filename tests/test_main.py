#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for skill-quant-research."""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant import (
    compute_performance_metrics,
    ma_crossover_strategy,
    _negative_sharpe,
    _portfolio_volatility,
    TRADING_DAYS_PER_YEAR,
    RISK_FREE_RATE,
)


class TestComputePerformanceMetrics:
    """Tests for compute_performance_metrics function."""

    def _make_result_df(self, strategy_returns, benchmark_returns=None):
        """Helper to create a result DataFrame."""
        n = len(strategy_returns)
        df = pd.DataFrame({
            "price": np.cumprod(1 + np.array(strategy_returns)) * 100,
            "ma_short": np.ones(n) * 100,
            "ma_long": np.ones(n) * 100,
            "signal": np.ones(n),
            "position": np.ones(n),
            "return": strategy_returns,
            "strategy_return": strategy_returns,
            "benchmark_return": benchmark_returns if benchmark_returns is not None else strategy_returns,
        })
        return df

    def test_positive_returns(self):
        """Test performance metrics with positive returns."""
        np.random.seed(42)
        sr = np.random.normal(0.001, 0.01, 252)
        br = np.random.normal(0.0005, 0.01, 252)
        df = self._make_result_df(sr, br)
        metrics = compute_performance_metrics(df)
        assert "error" not in metrics
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert 0.0 <= metrics["win_rate"] <= 1.0

    def test_max_drawdown_negative(self):
        """Test that max_drawdown is always <= 0."""
        np.random.seed(42)
        sr = np.random.normal(-0.001, 0.02, 252)
        br = np.random.normal(0.0005, 0.01, 252)
        df = self._make_result_df(sr, br)
        metrics = compute_performance_metrics(df)
        assert metrics["max_drawdown"] <= 0.0

    def test_empty_data(self):
        """Test that empty data returns error."""
        df = pd.DataFrame({
            "price": [],
            "strategy_return": [],
            "benchmark_return": [],
            "position": [],
        })
        # Need to ensure it handles empty gracefully
        df["strategy_return"] = pd.Series([], dtype=float)
        df["benchmark_return"] = pd.Series([], dtype=float)
        df["return"] = pd.Series([], dtype=float)
        df["position"] = pd.Series([], dtype=float)
        metrics = compute_performance_metrics(df)
        assert "error" in metrics


class TestMACrossoverStrategy:
    """Tests for ma_crossover_strategy function."""

    def test_basic_ma_cross(self):
        """Test basic MA crossover strategy produces valid signals."""
        np.random.seed(42)
        prices = pd.Series(
            np.cumprod(1 + np.random.normal(0.0005, 0.015, 100)) * 100,
            index=pd.date_range("2020-01-01", periods=100),
        )
        result = ma_crossover_strategy(prices, ma_short=5, ma_long=20)
        assert result is not None
        assert "signal" in result.columns
        assert "position" in result.columns
        assert "strategy_return" in result.columns
        assert len(result) > 0

    def test_signal_values(self):
        """Test that signals are only -1 or 1."""
        np.random.seed(42)
        prices = pd.Series(
            np.cumprod(1 + np.random.normal(0.001, 0.01, 200)) * 100,
            index=pd.date_range("2020-01-01", periods=200),
        )
        result = ma_crossover_strategy(prices, ma_short=5, ma_long=20)
        unique_signals = set(result["signal"].unique())
        assert unique_signals.issubset({-1, 1})

    def test_uptrend_bullish(self):
        """Test that in strong uptrend, the strategy becomes bullish."""
        prices = pd.Series(
            np.linspace(100, 500, 100),
            index=pd.date_range("2020-01-01", periods=100),
        )
        result = ma_crossover_strategy(prices, ma_short=5, ma_long=20)
        # In a linear uptrend, most positions should be long (1)
        assert result["position"].iloc[-1] == 1


class TestNegativeSharpe:
    """Tests for _negative_sharpe function."""

    def test_negative_sharpe_positive(self):
        """Test negative sharpe computation."""
        returns = np.random.normal(0.001, 0.01, (252, 3))
        returns_df = pd.DataFrame(returns)
        mean_ret = returns_df.mean() * TRADING_DAYS_PER_YEAR
        cov_matrix = returns_df.cov() * TRADING_DAYS_PER_YEAR
        weights = np.array([0.4, 0.3, 0.3])
        ns = _negative_sharpe(weights, mean_ret.values, cov_matrix.values)
        assert isinstance(ns, (float, np.floating))
        # Negative sharpe means the actual sharpe is -ns
        # We can't assert much about the sign, because it depends on random data

    def test_negative_sharpe_equal_weights(self):
        """Test negative sharpe with equal weights."""
        returns = np.random.normal(0.001, 0.01, (252, 3))
        returns_df = pd.DataFrame(returns)
        mean_ret = returns_df.mean() * TRADING_DAYS_PER_YEAR
        cov_matrix = returns_df.cov() * TRADING_DAYS_PER_YEAR
        weights = np.ones(3) / 3
        ns = _negative_sharpe(weights, mean_ret.values, cov_matrix.values)
        assert isinstance(ns, (float, np.floating))
        assert not np.isnan(ns)

    def test_portfolio_volatility_nan(self):
        """Test _portfolio_volatility with NaN in covariance matrix."""
        cov = np.array([[1.0, np.nan], [np.nan, 1.0]])
        weights = np.array([0.5, 0.5])
        vol = _portfolio_volatility(weights, cov)
        assert not np.isnan(vol)
        assert vol >= 0
