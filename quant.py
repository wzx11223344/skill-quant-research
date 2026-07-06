#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quant Research Assistant - 量化投研助手
========================================
因子分析 | 策略回测 | 组合优化 | 报告生成

用法:
    python quant.py factor --universe hs300 --factor momentum_20d --period 2022-01-01:2023-12-31
    python quant.py backtest --strategy ma_cross --symbol 000300 --start 2020-01-01
    python quant.py portfolio --symbols 000300,000905,NH01535 --optimizer risk_parity
    python quant.py report --strategy ma_cross --symbol 000300 --output report.html
"""

import argparse
import warnings
import os
import sys
import textwrap
import base64
from io import BytesIO
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局配置
# ============================================================

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'DejaVu Sans', 'Arial'],
    'axes.unicode_minus': False,
    'figure.dpi': 120,
    'savefig.dpi': 120,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.03

# ============================================================
# 1. 数据获取层
# ============================================================

def _safe_ak(symbol, fn, *args, **kwargs):
    """安全调用akshare函数，失败时返回None"""
    try:
        import akshare as ak
        return getattr(ak, fn)(*args, **kwargs)
    except Exception as e:
        print(f"[WARN] akshare.{fn}({symbol}) 获取失败: {e}")
        return None


def fetch_index_daily(symbol, start_date=None, end_date=None):
    """
    获取指数日线数据。
    symbol: 如 '000300' (沪深300), '000905' (中证500)
    返回 DataFrame，列: date, open, high, low, close, volume
    """
    df = _safe_ak(symbol, 'stock_zh_index_daily', symbol=f"sh{symbol}")
    if df is None:
        df = _safe_ak(symbol, 'stock_zh_index_daily', symbol=f"sz{symbol}")
    if df is None or df.empty:
        return None

    df = df.rename(columns={
        'date': 'date', 'open': 'open', 'high': 'high',
        'low': 'low', 'close': 'close', 'volume': 'volume'
    })
    if 'date' not in df.columns:
        return None

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['return'] = df['close'].pct_change()

    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    return df


def fetch_stock_universe(universe='hs300', top_n=30):
    """
    获取股票池成分股代码列表。
    universe: 'hs300' | 'zz500'
    返回 list of str (如 ['000001', '000002', ...])
    """
    try:
        import akshare as ak
    except ImportError:
        print("[ERROR] 请先安装 akshare: pip install akshare")
        sys.exit(1)

    index_code = '000300' if universe.lower() in ('hs300', '000300') else '000905'
    df = _safe_ak(universe, 'index_stock_cons_weight_csindex', symbol=index_code)
    if df is None or df.empty:
        print(f"[WARN] 无法获取 {universe} 成分股，使用预设股票池")
        # fallback: 常见沪深300成分股
        fallback = ['000001', '000002', '000858', '600519', '601318',
                    '600036', '000333', '600276', '601166', '600900',
                    '000651', '002415', '600030', '601398', '000568',
                    '600887', '002714', '601888', '600585', '000725',
                    '002304', '600809', '601899', '600031', '000063',
                    '002142', '601688', '600048', '601225', '000776']
        return fallback[:top_n]

    col_candidates = ['con_code', 'stock_code', 'code', '成分券代码', 'constituent_code']
    code_col = None
    for c in col_candidates:
        if c in df.columns:
            code_col = c
            break

    if code_col is None:
        print(f"[WARN] 成分股数据列名不匹配，可用列: {list(df.columns)}")
        return None

    codes = df[code_col].astype(str).str.replace('.SH', '').str.replace('.SZ', '').str.strip()
    codes = codes[codes.str.match(r'^\d{6}$')]
    return codes.head(top_n).tolist()


def fetch_stock_daily(symbol, start_date=None, end_date=None):
    """
    获取个股日线数据。
    返回 DataFrame，列: date, close, return
    """
    s = str(symbol).zfill(6)
    df = _safe_ak(s, 'stock_zh_a_hist', symbol=s, period='daily',
                  start_date=start_date.replace('-', '') if start_date else '20200101',
                  end_date=end_date.replace('-', '') if end_date else '20991231',
                  adjust='qfq')
    if df is None or df.empty:
        return None

    # 列名映射
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if '日期' in c or c == 'date':
            col_map[c] = 'date'
        elif '收盘' in c or c == 'close':
            col_map[c] = 'close'
    df = df.rename(columns=col_map)

    if 'date' not in df.columns or 'close' not in df.columns:
        return None

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['return'] = df['close'].pct_change()
    df = df.dropna(subset=['return'])

    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    return df


def fetch_fund_daily(symbol, start_date=None, end_date=None):
    """
    获取基金/ETF净值数据。
    返回 DataFrame，列: date, close, return
    """
    import akshare as ak
    df = _safe_ak(symbol, 'fund_open_fund_info_em', symbol=symbol, indicator='单位净值走势')
    if df is None or df.empty:
        return None

    # 列: 净值日期, 单位净值
    df = df.rename(columns={df.columns[0]: 'date', df.columns[1]: 'close'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['return'] = df['close'].pct_change()
    df = df.dropna(subset=['return'])

    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    return df


def fetch_returns_matrix(symbols, start_date='2020-01-01', end_date=None):
    """
    获取多个标的的收益率矩阵。
    symbols: list of str (指数代码或基金代码，如 ['000300', '000905', 'NH01535'])
    返回 (returns_df, price_df): 收益率矩阵和价格矩阵，行=日期，列=标的
    """
    return_dfs = {}
    price_dfs = {}

    for sym in symbols:
        # 先尝试指数
        df = fetch_index_daily(sym, start_date, end_date)
        if df is None:
            df = fetch_fund_daily(sym, start_date, end_date)
        if df is None:
            print(f"[WARN] 跳过 {sym}: 数据获取失败")
            continue
        return_dfs[sym] = df.set_index('date')['return']
        price_dfs[sym] = df.set_index('date')['close']

    if not return_dfs:
        raise ValueError("未能获取任何标的的数据")

    ret_matrix = pd.DataFrame(return_dfs).dropna()
    price_matrix = pd.DataFrame(price_dfs).dropna()
    return ret_matrix, price_matrix

# ============================================================
# 2. 因子分析
# ============================================================

def compute_momentum_factor(prices_df, window=20):
    """
    计算动量因子: 过去 window 日收益率。
    prices_df: index=date, columns=stock symbols
    """
    return prices_df.pct_change(window).shift(1)


def compute_volume_factor(data_dict, window=20):
    """计算成交量因子: 过去 window 日均成交量变化率（简化版）"""
    factors = {}
    for sym, df in data_dict.items():
        if df is None or 'volume' not in df.columns:
            continue
        df = df.copy()
        df['vol_ma'] = df['volume'].rolling(window).mean()
        df['vol_factor'] = df['volume'] / df['vol_ma'] - 1
        df = df.dropna(subset=['vol_factor'])
        factors[sym] = df.set_index('date')['vol_factor']
    if factors:
        return pd.DataFrame(factors)
    return None


def factor_ic_analysis(factor_df, forward_ret_df):
    """
    Rank IC (Spearman) 分析。
    返回 dict: ic_series, ic_mean, ic_std, icir, t_stat, ic_pos_ratio
    """
    common_dates = factor_df.index.intersection(forward_ret_df.index)
    common_stocks = factor_df.columns.intersection(forward_ret_df.columns)
    factor_df = factor_df.loc[common_dates, common_stocks]
    forward_ret_df = forward_ret_df.loc[common_dates, common_stocks]

    ic_series = []
    for date in factor_df.index:
        f_vals = factor_df.loc[date].dropna()
        r_vals = forward_ret_df.loc[date].dropna()
        common = f_vals.index.intersection(r_vals.index)
        if len(common) < 10:
            continue
        ic, _ = sp_stats.spearmanr(f_vals[common], r_vals[common])
        if not np.isnan(ic):
            ic_series.append(ic)

    if not ic_series:
        return None

    ic_arr = np.array(ic_series)
    ic_mean = np.mean(ic_arr)
    ic_std = np.std(ic_arr, ddof=1) or 1e-10
    icir = ic_mean / ic_std
    t_stat = ic_mean / ic_std * np.sqrt(len(ic_arr))
    ic_pos_ratio = np.sum(ic_arr > 0) / len(ic_arr)

    result = {
        'ic_series': pd.Series(ic_arr, name='RankIC'),
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        't_stat': t_stat,
        'ic_pos_ratio': ic_pos_ratio,
        'n_periods': len(ic_arr),
    }
    return result


def factor_group_backtest(factor_df, forward_ret_df, n_groups=5, plot_path=None):
    """
    分层回测：按因子值分为 n_groups 组，计算各组累计收益。
    返回 group_cumret DataFrame 和统计结果。
    """
    combined = []
    for date in factor_df.index:
        f_row = factor_df.loc[date].dropna()
        r_row = forward_ret_df.loc[date].dropna()
        common = f_row.index.intersection(r_row.index)
        if len(common) < n_groups * 3:
            continue
        group_labels = pd.qcut(f_row[common].rank(pct=True), n_groups, labels=range(1, n_groups+1))
        for stock in common:
            combined.append({
                'date': date,
                'stock': stock,
                'group': group_labels[stock],
                'return': r_row[stock]
            })
    if not combined:
        return None, None

    df_comb = pd.DataFrame(combined)
    group_ret = df_comb.groupby(['date', 'group'])['return'].mean().unstack()
    group_cumret = (1 + group_ret).cumprod()

    stats_list = []
    for g in range(1, n_groups+1):
        if g not in group_ret.columns:
            continue
        r = group_ret[g].dropna()
        ann_ret = r.mean() * TRADING_DAYS_PER_YEAR
        ann_vol = r.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe = (ann_ret - RISK_FREE_RATE) / (ann_vol if ann_vol > 0 else 1e-10)
        stats_list.append({
            '组别': f'第{g}组',
            '年化收益率': f'{ann_ret:.2%}',
            '年化波动率': f'{ann_vol:.2%}',
            '夏普比率': f'{sharpe:.2f}',
            '累计收益率': f'{group_cumret[g].iloc[-1]-1:.2%}',
        })
    stats_df = pd.DataFrame(stats_list)

    if n_groups in group_cumret.columns and 1 in group_cumret.columns:
        stats_list.append({
            '组别': '多空(L-S)',
            '年化收益率': '-',
            '年化波动率': '-',
            '夏普比率': '-',
            '累计收益率': f'{(group_cumret[n_groups].iloc[-1] - group_cumret[1].iloc[-1]):.2%}',
        })

    # 绘图
    if plot_path:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_groups))
        for g in range(1, n_groups+1):
            if g in group_cumret.columns:
                label = f'第{g}组' + (' (多头)' if g == n_groups else ' (空头)' if g == 1 else '')
                ax.plot(group_cumret.index, group_cumret[g].values,
                        color=colors[g-1], linewidth=1.5, label=label)
        ax.set_title('分层回测 - 各组累计收益曲线 (Factor Group Backtest)', fontsize=14, fontweight='bold')
        ax.set_xlabel('日期')
        ax.set_ylabel('累计净值')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    return group_cumret, stats_df


def run_factor_analysis(universe='hs300', factor_name='momentum_20d',
                        start_date='2022-01-01', end_date='2023-12-31',
                        output_dir='.'):
    """执行完整的因子分析流程"""
    print(f"\n{'='*60}")
    print(f"  因子分析: universe={universe}, factor={factor_name}")
    print(f"{'='*60}")

    # 1. 获取股票池
    print("\n[1/5] 获取股票池...")
    codes = fetch_stock_universe(universe, top_n=50)
    if not codes:
        print("[ERROR] 无法获取股票池")
        return None
    print(f"  获取到 {len(codes)} 只成分股")

    # 2. 获取个股数据
    print(f"\n[2/5] 获取个股日线数据 (前30只)...")
    stocks_to_fetch = codes[:30]
    data_dict = {}
    for i, code in enumerate(stocks_to_fetch):
        sys.stdout.write(f"\r  {i+1}/{len(stocks_to_fetch)}: {code}")
        sys.stdout.flush()
        df = fetch_stock_daily(code, start_date, end_date)
        if df is not None and len(df) > 60:
            df['symbol'] = code
            data_dict[code] = df
    print()

    if len(data_dict) < 10:
        print(f"[ERROR] 有效个股数据不足 ({len(data_dict)} 只)，无法进行因子分析")
        return None
    print(f"  有效个股: {len(data_dict)} 只")

    # 3. 计算因子
    print(f"\n[3/5] 计算因子: {factor_name}...")
    prices_dict = {}
    for sym, df in data_dict.items():
        prices_dict[sym] = df.set_index('date')['close'].rename(sym)
    prices_df = pd.DataFrame(prices_dict).sort_index()

    if factor_name.startswith('momentum'):
        try:
            window = int(factor_name.split('_')[1].replace('d', ''))
        except (IndexError, ValueError):
            window = 20
        factor_df = compute_momentum_factor(prices_df, window)
    elif factor_name.startswith('volume'):
        factor_df = compute_volume_factor(data_dict)
    else:
        print(f"[ERROR] 未知因子: {factor_name}")
        return None

    if factor_df is None or factor_df.empty:
        print("[ERROR] 因子计算失败")
        return None

    forward_ret_df = prices_df.pct_change().shift(-1)  # T+1 forward return
    print(f"  因子覆盖 {len(factor_df.columns)} 只股票, {len(factor_df)} 个交易日")

    # 4. IC 分析
    print(f"\n[4/5] Rank IC (Spearman) 分析...")
    ic_result = factor_ic_analysis(factor_df, forward_ret_df)
    if ic_result is None:
        print("[ERROR] IC分析失败")
        return None

    print(f"\n  {'─'*50}")
    print(f"  Rank IC 均值:       {ic_result['ic_mean']:+.4f}")
    print(f"  Rank IC 标准差:     {ic_result['ic_std']:.4f}")
    print(f"  ICIR (IC/IC_std):   {ic_result['icir']:.4f}")
    print(f"  t-统计量:           {ic_result['t_stat']:.4f}")
    print(f"  IC>0 比例:          {ic_result['ic_pos_ratio']:.2%}")
    print(f"  有效期数:           {ic_result['n_periods']}")
    print(f"  {'─'*50}")

    # 5. 分层回测
    print(f"\n[5/5] 5分组回测...")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'factor_group_backtest.png')
    group_cumret, stats_df = factor_group_backtest(factor_df, forward_ret_df,
                                                     n_groups=5, plot_path=plot_path)
    if stats_df is not None:
        print(f"\n各组绩效统计:")
        print(stats_df.to_string(index=False))

    # IC 时序图
    ic_plot_path = os.path.join(output_dir, 'factor_ic_series.png')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(ic_result['ic_series'])), ic_result['ic_series'].values,
           color=['#e74c3c' if v < 0 else '#2ecc71' for v in ic_result['ic_series'].values],
           alpha=0.7, width=0.8)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=ic_result['ic_mean'], color='#3498db', linestyle='--',
               linewidth=1.5, label=f"IC均值={ic_result['ic_mean']:.4f}")
    ax.set_title('Rank IC 序列 (Spearman)', fontsize=14, fontweight='bold')
    ax.set_xlabel('期数')
    ax.set_ylabel('Rank IC')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ic_plot_path)
    plt.close(fig)

    print(f"\n图表已保存: {plot_path}, {ic_plot_path}")
    return {'ic_result': ic_result, 'group_stats': stats_df, 'group_cumret': group_cumret}

# ============================================================
# 3. 策略回测
# ============================================================

def ma_crossover_strategy(prices, ma_short=5, ma_long=20):
    """
    均线交叉策略。
    prices: pd.Series，index=date
    返回 DataFrame: date, price, ma_short, ma_long, signal, position, return, strategy_return
    """
    df = pd.DataFrame({'price': prices})
    df['ma_short'] = df['price'].rolling(ma_short).mean()
    df['ma_long'] = df['price'].rolling(ma_long).mean()
    df['signal'] = 0
    df.loc[df['ma_short'] > df['ma_long'], 'signal'] = 1
    df.loc[df['ma_short'] <= df['ma_long'], 'signal'] = -1
    df['position'] = df['signal'].shift(1).fillna(0)  # T日信号决定T+1日持仓
    df['return'] = df['price'].pct_change()
    df['strategy_return'] = df['position'] * df['return']
    df['benchmark_return'] = df['return']
    return df.dropna()


def compute_performance_metrics(result_df, rf=RISK_FREE_RATE):
    """
    计算完整绩效指标。
    result_df: 包含 strategy_return, benchmark_return 列
    """
    sr = result_df['strategy_return'].dropna()
    br = result_df['benchmark_return'].dropna()

    if len(sr) == 0:
        return {'error': '无有效收益率数据'}

    # 策略指标
    total_ret = (1 + sr).prod() - 1
    ann_ret = (1 + total_ret) ** (TRADING_DAYS_PER_YEAR / len(sr)) - 1
    ann_vol = sr.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (ann_ret - rf) / (ann_vol if ann_vol > 0 else 1e-10)

    # 最大回撤
    cum_ret = (1 + sr).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min()
    calmar = ann_ret / (abs(max_dd) if abs(max_dd) > 0 else 1e-10)

    # 胜率
    win_rate = (sr > 0).sum() / len(sr)

    # 基准指标
    bench_total = (1 + br).prod() - 1
    bench_ann = (1 + bench_total) ** (TRADING_DAYS_PER_YEAR / len(br)) - 1
    bench_vol = br.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    bench_sharpe = (bench_ann - rf) / (bench_vol if bench_vol > 0 else 1e-10)

    # 超额
    excess = sr - br
    info_ratio = excess.mean() / (excess.std() if excess.std() > 0 else 1e-10) * np.sqrt(TRADING_DAYS_PER_YEAR)

    metrics = {
        'total_return': total_ret,
        'annual_return': ann_ret,
        'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'benchmark_return': bench_ann,
        'benchmark_volatility': bench_vol,
        'benchmark_sharpe': bench_sharpe,
        'excess_return': ann_ret - bench_ann,
        'information_ratio': info_ratio,
        'n_trades': (result_df['position'].diff().abs() > 0).sum(),
    }
    return metrics


def run_backtest(strategy='ma_cross', symbol='000300',
                 start_date='2020-01-01', end_date=None,
                 output_dir='.'):
    """执行策略回测"""
    print(f"\n{'='*60}")
    print(f"  策略回测: strategy={strategy}, symbol={symbol}")
    print(f"{'='*60}")

    print(f"\n[1/3] 获取数据...")
    df = fetch_index_daily(symbol, start_date, end_date)
    if df is None:
        print(f"[ERROR] 无法获取 {symbol} 数据")
        return None
    if df.empty or 'close' not in df.columns:
        print(f"[ERROR] {symbol} 数据为空")
        return None
    print(f"  数据范围: {df['date'].min().date()} ~ {df['date'].max().date()}, 共 {len(df)} 个交易日")

    print(f"\n[2/3] 执行 {strategy} 策略...")
    prices = df.set_index('date')['close']

    if strategy == 'ma_cross':
        result = ma_crossover_strategy(prices, ma_short=5, ma_long=20)
    else:
        print(f"[ERROR] 未知策略: {strategy}")
        return None

    if result is None or result.empty:
        print("[ERROR] 策略回测结果为空")
        return None

    metrics = compute_performance_metrics(result)
    if 'error' in metrics:
        print(f"[ERROR] {metrics['error']}")
        return None

    # 打印指标
    print(f"\n[3/3] 绩效指标:")
    print(f"  {'─'*50}")
    print(f"  累计收益率:     {metrics['total_return']:>10.2%}")
    print(f"  年化收益率:     {metrics['annual_return']:>10.2%}")
    print(f"  年化波动率:     {metrics['annual_volatility']:>10.2%}")
    print(f"  夏普比率:       {metrics['sharpe_ratio']:>10.2f}")
    print(f"  最大回撤:       {metrics['max_drawdown']:>10.2%}")
    print(f"  Calmar比率:     {metrics['calmar_ratio']:>10.2f}")
    print(f"  胜率:           {metrics['win_rate']:>10.2%}")
    print(f"  ──────────────────────────────────────────────")
    print(f"  基准年化收益:   {metrics['benchmark_return']:>10.2%}")
    print(f"  超额收益:       {metrics['excess_return']:>10.2%}")
    print(f"  信息比率:       {metrics['information_ratio']:>10.2f}")
    print(f"  交易次数:       {metrics['n_trades']:>10}")
    print(f"  {'─'*50}")

    # 绘图
    os.makedirs(output_dir, exist_ok=True)

    # 净值曲线
    cum_strat = (1 + result['strategy_return']).cumprod()
    cum_bench = (1 + result['benchmark_return']).cumprod()
    drawdown = (cum_strat - cum_strat.cummax()) / cum_strat.cummax()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]},
                            sharex=True)

    ax1 = axes[0]
    ax1.plot(result.index, cum_strat.values, color='#e74c3c', linewidth=1.5, label='策略净值')
    ax1.plot(result.index, cum_bench.values, color='#3498db', linewidth=1.2,
             linestyle='--', alpha=0.8, label='基准净值')
    # 标记买卖点
    buy_signals = result[result['position'].diff() > 0]
    sell_signals = result[result['position'].diff() < 0]
    ax1.scatter(buy_signals.index, cum_strat.loc[buy_signals.index], marker='^',
                color='green', s=40, alpha=0.7, label='买入信号')
    ax1.scatter(sell_signals.index, cum_strat.loc[sell_signals.index], marker='v',
                color='red', s=40, alpha=0.7, label='卖出信号')

    ax1.set_title(f'MA交叉策略回测 ({symbol}) | 年化={metrics["annual_return"]:.2%} '
                  f'夏普={metrics["sharpe_ratio"]:.2f} 最大回撤={metrics["max_drawdown"]:.2%}',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('累计净值')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(result.index, drawdown.values, 0, color='#e74c3c', alpha=0.3)
    ax2.plot(result.index, drawdown.values, color='#e74c3c', linewidth=0.8)
    ax2.set_ylabel('回撤')
    ax2.set_xlabel('日期')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate()
    fig.tight_layout()

    bt_plot_path = os.path.join(output_dir, 'backtest_result.png')
    fig.savefig(bt_plot_path)
    plt.close(fig)
    print(f"图表已保存: {bt_plot_path}")

    return {'result': result, 'metrics': metrics, 'cum_strat': cum_strat, 'cum_bench': cum_bench}

# ============================================================
# 4. 组合优化
# ============================================================

def _portfolio_volatility(weights, cov_matrix):
    """组合波动率"""
    return np.sqrt(weights.T @ cov_matrix @ weights)


def _negative_sharpe(weights, mean_ret, cov_matrix, rf=RISK_FREE_RATE):
    """负夏普比率（用于最小化）"""
    port_ret = weights @ mean_ret
    port_vol = _portfolio_volatility(weights, cov_matrix)
    return -(port_ret - rf) / (port_vol if port_vol > 0 else 1e-10)


def _risk_parity_objective(weights, cov_matrix):
    """风险平价的优化目标：各资产风险贡献的方差"""
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    marginal_contrib = cov_matrix @ weights
    risk_contrib = weights * marginal_contrib / (port_vol if port_vol > 0 else 1e-10)
    target = 1.0 / len(weights)
    return np.sum((risk_contrib - target) ** 2)


def mean_variance_optimization(returns_df, target_return=None, allow_short=False):
    """
    均值-方差优化。
    返回 (optimal_weights, metrics)
    """
    mean_ret = returns_df.mean() * TRADING_DAYS_PER_YEAR
    cov_matrix = returns_df.cov() * TRADING_DAYS_PER_YEAR
    n = len(mean_ret)

    if target_return is None:
        target_return = mean_ret.median()

    bounds = [(-1, 1) if allow_short else (0, 1) for _ in range(n)]
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: w @ mean_ret - target_return},
    ]

    x0 = np.ones(n) / n
    result = minimize(_portfolio_volatility, x0, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-12})

    if not result.success:
        print(f"[WARN] 均值-方差优化未收敛: {result.message}")

    w_opt = result.x
    w_opt = np.clip(w_opt, 0, None)
    w_opt = w_opt / w_opt.sum()

    port_ret = w_opt @ mean_ret
    port_vol = _portfolio_volatility(w_opt, cov_matrix)
    sharpe = (port_ret - RISK_FREE_RATE) / (port_vol if port_vol > 0 else 1e-10)

    weights = {col: w for col, w in zip(returns_df.columns, w_opt)}
    metrics = {
        'expected_return': port_ret,
        'expected_volatility': port_vol,
        'sharpe_ratio': sharpe,
        'optimizer': 'mean_variance',
        'target_return': target_return,
    }
    return weights, metrics


def risk_parity_optimization(returns_df, allow_short=False):
    """
    风险平价优化。
    """
    cov_matrix = returns_df.cov() * TRADING_DAYS_PER_YEAR
    n = len(returns_df.columns)

    bounds = [(-1, 1) if allow_short else (1e-6, 1) for _ in range(n)]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    x0 = np.ones(n) / n

    result = minimize(_risk_parity_objective, x0, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': 1000, 'ftol': 1e-12})

    if not result.success:
        print(f"[WARN] 风险平价优化未收敛: {result.message}")

    w_opt = result.x
    w_opt = np.clip(w_opt, 0, None)
    w_opt = w_opt / w_opt.sum()

    mean_ret = returns_df.mean() * TRADING_DAYS_PER_YEAR
    port_ret = w_opt @ mean_ret
    port_vol = _portfolio_volatility(w_opt, cov_matrix)
    sharpe = (port_ret - RISK_FREE_RATE) / (port_vol if port_vol > 0 else 1e-10)

    # 各资产风险贡献
    marginal_contrib = cov_matrix @ w_opt
    risk_contrib = w_opt * marginal_contrib / (port_vol if port_vol > 0 else 1e-10)

    weights = {col: w for col, w in zip(returns_df.columns, w_opt)}
    risk_contribs = {col: rc for col, rc in zip(returns_df.columns, risk_contrib)}
    metrics = {
        'expected_return': port_ret,
        'expected_volatility': port_vol,
        'sharpe_ratio': sharpe,
        'optimizer': 'risk_parity',
        'risk_contributions': risk_contribs,
    }
    return weights, metrics


def compute_efficient_frontier(returns_df, n_points=50, allow_short=False):
    """
    计算有效前沿。
    返回 (frontier_returns, frontier_vols, max_sharpe_weights, min_vol_weights)
    """
    mean_ret = returns_df.mean() * TRADING_DAYS_PER_YEAR
    cov_matrix = returns_df.cov() * TRADING_DAYS_PER_YEAR
    n = len(mean_ret)

    # 先求最大夏普组合
    bounds = [(-1, 1) if allow_short else (0, 1) for _ in range(n)]
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    x0 = np.ones(n) / n

    max_sharpe_result = minimize(_negative_sharpe, x0,
                                  args=(mean_ret, cov_matrix, RISK_FREE_RATE),
                                  method='SLSQP', bounds=bounds, constraints=constraints,
                                  options={'maxiter': 1000, 'ftol': 1e-12})
    max_sharpe_w = max_sharpe_result.x
    max_sharpe_w = np.clip(max_sharpe_w, 0, None)
    max_sharpe_w = max_sharpe_w / max_sharpe_w.sum()

    # 最小方差组合
    min_vol_result = minimize(_portfolio_volatility, x0,
                               args=(cov_matrix,),
                               method='SLSQP', bounds=bounds, constraints=constraints,
                               options={'maxiter': 1000, 'ftol': 1e-12})
    min_vol_w = min_vol_result.x
    min_vol_w = np.clip(min_vol_w, 0, None)
    min_vol_w = min_vol_w / min_vol_w.sum()

    min_ret = min_vol_w @ mean_ret
    max_ret = max(mean_ret)
    target_returns = np.linspace(min_ret, max_ret * 0.95, n_points)

    frontier_vols = []
    achieved_returns = []
    for tgt in target_returns:
        constraints_t = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, t=tgt: w @ mean_ret - t},
        ]
        res = minimize(_portfolio_volatility, x0, args=(cov_matrix,),
                       method='SLSQP', bounds=bounds, constraints=constraints_t,
                       options={'maxiter': 1000, 'ftol': 1e-12})
        if res.success:
            frontier_vols.append(_portfolio_volatility(res.x, cov_matrix))
            achieved_returns.append(res.x @ mean_ret)

    return (np.array(achieved_returns), np.array(frontier_vols),
            max_sharpe_w, min_vol_w)


def run_portfolio_optimization(symbols_str, optimizer='risk_parity',
                               start_date='2020-01-01', end_date=None,
                               output_dir='.'):
    """执行组合优化"""
    symbols = [s.strip() for s in symbols_str.split(',')]
    print(f"\n{'='*60}")
    print(f"  组合优化: optimizer={optimizer}, symbols={symbols}")
    print(f"{'='*60}")

    print(f"\n[1/3] 获取数据...")
    ret_matrix, price_matrix = fetch_returns_matrix(symbols, start_date, end_date)
    print(f"  数据范围: {ret_matrix.index[0].date()} ~ {ret_matrix.index[-1].date()}")
    print(f"  有效标的: {list(ret_matrix.columns)}")

    print(f"\n[2/3] 执行 {optimizer} 优化...")
    if optimizer == 'risk_parity':
        weights, metrics = risk_parity_optimization(ret_matrix)
    elif optimizer == 'mean_variance':
        weights, metrics = mean_variance_optimization(ret_matrix)
    else:
        print(f"[ERROR] 未知优化器: {optimizer}")
        return None

    print(f"\n  最优权重:")
    for asset, w in sorted(weights.items(), key=lambda x: -x[1]):
        bar = '█' * int(w * 50)
        print(f"    {asset:12s}: {w:7.2%}  {bar}")

    print(f"\n  组合指标:")
    print(f"    预期年化收益: {metrics['expected_return']:.2%}")
    print(f"    预期年化波动: {metrics['expected_volatility']:.2%}")
    print(f"    夏普比率:     {metrics['sharpe_ratio']:.2f}")

    if optimizer == 'risk_parity' and 'risk_contributions' in metrics:
        print(f"\n  风险贡献:")
        for asset, rc in sorted(metrics['risk_contributions'].items(), key=lambda x: -x[1]):
            print(f"    {asset:12s}: {rc:.2%}")

    # 有效前沿
    print(f"\n[3/3] 计算有效前沿并绘图...")
    os.makedirs(output_dir, exist_ok=True)
    ef_ret, ef_vol, max_sharpe_w, min_vol_w = compute_efficient_frontier(ret_matrix, n_points=50)

    mean_ret = ret_matrix.mean() * TRADING_DAYS_PER_YEAR
    cov_matrix = ret_matrix.cov() * TRADING_DAYS_PER_YEAR

    fig, ax = plt.subplots(figsize=(10, 7))

    # 有效前沿
    ax.plot(ef_vol, ef_ret, color='#3498db', linewidth=2, label='有效前沿 (Efficient Frontier)')

    # 各资产
    for col in ret_matrix.columns:
        asset_vol = np.sqrt(cov_matrix.loc[col, col])
        ax.scatter(asset_vol, mean_ret[col], s=100, alpha=0.7, label=col, zorder=5)
        ax.annotate(col, (asset_vol, mean_ret[col]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)

    # 最优组合
    optimal_vol = metrics['expected_volatility']
    optimal_ret = metrics['expected_return']
    ax.scatter(optimal_vol, optimal_ret, s=200, marker='*', color='#e74c3c',
               zorder=10, label=f'最优组合 ({optimizer})')

    # 最大夏普
    ms_vol = _portfolio_volatility(max_sharpe_w, cov_matrix)
    ms_ret = max_sharpe_w @ mean_ret
    ax.scatter(ms_vol, ms_ret, s=120, marker='D', color='#f39c12',
               zorder=10, label='最大夏普组合')

    # 最小方差
    mv_vol = _portfolio_volatility(min_vol_w, cov_matrix)
    mv_ret = min_vol_w @ mean_ret
    ax.scatter(mv_vol, mv_ret, s=120, marker='s', color='#2ecc71',
               zorder=10, label='最小方差组合')

    ax.set_title(f'有效前沿与最优组合 | {optimizer}', fontsize=14, fontweight='bold')
    ax.set_xlabel('年化波动率')
    ax.set_ylabel('年化收益率')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.tight_layout()
    ef_plot_path = os.path.join(output_dir, 'efficient_frontier.png')
    fig.savefig(ef_plot_path)
    plt.close(fig)

    print(f"图表已保存: {ef_plot_path}")

    return {'weights': weights, 'metrics': metrics,
            'efficient_frontier': (ef_ret, ef_vol)}

# ============================================================
# 5. 报告生成
# ============================================================

def fig_to_base64(fig):
    """将 matplotlib figure 转为 base64 字符串"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_html_report(strategy='ma_cross', symbol='000300',
                         start_date='2020-01-01', end_date=None,
                         output_path='report.html',
                         output_dir='.'):
    """生成包含所有图表的独立 HTML 报告"""
    print(f"\n{'='*60}")
    print(f"  生成报告: strategy={strategy}, symbol={symbol}")
    print(f"{'='*60}")

    # 运行回测
    bt_result = run_backtest(strategy, symbol, start_date, end_date, output_dir)
    if bt_result is None:
        print("[ERROR] 回测失败，无法生成报告")
        return None

    result = bt_result['result']
    metrics = bt_result['metrics']

    # 生成所有图表
    print("\n[生成图表]...")
    charts = {}
    os.makedirs(output_dir, exist_ok=True)

    # 图1: 净值曲线
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    cum_strat = (1 + result['strategy_return']).cumprod()
    cum_bench = (1 + result['benchmark_return']).cumprod()
    ax1.plot(result.index, cum_strat.values, color='#e74c3c', linewidth=2, label='策略净值')
    ax1.plot(result.index, cum_bench.values, color='#3498db', linewidth=1.5,
             linestyle='--', alpha=0.7, label='基准净值')
    ax1.set_title('累计净值曲线', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig1.autofmt_xdate()
    fig1.tight_layout()
    charts['nav'] = fig_to_base64(fig1)
    plt.close(fig1)

    # 图2: 回撤曲线
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    dd = (cum_strat - cum_strat.cummax()) / cum_strat.cummax()
    ax2.fill_between(result.index, dd.values, 0, color='#e74c3c', alpha=0.3)
    ax2.plot(result.index, dd.values, color='#e74c3c', linewidth=0.8)
    ax2.set_title('回撤曲线', fontsize=14, fontweight='bold')
    ax2.set_ylabel('回撤幅度')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig2.autofmt_xdate()
    fig2.tight_layout()
    charts['drawdown'] = fig_to_base64(fig2)
    plt.close(fig2)

    # 图3: 年度收益
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    annual_ret_strat = result['strategy_return'].resample('YE').apply(lambda x: (1+x).prod()-1)
    annual_ret_bench = result['benchmark_return'].resample('YE').apply(lambda x: (1+x).prod()-1)
    years = [d.year for d in annual_ret_strat.index]
    x = np.arange(len(years))
    w = 0.35
    ax3.bar(x - w/2, annual_ret_strat.values, w, color='#e74c3c', alpha=0.8, label='策略')
    ax3.bar(x + w/2, annual_ret_bench.values, w, color='#3498db', alpha=0.8, label='基准')
    ax3.set_xticks(x)
    ax3.set_xticklabels(years)
    ax3.set_title('年度收益对比', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    fig3.tight_layout()
    charts['annual'] = fig_to_base64(fig3)
    plt.close(fig3)

    # 图4: 月度收益热力图
    monthly = result['strategy_return'].resample('ME').apply(lambda x: (1+x).prod()-1)
    monthly_df = monthly.groupby([monthly.index.year, monthly.index.month]).mean().unstack()
    monthly_df.columns = [f'{m}月' for m in monthly_df.columns]
    fig4, ax4 = plt.subplots(figsize=(10, max(4, len(monthly_df)*0.5)))
    im = ax4.imshow(monthly_df.values, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
    ax4.set_xticks(range(len(monthly_df.columns)))
    ax4.set_xticklabels(monthly_df.columns, rotation=45)
    ax4.set_yticks(range(len(monthly_df.index)))
    ax4.set_yticklabels(monthly_df.index)
    ax4.set_title('月度收益热力图', fontsize=14, fontweight='bold')
    for i in range(len(monthly_df.index)):
        for j in range(len(monthly_df.columns)):
            val = monthly_df.values[i, j]
            if not np.isnan(val):
                ax4.text(j, i, f'{val:.1%}', ha='center', va='center', fontsize=7,
                         color='white' if abs(val) > 0.05 else 'black')
    fig4.colorbar(im, ax=ax4, format=mticker.PercentFormatter(1.0))
    fig4.tight_layout()
    charts['heatmap'] = fig_to_base64(fig4)
    plt.close(fig4)

    # 构建 HTML 报告
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>量化策略绩效报告 - {strategy}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; background: #f5f6fa; color: #2c3e50; line-height: 1.6; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
.header {{ background: linear-gradient(135deg, #2c3e50, #34495e); color: white; padding: 40px 30px; border-radius: 10px; margin-bottom: 30px; }}
.header h1 {{ font-size: 28px; margin-bottom: 8px; }}
.header .meta {{ font-size: 14px; opacity: 0.85; }}
.card {{ background: white; border-radius: 10px; padding: 25px; margin-bottom: 25px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
.card h2 {{ font-size: 20px; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }}
.metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 15px; }}
.metric {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; }}
.metric .label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
.metric .value {{ font-size: 24px; font-weight: bold; margin-top: 4px; }}
.metric.positive .value {{ color: #27ae60; }}
.metric.negative .value {{ color: #e74c3c; }}
.chart-img {{ width: 100%; border-radius: 8px; margin-top: 10px; }}
.footer {{ text-align: center; color: #95a5a6; font-size: 12px; padding: 20px; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>量化策略绩效报告</h1>
  <div class="meta">
    策略: {strategy} | 标的: {symbol} |
    回测区间: {start_date} ~ {end_date or result.index[-1].strftime('%Y-%m-%d')} |
    生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </div>
</div>

<div class="card">
  <h2>绩效摘要</h2>
  <div class="metrics-grid">
    <div class="metric {'positive' if metrics['annual_return']>0 else 'negative'}">
      <div class="label">年化收益率</div>
      <div class="value">{metrics['annual_return']:.2%}</div>
    </div>
    <div class="metric">
      <div class="label">年化波动率</div>
      <div class="value">{metrics['annual_volatility']:.2%}</div>
    </div>
    <div class="metric {'positive' if metrics['sharpe_ratio']>0 else 'negative'}">
      <div class="label">夏普比率</div>
      <div class="value">{metrics['sharpe_ratio']:.2f}</div>
    </div>
    <div class="metric negative">
      <div class="label">最大回撤</div>
      <div class="value">{metrics['max_drawdown']:.2%}</div>
    </div>
    <div class="metric">
      <div class="label">Calmar比率</div>
      <div class="value">{metrics['calmar_ratio']:.2f}</div>
    </div>
    <div class="metric">
      <div class="label">胜率</div>
      <div class="value">{metrics['win_rate']:.2%}</div>
    </div>
    <div class="metric">
      <div class="label">基准年化收益</div>
      <div class="value">{metrics['benchmark_return']:.2%}</div>
    </div>
    <div class="metric {'positive' if metrics['excess_return']>0 else 'negative'}">
      <div class="label">超额收益</div>
      <div class="value">{metrics['excess_return']:.2%}</div>
    </div>
    <div class="metric">
      <div class="label">信息比率</div>
      <div class="value">{metrics['information_ratio']:.2f}</div>
    </div>
    <div class="metric">
      <div class="label">交易次数</div>
      <div class="value">{metrics['n_trades']}</div>
    </div>
  </div>
</div>

<div class="card">
  <h2>净值曲线</h2>
  <img class="chart-img" src="data:image/png;base64,{charts['nav']}" alt="净值曲线">
</div>

<div class="card">
  <h2>回撤分析</h2>
  <img class="chart-img" src="data:image/png;base64,{charts['drawdown']}" alt="回撤曲线">
</div>

<div class="card">
  <h2>年度收益</h2>
  <img class="chart-img" src="data:image/png;base64,{charts['annual']}" alt="年度收益">
</div>

<div class="card">
  <h2>月度收益热力图</h2>
  <img class="chart-img" src="data:image/png;base64,{charts['heatmap']}" alt="月度收益热力图">
</div>

<div class="footer">
  Quant Research Assistant | 量化投研助手 | 报告自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>

</div>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\n报告已生成: {output_path}")
    return output_path

# ============================================================
# 6. CLI 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Quant Research Assistant - 量化投研助手',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
        示例:
          python quant.py factor --universe hs300 --factor momentum_20d
          python quant.py backtest --strategy ma_cross --symbol 000300 --start 2020-01-01
          python quant.py portfolio --symbols 000300,000905,NH01535 --optimizer risk_parity
          python quant.py report --strategy ma_cross --symbol 000300 --output report.html
        ''')
    )

    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # factor 子命令
    factor_parser = subparsers.add_parser('factor', help='因子分析')
    factor_parser.add_argument('--universe', default='hs300',
                               choices=['hs300', 'zz500'],
                               help='股票池 (默认: hs300)')
    factor_parser.add_argument('--factor', default='momentum_20d',
                               help='因子名称 (如 momentum_20d)')
    factor_parser.add_argument('--period', default='2022-01-01:2023-12-31',
                               help='分析区间 (格式: start:end)')
    factor_parser.add_argument('--output-dir', default='./output',
                               help='输出目录 (默认: ./output)')

    # backtest 子命令
    bt_parser = subparsers.add_parser('backtest', help='策略回测')
    bt_parser.add_argument('--strategy', default='ma_cross',
                           choices=['ma_cross'],
                           help='策略名称 (默认: ma_cross)')
    bt_parser.add_argument('--symbol', default='000300',
                           help='标的代码 (默认: 000300)')
    bt_parser.add_argument('--start', default='2020-01-01',
                           help='回测起始日期')
    bt_parser.add_argument('--end', default=None,
                           help='回测结束日期')
    bt_parser.add_argument('--output-dir', default='./output',
                           help='输出目录 (默认: ./output)')

    # portfolio 子命令
    port_parser = subparsers.add_parser('portfolio', help='组合优化')
    port_parser.add_argument('--symbols', default='000300,000905,NH01535',
                             help='标的代码，逗号分隔 (默认: 000300,000905,NH01535)')
    port_parser.add_argument('--optimizer', default='risk_parity',
                             choices=['risk_parity', 'mean_variance'],
                             help='优化方法 (默认: risk_parity)')
    port_parser.add_argument('--start', default='2020-01-01',
                             help='起始日期')
    port_parser.add_argument('--end', default=None,
                             help='结束日期')
    port_parser.add_argument('--output-dir', default='./output',
                             help='输出目录 (默认: ./output)')

    # report 子命令
    report_parser = subparsers.add_parser('report', help='生成HTML报告')
    report_parser.add_argument('--strategy', default='ma_cross',
                               help='策略名称')
    report_parser.add_argument('--symbol', default='000300',
                               help='标的代码')
    report_parser.add_argument('--start', default='2020-01-01',
                               help='起始日期')
    report_parser.add_argument('--end', default=None,
                               help='结束日期')
    report_parser.add_argument('--output', default='report.html',
                               help='输出文件路径')
    report_parser.add_argument('--output-dir', default='./output',
                               help='临时图表输出目录')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # 解析 period
    start_date = None
    end_date = None
    if hasattr(args, 'period') and args.period:
        parts = args.period.split(':')
        start_date = parts[0]
        if len(parts) > 1:
            end_date = parts[1]
    if hasattr(args, 'start') and args.start:
        start_date = args.start

    # 确保输出目录存在
    output_dir = getattr(args, 'output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)

    if args.command == 'factor':
        run_factor_analysis(
            universe=args.universe,
            factor_name=args.factor,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
        )

    elif args.command == 'backtest':
        run_backtest(
            strategy=args.strategy,
            symbol=args.symbol,
            start_date=start_date,
            end_date=getattr(args, 'end', None),
            output_dir=output_dir,
        )

    elif args.command == 'portfolio':
        run_portfolio_optimization(
            symbols_str=args.symbols,
            optimizer=args.optimizer,
            start_date=start_date,
            end_date=getattr(args, 'end', None),
            output_dir=output_dir,
        )

    elif args.command == 'report':
        generate_html_report(
            strategy=args.strategy,
            symbol=args.symbol,
            start_date=start_date,
            end_date=getattr(args, 'end', None),
            output_path=args.output,
            output_dir=output_dir,
        )


if __name__ == '__main__':
    main()
