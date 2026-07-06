---
name: "quant-research-assistant"
description: "Quantitative investment research: factor analysis, backtesting, portfolio optimization. Invoke when user asks for quantitative analysis, factor research, backtesting, or portfolio construction."
---

# 量化投研助手 (Quant Research Assistant)

量化投资研究的全流程工具。支持因子挖掘与测试、策略回测、组合优化三大核心能力。

## 触发条件

- "帮我做因子分析" / "测试一下这个因子" / "回测一下这个策略"
- "组合优化" / "资产配置" / "有效前沿"
- "quantitative analysis" / "factor research" / "portfolio optimization"

## 核心能力

### 1. 因子分析
- 单因子 IC 测试（IC均值/ICIR/t统计量）
- 分层回测（5/10组）— 各组收益曲线
- 因子正交化与合成
- 因子相关性矩阵

### 2. 策略回测
- 支持规则型策略（均线/动量/反转/突破）
- 完整绩效指标：年化收益/夏普/最大回撤/Calmar/胜率
- 交易成本建模（佣金+滑点）
- TeX风格的LaTeX绩效报告

### 3. 组合优化
- 均值-方差模型（Markowitz）
- 风险平价（Risk Parity）
- Black-Litterman模型
- 有效前沿可视化
- 鲁棒优化（shrinkage estimation）

## 使用方法

```
python quant.py factor --universe hs300 --factor momentum_20d
python quant.py backtest --strategy ma_cross --start 2020-01-01
python quant.py portfolio --assets AAPL,TSLA,GOOGL,MSFT,AMZN --optimizer risk_parity
python quant.py report --strategy momentum --output report.pdf
```

## 技术栈
- 数值计算: numpy, scipy (optimization)
- 数据获取: akshare
- 可视化: matplotlib, plotly
- 报告: LaTeX模板
