---
name: "quant-research-assistant"
description: "Quantitative investment research: factor IC testing, layer backtest, MA crossover backtesting, Markowitz/Risk Parity portfolio optimization, HTML report. Invoke when user asks for quantitative analysis, factor research, backtesting, or portfolio construction."
---

# 量化投研助手 (Quant Research Assistant)

量化投资研究工具。支持单因子IC测试、分层回测、均线交叉策略回测、组合优化（Markowitz / Risk Parity）和 HTML 报表生成。

## 能力边界

### ✅ 已实现的功能
- **因子分析**: 单因子 IC 测试（IC 均值 / ICIR / t 统计量），5 分组分层回测及累计收益曲线
- **策略回测**: 均线交叉策略（MA5/MA20），完整绩效指标（年化收益 / 夏普 / 最大回撤 / Calmar / 胜率 / 信息比率）
- **组合优化**: 均值-方差模型（Markowitz，有效前沿图）、风险平价（Risk Parity）
- **报告生成**: 自包含 HTML 报告（内嵌 base64 图表：净值曲线 / 回撤 / 年度收益 / 月度热力图）
- **数据源**: 沪深 300 / 中证 500 成分股，通过 akshare 获取真实行情

### ❌ 暂不支持的功能
- **不包含因子正交化与合成**（可手动在代码中扩展）
- **不包含 Black-Litterman 模型**（目前仅支持 Markowitz 和 Risk Parity）
- **策略类型仅含均线交叉**（后续版本会增加动量、反转、突破等策略）
- **报告格式仅 HTML**（不生成 LaTeX 或 PDF）
- **离线不可用**（数据需通过 akshare 联网获取）

## 触发条件

- "帮我做因子分析" / "测试一下这个因子" / "回测一下这个策略"
- "组合优化" / "资产配置" / "有效前沿" / "风险平价"
- "quantitative analysis" / "factor research" / "portfolio optimization"

## 使用方法

```bash
# 因子分析：计算动量因子的 IC，做5分组回测
python quant.py factor --universe hs300 --factor momentum_20d --period 2022-01-01:2023-12-31

# 策略回测：MA5/MA20 均线交叉策略
python quant.py backtest --strategy ma_cross --symbol 000300 --start 2020-01-01

# 组合优化：风险平价
python quant.py portfolio --symbols 000300,000905,NH01535 --optimizer risk_parity

# 组合优化：均值-方差 + 有效前沿图
python quant.py portfolio --symbols 000300,000905 --optimizer markowitz

# 生成 HTML 报告
python quant.py report --strategy ma_cross --symbol 000300 --output report.html
```

## 输出示例

### 因子分析输出（终端）
```
沪深300成分股 动量_20d 因子分析 (2022-01-01 ~ 2023-12-31)
============================================================
IC 均值:   0.034
ICIR:      0.521
IC > 0 比例: 61.8%
t 统计量:   1.962
显著性:    显著 (p < 0.05)

分层回测（5组）:
Q1 (最高): 累计收益 +12.3%  [====================]
Q2:       累计收益 +6.8%   [===========]
Q3:       累计收益 +1.2%   [==]
Q4:       累计收益 -3.5%   [==-------]
Q5 (最低): 累计收益 -8.1%  [---------------]

图表已保存: factor_momentum_20d.png
```

### 策略回测输出
```
均线交叉策略回测结果 (000300, 2020-01-01 ~ 今)
==============================================
年化收益率:   8.65%
夏普比率:     0.72
最大回撤:     18.3%
Calmar比率:   0.47
胜率:         42.1%
信息比率:     0.55
总交易次数:   47
```

## 常见问题 (FAQ)

### Q: 运行报错怎么办？
A: 先确认 `pip install -r requirements.txt` 已执行。如果 akshare 接口返回空数据，可能是网络问题，建议检查网络连接后重试。

### Q: 回测结果为什么看起来不理想？
A: 均线交叉策略在震荡市中会频繁产生假信号。这是策略本身特性的体现，不是工具的 bug。建议调整回测周期或参数进一步测试。

### Q: 为什么只支持均线交叉一种策略？
A: 当前版本聚焦于实现正确的回测框架和绩效评估体系。后续版本会扩展更多策略类型（动量、反转、突破等）。

### Q: 因子分析的股票池怎么选？
A: 支持 `--universe hs300`（沪深300）和 `--universe zz500`（中证500）。因子默认为 `momentum_20d`（20日动量）。

### Q: 可以自定义因子吗？
A: 可以修改 `quant.py` 中的因子计算逻辑。核心框架支持扩展。

## 技术栈
- 数值计算: numpy, scipy (scipy.optimize.minimize)
- 数据获取: akshare
- 可视化: matplotlib (图表), plotly (报告)
- 报告: HTML (自包含，内嵌 base64 图表)

## 依赖
- numpy >= 1.24.0
- scipy >= 1.10.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- akshare >= 1.10.0
- plotly >= 5.0.0
