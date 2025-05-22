# Quantitative Trading Strategy:  Multi-Indicator Combination

![Strategy Overview]
![image](https://github.com/user-attachments/assets/f23565ea-623e-4949-9720-57ed01471dff)


A quantitative trading strategy combining **Bollinger Bands**, **Relative Strength Index (RSI)**, and **Moving Average Convergence Divergence (MACD)** to generate buy/sell signals.

## 📌 Strategy Logic
### Indicator Combination Rules
1. **Bollinger Bands**  
   - Middle Band: 20-day Simple Moving Average (SMA)  
   - Upper/Lower Bands: Middle Band ± 2 Standard Deviations  
   - **Buy Signal**: Price breaks below Lower Band then crosses back above  
   - **Sell Signal**: Price breaks above Upper Band then crosses back below  

2. **RSI**  
   - Period: 14 days  
   - **Oversold Signal** (Buy): RSI < 30  
   - **Overbought Signal** (Sell): RSI > 70  

3. **MACD**  
   - Fast Line: 12-day EMA  
   - Slow Line: 26-day EMA  
   - Signal Line: 9-day EMA  
   - **Buy Signal**: MACD Line (Fast-Slow) crosses above Signal Line  
   - **Sell Signal**: MACD Line crosses below Signal Line  

### Multi-Indicator Synergy
- **Entry Condition**: Bollinger Buy Signal + RSI Oversold + MACD Golden Cross  
- **Exit Condition**: Bollinger Sell Signal + RSI Overbought + MACD Death Cross  

## Backtest_result
  - Initial capital: $10,000.00
  - Final assets: $30,108.70
  - Cumulative return: 201.09%
  - Annualized return: 4.44%
  - Sharpe ratio: 0.40
  - Maximum drawdown: 90.90%
  - Volatility: 46.84%
  - Number of transactions: 55 times (27 full rounds)
  - Win rate: 100.0%
  - Profit-loss ratio: ∞
  - Alpha/Beta: -0.14/1.00

## 🛠 Quick Start
### Prerequisites
```bash
pip install pandas numpy matplotlib talib alpha_vantage  # Quantitative analysis libraries
