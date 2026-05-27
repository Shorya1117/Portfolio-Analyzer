# 📈 Wealth Portfolio Analyzer

A comprehensive financial analytics platform built using Streamlit that enables users to analyze and optimize portfolios across **stocks, mutual funds, and fixed deposits (FDs)** using advanced quantitative finance techniques.

This system goes beyond basic portfolio analysis by integrating **risk-adjusted metrics, macroeconomic factors, and credit-risk modeling**, simulating real-world investment decision frameworks.

---

## 🚀 What's New in v2.0

* ✅ Sharpe Ratio corrected using **risk-free rate (India 10Y G-Sec proxy)**
* ✅ Added **Sortino Ratio (downside risk)**
* ✅ Added **Maximum Drawdown & Calmar Ratio**
* ✅ Added **Value at Risk (VaR 95%)**
* ✅ Integrated **Live Inflation (World Bank API)**
* ✅ Integrated **Dynamic Risk-Free Rate (India)**
* ✅ Fixed Deposit upgraded with:

  * Credit risk modeling
  * Inflation-adjusted real returns
* ✅ Efficient Frontier improved (FD excluded for accuracy)
* ✅ Robust **column auto-detection system**
* ✅ Rolling returns & drawdown visualization
* ✅ Strict weight validation (execution blocking)

---

## 📌 Overview

The Wealth Portfolio Analyzer provides a unified environment to:

* Analyze multi-asset portfolios
* Evaluate risk-adjusted performance
* Incorporate macroeconomic factors (inflation & risk-free rate)
* Simulate optimal portfolio allocation
* Model real vs nominal returns

It represents a practical implementation of **Modern Portfolio Theory (MPT)** enhanced with **institutional risk metrics**.

---

## 🚀 Core Modules

### 🔥 1. Dynamic Multi-Asset Portfolio

* Combine **stocks + mutual funds + fixed deposits**
* Automatic time-series alignment
* Computes:

  * Portfolio Return & Volatility
  * Covariance & Correlation Matrix
  * Sharpe Ratio (risk-free adjusted)
* Efficient Frontier using Monte Carlo simulation
* Optimal portfolio selection

---

### 📊 2. Multi-Stock Portfolio

* Analyze multiple equities simultaneously
* Diversification and correlation analysis
* Risk-return optimization

---

### 🏦 3. Multi-Mutual Fund Portfolio

* NAV-based portfolio analysis
* Comparative performance evaluation
* Risk-adjusted metrics

---

### 📈 4. Single Asset Deep Dive

Supports both:

* Stocks
* Mutual Funds

Calculates:

* CAGR
* Volatility (Annualized)
* Sharpe Ratio
* Sortino Ratio
* Max Drawdown
* Calmar Ratio
* VaR (95%)
* Skewness & Kurtosis

---

### 🛡️ 5. Fixed Deposit & Credit Risk Engine

* Models FD as a **risk-adjusted fixed-income instrument**
* Incorporates:

  * Credit default probability
  * Partial recovery assumption
  * Inflation adjustment (Fisher Equation)
* Outputs:

  * Risk-adjusted CAGR
  * Real yield (purchasing power)
  * Survival probability
* Sensitivity analysis vs inflation

---

## 🧠 Advanced Financial Concepts Implemented

* Portfolio Return (Weighted Mean)
* Portfolio Risk: √(wᵀ Σ w)
* Sharpe Ratio (risk-free adjusted)
* Sortino Ratio (downside risk)
* Maximum Drawdown
* Calmar Ratio
* Value at Risk (VaR 95%)
* Efficient Frontier (Monte Carlo Simulation)
* CAGR (Compound Annual Growth Rate)
* Fisher Equation (Real Return)
* Credit Risk Modeling (Expected Payoff)

---

## 🌐 Macroeconomic Integration

* Live Inflation Data (World Bank API)
* Risk-Free Rate (India 10Y G-Sec Proxy)
* Manual override support

---

## 📊 Visual Analytics

* Price/NAV trends
* Return distributions with normal fit
* Efficient Frontier visualization
* Drawdown curves
* Rolling returns
* FD growth vs Inflation comparison

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Plotly
* SciPy
* Requests (API integration)

---

## ⚙️ How It Works

* Auto-detects financial data columns (Date, Price/NAV, Dividend)
* Converts data into return series
* Aligns multiple assets on a common timeline
* Integrates fixed-income instruments as risk-adjusted assets
* Uses statistical and probabilistic models for portfolio evaluation
* Simulates thousands of portfolios for optimization

---

## 🚀 How to Run

### 1. Clone Repository

git clone https://github.com/93527Rupali38898/stock-portfolio-analyser.git
cd stock-portfolio-analyser

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Run Application

streamlit run stock_analysis_app.py

👉 Open in browser: http://localhost:8501

---

## 📂 Input Format

### Market Assets (Stocks / Mutual Funds)

CSV must contain:

* Date
* Closing Price / NAV
* (Optional) Dividend

---

### Case Study

Year | Asset1 Returns | Asset2 Returns

---

## 🧠 Skills Demonstrated

* Financial Modeling
* Quantitative Analysis
* Portfolio Optimization
* Risk Management
* Statistical Modeling
* Data Visualization
* API Integration
* Dashboard Development

---

## 👩‍💻 Author

Rupali Goyal

---

## 📌 Note

This project is built for educational purposes and financial data analysis.
