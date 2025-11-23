# Reinforcement Learning for Stock Trading (DQN + TD3 + Sentiment + Multi‑Asset Portfolio)

## 📌 Abstract  
This project develops an advanced Reinforcement Learning (RL) framework for **algorithmic stock trading**, integrating:  
- **DQN** for single‑asset trading  
- **TD3** for multi‑asset continuous‑action portfolio allocation  
- **FinBERT-based sentiment analysis**  
- Technical indicators & engineered features  
- Multi‑asset environment with risk‑adjusted rewards  
- Evaluation with Sharpe, Sortino, Max Drawdown  
- Visualization of actions, weights, equity curve  

The system is designed for **robustness, interpretability, and high‑quality portfolio decisions**.

---

# 🚀 Features  
### ✔ Deep Q‑Learning (DQN)
- Works on single‑asset discrete Buy/Hold/Sell  
- Window‑based OHLCV + indicators  

### ✔ Twin Delayed Deep Deterministic Policy Gradient (TD3)
- Continuous portfolio weights  
- Allocates capital across **AAPL, MSFT, GOOG, NVDA**  
- Includes **soft target updates**, **delayed policy updates**, **Gaussian exploration noise**

### ✔ Sentiment Integration
- FinBERT sentiment score added as a feature  
- Optional reward shaping using positive/negative sentiment

### ✔ Robust Evaluation
Metrics include:  
- **Sharpe Ratio**  
- **Sortino Ratio**  
- **Max Drawdown**  
- **Total Return**  
- **Equity Curve**

### ✔ Professional‑quality Visualizations
- Equity curve  
- Multi‑asset buy/sell markers  
- Portfolio weights heatmap  
- Action arrow plots  
- Multi‑asset price panels  

---

# 📂 Project Structure
```
stock-trading-rl/
│
├── data/
├── models/
├── models_td3/
├── experiments/
│   └── eval_xxxxxx/
│        ├── equity_curve.png
│        ├── prices_with_arrows.png
│        ├── weights_heatmap.png
│        ├── metrics.csv
│        ├── weights_timeseries.csv
│        └── equity_timeseries.csv
│
├── src/
│   ├── train_dqn.py
│   ├── train_td3.py
│   ├── evaluate_td3.py
│   ├── robust_eval_td3.py
│   ├── env_trading.py
│   ├── env_portfolio.py
│   ├── sentiment_fetcher.py
│   ├── data_loader.py
│   └── models.py
│
├── README.md
└── config.yaml
```

---

# 🧠 Model Architectures  

## **DQN Architecture**
```
Input (window * features)
 → Linear(256) → ReLU
 → Linear(256) → ReLU
 → Dueling:
        Value head: Linear → 1
        Advantage head: Linear → actions
```

## **TD3 Architecture**
### Actor:
```
state_dim → 256 → 256 → tanh → portfolio weights
```
### Critics (Twin Q‑networks):
```
Concat(state, action) → 256 → 256 → Q-value
```

---

# 📊 Evaluation Results

## **📌 DQN (Single Asset – AAPL)**  
| Metric | Value |
|--------|--------|
| Final Net Worth | **109,019** |
| Sharpe Ratio | **0.011** |
| Sortino Ratio | **0.008** |
| Max Drawdown | **25,653** |

**DQN Equity Curve:**  
*(Image not embedded here, but included in repo)*

---

## **📌 TD3 (Multi‑Asset Portfolio)**  
| Metric | Value |
|--------|--------|
| Final Net Worth | **473,230** |
| Sharpe Ratio | **0.987** |
| Sortino Ratio | **0.973** |
| Max Drawdown | **187,452** |

**TD3 Equity Curve:**  
*(Image generated during evaluation)*

---

# 📈 Visualizations  

### ✔ Equity Curve  
Shows growth from $100,000 → $473,230$

### ✔ Multi‑Asset Buy/Sell Plots  
Per‑asset subplots with green (buy) and red (sell) markers

### ✔ Portfolio Weights Heatmap  
Displays which assets dominated the portfolio over time

---

# ⚙ Installation

### 1. Clone the repo
```bash
git clone https://github.com/Pranav-Adusumilli/stock-trading-rl.git
cd stock-trading-rl
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
pip install transformers yfinance torch gymnasium pandas matplotlib seaborn
```

### 3. Download FinBERT
```bash
python - << "EOF"
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
AutoModel.from_pretrained("yiyanghkust/finbert-tone")
EOF
```

---

# 🏋️ Training

## **Train DQN**
```bash
python -m src.train_dqn --config config.yaml
```

## **Train TD3 (Multi‑Asset)**
```bash
python -m src.train_td3 --config config_multi_td3.yaml
```

---

# 🧪 Evaluation

## **Evaluate TD3**
```bash
python -m src.evaluate_td3 --actor models_td3/actor_latest.pth --config config_multi_td3.yaml
```

Outputs saved to:
```
experiments/eval_YYYYMMDD_HHMMSS/
```

Includes:
- equity_curve.png  
- prices_with_arrows.png  
- weights_heatmap.png  
- metrics.csv  
- weights_timeseries.csv  

---

# 🏁 Final Notes
This project demonstrates:

- How RL can make intelligent trading decisions  
- How multi‑asset continuous RL (TD3) vastly outperforms discrete DQN  
- How sentiment can be fused with prices to create a smart hybrid agent  
  

