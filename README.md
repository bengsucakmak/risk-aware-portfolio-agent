# Risk-Aware Portfolio Management Agent
**Reinforcement Learning–Based Decision System with Explicit Risk Awareness**

---

## Executive Summary

Risk-Aware Portfolio Management Agent is a **reinforcement learning–based decision-making system** designed to manage financial portfolios while explicitly accounting for **risk exposure**, not just returns.

Unlike conventional RL trading agents that optimize cumulative reward alone, this system incorporates **volatility, drawdown, and investor risk preferences** directly into the learning process, resulting in **more stable and interpretable portfolio behavior**.

The project emphasizes **decision quality, risk control, and explainability** over short-term profit maximization.

---

## Motivation

Most RL-based portfolio management approaches:
- Focus purely on return maximization
- Exhibit unstable behavior under volatile market conditions
- Ignore investor-specific risk preferences
- Provide little to no explanation for their decisions

This project addresses these limitations by designing a **risk-sensitive RL framework** that:
- Penalizes excessive volatility and drawdowns
- Adapts behavior based on user-defined risk profiles
- Produces interpretable actions aligned with risk tolerance
- Prioritizes long-term stability over aggressive gains

---

## System Architecture

The system is designed as a layered architecture separating **data processing, environment dynamics, agent learning, explainability, and presentation**.

![System Architecture](assets/rl_system_architecture.png)

---

## Architecture Breakdown

### 1. User Layer
- Risk profile selection:
  - Conservative
  - Balanced
  - Aggressive
- Virtual portfolio management (paper trading)
- Optional user approval for agent decisions

---

### 2. Data Layer
- Market data ingestion:
  - Asset prices
  - Technical indicators
- Feature engineering:
  - RSI
  - Bollinger Bands
  - ATR
- Train / test split and normalization

---

### 3. Reinforcement Learning Environment (RiskAwareEnv)

#### State Space
- Cash balance
- Asset quantities
- Current prices
- Technical indicators

#### Action Space
- Buy / Sell / Hold ratios per asset
- Portfolio reallocation decisions

#### Reward Function
The reward function explicitly balances **return and risk**:

- Positive reward for portfolio returns
- Penalties for:
  - High volatility
  - Large drawdowns
- Risk-profile-dependent weighting coefficient (λ)

This formulation allows the agent to **learn different behaviors** under different risk preferences.

---

### 4. RL Agent

- Algorithm: **Proximal Policy Optimization (PPO)**
- Policy Network: Multi-layer perceptron (MLP)
- Training phase:
  - Risk-profile-conditioned learning
- Inference phase:
  - Daily portfolio allocation decisions

---

### 5. Model Outputs

- Daily action vector
- Buy / Sell / Hold decisions
- Action intensity (signal strength)

---

### 6. Explainability Layer

The explainability layer answers the question:

**“Why did the agent take this action?”**

- Interpretation of signal strength
- Alignment with selected risk profile
- Sensitivity analysis of contributing factors:
  - Momentum
  - Volatility
  - Trend indicators

This layer improves trust and usability of RL-driven decisions.

---

### 7. Feedback & Evaluation

- Portfolio state updated after each action
- Performance metrics:
  - Total return
  - Sharpe ratio
  - Volatility
  - Maximum drawdown
- Continuous feedback loop between environment and agent

---

### 8. Presentation & Reporting

- Streamlit-based dashboard
- Step-by-step portfolio simulation
- Backtesting and ablation results
- Automated PDF report generation for performance summaries

---

## Reinforcement Learning Strategy

- Policy-based reinforcement learning (PPO)
- Risk-aware reward shaping
- Stable learning under market uncertainty
- Clear separation between training and inference workflows

---

## Key Design Decisions

- **Risk-aware reward formulation** instead of pure return optimization
- **Investor profile conditioning** for adaptable behavior
- **Explainability as a core component**, not an afterthought
- **Decision-centric evaluation** using financial risk metrics

---

## Experimental Results & Insights

- Risk-aware agents exhibit:
  - Lower volatility
  - Reduced maximum drawdown
  - More stable long-term performance
- Compared to risk-neutral agents:
  - Slightly lower peak returns
  - Significantly improved consistency and robustness

This tradeoff reflects **realistic investment behavior** rather than speculative strategies.

---

## Technologies Used

- Python
- PyTorch
- Stable-Baselines3
- NumPy, Pandas
- Matplotlib
- OpenAI Gym-style custom environments
- Streamlit

---

## Future Work

- Multi-risk-profile ensemble agents
- Online learning with market regime adaptation
- Transaction cost and slippage modeling
- Deployment as a decision-support service
- Integration with live market data APIs

---

## License

MIT License
