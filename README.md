# 🧠 FinWise Core
### Institutional-Grade AI Financial Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-blueviolet?style=for-the-badge&logo=neo4j)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![Gradio](https://img.shields.io/badge/Gradio-Terminal-orange?style=for-the-badge&logo=gradio)

**FinWise Core** is a modular, high-frequency financial intelligence engine designed for professional market analysis. It combines **probabilistic price forecasting** (LSTM + Heston Monte Carlo) with **real-time sentiment analysis** (FinBERT + News Pipeline) and a **Knowledge Graph (KG)** backbone to deliver regime-aware market insights.

---

## 🏗️ Modular Architecture

The project follows a **PM-grade Domain-Driven Design (DDD)** structure to ensure scalability, maintainability, and clear separation of concerns.

```mermaid
graph TD
    User([User])
    
    subgraph "App Layer"
        SL[Streamlit UI<br>(Recall/Chat)]
        GR[Gradio Terminal<br>(Risk/Quant)]
    end
    
    subgraph "Core Layer"
        ORC[Orchestrator]
        CFG[Config]
    end
    
    subgraph "Intelligence Layer"
        P1[Price Engine<br>(LSTM + Heston MC)]
        P2[Sentiment Engine<br>(FinBERT + Pipeline)]
    end
    
    subgraph "Data Layer"
        KG[(Neo4j Knowledge Graph)]
        RE[Recall Engine<br>(Vector/SQL)]
    end
    
    User --> SL & GR
    SL & GR --> ORC
    ORC --> P1 & P2
    P1 & P2 --> KG
    SL --> RE
```

## 📂 Project Structure

| Directory | Component | Description |
|-----------|-----------|-------------|
| **`app/`** | Application Layer | Entry points for granular UI experiences. |
| **`core/`** | Core Logic | Shared configuration, state orchestration, and utilities. |
| **`intelligence/`** | Intelligence Engines | **P1 (Price)**: Deep learning models.<br>**P2 (Sentiment)**: NLP pipelines. |
| **`kg/`** | Knowledge Graph | Neo4j writers, schema definitions, and state adapters. |
| **`recall_engine/`** | Recall & Chat | RAG systems, LLM query generation, and conversation state. |

---

## 🚀 Key Features

### 1. 📈 Probabilistic Price Intelligence (P1)
- **Deep Learning**: LSTM-based price forecasting trained on 3+ years of OHLCV data.
- **Risk Modeling**: Heston Stochastic Volatility Monte Carlo simulations (5000+ paths).
- **Regime Awareness**: Automatic detection of Bullish, Bearish, or Sideways market regimes.
- **Asset Context**: Relative Value at Risk (VaR) calculation.

### 2. 🧠 Semantic Sentiment Intelligence (P2)
- **FinBERT Integration**: Institutional-grade sentiment analysis tailored for financial texts.
- **R2 Relevance Filter**: Smart filtering to exclude noise (marketing, SEO spam, non-financial news).
- **Aspect-Based Analysis**: Decomposes news into factors like *Earnings*, *Litigation*, *M&A*, etc.
- **Impact Scoring**: Weighted scoring based on source credibility, recency, and specific entity mentions.

### 3. 🕸️ Knowledge Graph (KG)
- **Neo4j AuraDB**: graph-native storage for complex relationships (Company -> Metric -> Value).
- **Temporal Tracking**: Tracks financial state snapshots over time.

### 4. 🗣️ Conversational AI (Recall)
- **Gemini Pro Integration**: Context-aware chat assistant for financial queries.
- **Natural Language Querying**: Converts user questions ("How is Apple doing?") into precise database queries.

---

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- Git
- Neo4j AuraDB Instance
- Google Gemini API Key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/finwise_core.git
   cd finwise_core
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   python -m textblob.download_corpora
   ```

4. **Configure Environment**
   - Copy `.env.example` to `.env`.
   - Fill in your API keys (Neo4j, Gemini, NewsAPI).

   ```bash
   cp .env.example .env
   ```

---

## 🖥️ Usage

Run the module corresponding to your role:

### 1. Analyst Chat Interface (Streamlit)
*Best for: Deep dives, conversational research, database querying.*

```bash
streamlit run app/streamlit_app.py
```

### 2. Quant Risk Terminal (Gradio)
*Best for: Fast lookup, risk metrics, technical charts, Monte Carlo visualization.*

```bash
python app/gradio_app.py
```

---

## 🤝 Contribution (Internal)

1. **Strict Git Discipline**: Meaningful commit messages are mandatory.
2. **Modular Dev**: Place new intelligence logic in `intelligence/` (create P3, P4 if needed).
3. **No Direct Database Writes**: Use the `kg_writer` module.

---

*© 2024 FinWise Financial Intelligence. Internal Use Only.*
