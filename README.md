# Finance_ai_agent


This project is a multi-agent AI system that analyses the financial 
risk of US stocks and portfolios. You ask a question in natural 
language and the system returns a complete risk report.

## What it does

Given a stock like Apple (AAPL) or a portfolio like 50% Apple and 
50% NVIDIA, the system automatically:
- Calculates financial risk using GARCH, VaR and Expected Shortfall
- Analyses recent news sentiment
- Checks official SEC financial statements
- Suggests an optimal portfolio allocation using Sharpe ratio

## How it works

The system uses 3 specialized AI agents coordinated by a Team Leader:

- **Financial Analyst** — computes risk metrics (GARCH, VaR, Altman Z-Score)
- **News Analyst** — analyses news sentiment with TextBlob NLP
- **Audit Expert** — checks SEC EDGAR filings and audit standards

Each agent uses custom Python tools and a RAG knowledge base built 
with ChromaDB containing financial standards (Basel III, ISA, IFRS).

## How to use it
Run app.py in one terminal and use the following command in a second terminal:

```bash
curl -X POST https://YOUR_URL/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Analyse the financial risk of Apple (AAPL)"}'
```

## Tech Stack

Agno · Google Gemini 2.5 Flash · GARCH · Monte Carlo · 
ChromaDB · SEC EDGAR · Flask · Google Cloud Run

## Author

Alexandre Dutailly — Master in Data Science & AI, University of Liège
