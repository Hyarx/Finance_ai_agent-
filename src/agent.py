import os
import chromadb
from agno.agent import Agent
from agno.team import Team
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from src.tools import calculate_financial_risk, analyze_sentiment, get_sec_filings, calculate_portfolio_performance

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "info9023-lab2")

def get_gemini():
    return Gemini(
        id="gemini-2.5-flash",
        vertexai=True,
        project_id=PROJECT_ID,
        location="us-central1"
    )

# --- RAG setup ---
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="risk_knowledge")



def load_documents(directory):
    documents, metadatas, ids = [], [], []
    for filename in os.listdir(directory):
        if not filename.endswith(".md"):
            continue
        with open(os.path.join(directory, filename)) as f:
            content = f.read()
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": filename, "chunk": i})
            ids.append(f"{filename}_{i}")
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Knowledge base loaded: {len(documents)} chunks")

load_documents("data/")

def search_knowledge_base(query: str) -> str:

    results = collection.query(query_texts=[query], n_results=3)
    formatted = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        formatted.append(f"[Source: {meta['source']}]\n{doc}")
    return "\n\n---\n\n".join(formatted)

# --- Agent 1 : Financial Analyst ---
financial_agent = Agent(
    name="Financial Analyst",
    model=get_gemini(),
    tools=[
        YFinanceTools(
            enable_stock_price=True,
            enable_company_info=True,
            enable_company_news=True,
            enable_analyst_recommendations=True,
        ),
        calculate_financial_risk,
        calculate_portfolio_performance,
    ],
    instructions="""You are a quantitative financial analyst.
    When asked to analyse a portfolio:
    IMMEDIATELY call calculate_financial_risk with all tickers and weights.
    THEN call calculate_portfolio_performance with same tickers and weights.
    Return ALL results without additional analysis.
    NEVER truncate or summarize tool outputs.
    DO NOT ask clarifying questions. DO NOT explain what you are doing.
    Just call the tools and return the raw results.""",
)
# --- Agent 2 : News & Sentiment ---
news_agent = Agent(
    name="News Analyst",
    model=get_gemini(),
    tools=[
        YFinanceTools(
            enable_stock_price=False,
            enable_company_news=True,
        ),
        analyze_sentiment,
    ],
    instructions="""You are a financial news analyst.
    For EACH ticker in the portfolio, call get_company_news separately.
    Then call analyze_sentiment ONCE on ALL combined news titles and summaries.
    Return sentiment score for each company separately.
    DO NOT make more than N+1 tool calls (N = number of tickers + 1 sentiment call).
    Be direct and concise.""",
)

# --- Agent 3 : Audit & Compliance ---
audit_agent = Agent(
    name="Audit Expert",
    model=get_gemini(),
    tools=[search_knowledge_base, get_sec_filings],
    instructions="""You are an audit and compliance expert.
    For each company (or each company in a portfolio):
    - Use get_sec_filings to retrieve official SEC 10-K filings
    - Use search_knowledge_base to check against ISA standards,
      financial ratios benchmarks and audit red flags
    - Identify compliance issues, accounting anomalies or red flags
    - For portfolios, assess each company individually then give
      an overall compliance summary""",
)
# --- Team ---
team = Team(
    name="Company Risk Intelligence Team",
    mode="coordinate",
    model=get_gemini(),
    members=[financial_agent, news_agent, audit_agent],
    instructions="""You lead a team of financial risk experts.

    Delegate ALL tasks simultaneously:
    1. Financial Analyst: calculate_financial_risk + calculate_portfolio_performance
    2. News Analyst: analyze_sentiment
    3. Audit Expert: get_sec_filings + search_knowledge_base

    Final report structure:

    ## 1. Financial Risk
    Include ALL numbers: GARCH (alpha, beta, persistence), volatility, 
    VaR 99%/95%, ES 97.5%, ratios (ROE, P/E, D/E, Current Ratio), 
    Altman Z-Score, Sharpe Ratio. Interpret each metric briefly.

    ## 2. News Sentiment
    Sentiment score, polarity, subjectivity, key risks and opportunities.

    ## 3. Audit & Compliance
    10-K date, Revenue, Net Income, Assets, Liabilities, Cash Flow, 
    compliance status, red flags.

    ## 4. Conclusion
    Risk level, key strengths, key risks, recommendation (Hold/Buy/Sell).

    Include ALL numbers. Interpret briefly. Be structured.""",
    markdown=True,
    debug_mode=True,
)
if __name__ == "__main__":
    team.print_response(
        "Analyse the financial risk of Apple (AAPL)",
        stream=True
    )