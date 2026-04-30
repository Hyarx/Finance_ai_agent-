from tools import get_sec_filings, analyze_sentiment

# Test SEC EDGAR
print("=== TEST SEC EDGAR ===")
print(get_sec_filings("AAPL"))
print(get_sec_filings("NVDA"))
print(get_sec_filings("MSFT"))

# Test sentiment
print("=== TEST SENTIMENT ===")
print(analyze_sentiment("Apple beats earnings with record revenue growth and strong margins"))
print(analyze_sentiment("Apple faces massive lawsuit and declining sales in China crisis"))
print(analyze_sentiment("Microsoft reports stable quarterly results"))