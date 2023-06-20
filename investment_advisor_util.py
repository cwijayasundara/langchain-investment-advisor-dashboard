def format_large_number(num):
    if abs(num) >= 1_000_000_000_000:
        return f"${num / 1_000_000_000_000:.2f}T"
    elif abs(num) >= 1_000_000_000:
        return f"${num / 1_000_000_000:.2f}B"
    elif abs(num) >= 1_000_000:
        return f"${num / 1_000_000:.2f}M"
    else:
        return str(num)


stocks = {
    "Apple - 'AAPL'": {"name": "APPLE INC", "symbol": "AAPL", "cik": "0000320193", "url" : "https://www.apple.com"
                                                                                           "/newsroom/pdfs/FY23_Q2_Consolidated_Financial_Statements.pdf"},
    "Alphabet - 'GOOG'": {"name": "Alphabet Inc.", "symbol": "GOOG", "cik": "0001652044", "url" : "https://abc.xyz"
                                                                                                  "/investor/static/pdf/2023Q1_alphabet_earnings_release.pdf?cache=0924ccf"},
    "Facebook - 'META'": {"name": "META PLATFORMS INC", "symbol": "META", "cik": "0001326801"},
    "Amazon - 'AMZN'": {"name": "AMAZON COM INC", "symbol": "AMZN", "cik": "0001018724"},
    "Netflix - 'NFLX'": {"name": "NETFLIX INC", "symbol": "NFLX", "cik": "0001065280"},
    "Microsoft - 'MSFT'": {"name": "MICROSOFT CORP", "symbol": "MSFT", "cik": "0000789019"},
    "Tesla - 'TSLA'": {"name": "TESLA INC", "symbol": "TSLA", "cik": "0001318605"}
}

ten_k_file_url_dict = {
    "Apple": "https://www.apple.com/newsroom/pdfs/FY23_Q2_Consolidated_Financial_Statements.pdf",
    "Alphabet": "https://abc.xyz/investor/static/pdf/2023Q1_alphabet_earnings_release.pdf?cache=0924ccf"
}
