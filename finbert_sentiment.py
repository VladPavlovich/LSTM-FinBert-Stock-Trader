from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import requests
from datetime import datetime, timedelta

# Load FinBERT model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
model.eval()

# In-memory cache
_sentiment_cache = {}

# Optional keyword sentiment boost
boost_keywords = {
    "surge": 0.05,
    "bullish": 0.05,
    "rally": 0.05,
    "strong demand": 0.05,
    "record": 0.05,
    "upgrade": 0.03,
    "beat": 0.03,
    "AI": 0.02
}

def get_sentiment_score(texts):
    """Compute boosted average sentiment score from FinBERT."""
    scores = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)

        # Base sentiment: pos - neg
        score = probs[0][2] - probs[0][0]
        base_score = score.item()

        # Optional keyword-based boost
        boost = sum(weight for kw, weight in boost_keywords.items() if kw.lower() in text.lower())
        final_score = base_score + boost

        # Logging
        print(f"\n📈 Text: {text}")
        print(f"🔍 Probabilities: {probs.tolist()[0]}")
        print(f"✨ Final Sentiment Score (with boost if needed): {final_score:+.3f}")
        scores.append(final_score)

    return sum(scores) / len(scores) if scores else 0.0


def fetch_apple_news(date, api_key="c69d24631c9249a6ad530e984ba3889e", max_lookback=5):
    """
    Fetch headlines about Apple for a specific date using NewsAPI.
    If none found, look back up to `max_lookback` previous days for real headlines.
    """
    if date in _sentiment_cache:
        return _sentiment_cache[date]

    date_dt = datetime.strptime(date, "%Y-%m-%d")
    for i in range(max_lookback + 1):
        try_date = (date_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        if try_date in _sentiment_cache:
            return _sentiment_cache[try_date]

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "Apple OR AAPL",
            "from": try_date,
            "to": try_date,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 30,
            "apiKey": api_key
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()
            headlines = [a["title"] for a in data.get("articles", [])]
            if headlines:
                _sentiment_cache[try_date] = headlines
                if i > 0:
                    print(f" Using headlines from {try_date} for missing {date}")
                return headlines
        except Exception as e:
            print(f"Error fetching news for {try_date}: {e}")

    print(f" No headlines found within {max_lookback} days of {date}")
    return []


def predict_with_finbert(sentiment_score, base_prediction):
    """Use FinBERT sentiment to override base prediction if strong enough."""
    if sentiment_score >= 0.15:
        return 1  # strong positive sentiment
    elif sentiment_score <= -0.15:
        return 0  # strong negative sentiment
    else:
        return base_prediction  # fallback to LSTM prediction
