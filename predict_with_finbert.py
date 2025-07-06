import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from lstm_model import LSTM_CNN_Attention
from finbert_sentiment import fetch_apple_news, get_sentiment_score

# ── Load CSV ────────────────────────────────────────────────────────────────
csv = pd.read_csv("May2025Apple.csv")
csv["Date"] = pd.to_datetime(csv["Date"], format="%Y-%m-%d")
csv.set_index("Date", inplace=True)
csv["Label"] = (csv["Close"].shift(-1) > csv["Close"]).astype(int)
csv = csv.dropna(subset=["Close"])
print("✅ CSV loaded. First rows:\n", csv.head(), "\n")

# ── Load Model ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTM_CNN_Attention(input_size=14).to(device)
model.load_state_dict(torch.load("pre-may-model.pth", map_location=device))
model.eval()

# ── Load Features ────────────────────────────────────────────────────────────
X_test = torch.tensor(np.load("X_test_pre_may.npy"), dtype=torch.float32)
target_dates = list(csv.index)
loader = DataLoader(TensorDataset(X_test[:len(target_dates)]), batch_size=1)

# ── Evaluation ───────────────────────────────────────────────────────────────
correct_with_bert = 0
correct_without_bert = 0
total = len(target_dates)

print("🔍 Comparison of predictions:\n")

for j, (xb,) in enumerate(loader):
    xb = xb.to(device)
    date_dt = target_dates[j]
    date_str = date_dt.strftime("%Y-%m-%d")

    # LSTM model prediction
    with torch.no_grad():
        probs = torch.softmax(model(xb), dim=1)[0]
    prob_up, prob_down = probs[1].item(), probs[0].item()
    pred_without_bert = 1 if prob_up > prob_down else 0

    # FinBERT news sentiment
    news = fetch_apple_news(date_str)
    sent_score = get_sentiment_score(news) if news else 0.0
    sent_conf = (sent_score + 1) / 2  # Normalize to [0, 1]

    # Combine LSTM and FinBERT (50/50)
    comb_up = 0.5 * prob_up + 0.5 * sent_conf
    comb_down = 0.5 * prob_down + 0.5 * (1 - sent_conf)
    pred_with_bert = 1 if comb_up > comb_down else 0

    # Compare with true label
    true = int(csv.loc[date_dt, "Label"])
    correct_with_bert += (pred_with_bert == true)
    correct_without_bert += (pred_without_bert == true)

    # Output
    print(f"\n[{date_str}] True: {true} | No BERT: {pred_without_bert} | BERT: {pred_with_bert} | Sentiment: {sent_score:+.3f}")
    print(f"   🔗 Blended Prob UP: {comb_up:.3f} | DOWN: {comb_down:.3f}")
    if news:
        print(f"   ✅ News used ({len(news)} headlines):")
        for i, headline in enumerate(news):
            print(f"     [{i+1}] {headline}")
    else:
        print("    No news used")

# ── Accuracy Summary ─────────────────────────────────────────────────────────
print("\ Accuracy Results:")
print(f" Without FinBERT: {correct_without_bert}/{total} = {correct_without_bert / total:.4f}")
print(f" With FinBERT:    {correct_with_bert}/{total} = {correct_with_bert / total:.4f}")
