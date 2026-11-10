import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS

# 1. Dataset: Original 18 comments + 1 negation test comment, with added topic classification
reviews = [
    "The bottle keeps my water cold all day.",
    "Great size and easy to carry to class.",
    "I love the simple design and sturdy lid.",
    "It leaks in my bag. Very disappointing.",
    "The cap is hard to open and the bottle feels heavy.",
    "Good value, but cleaning the narrow opening is difficult.",
    "Keeps drinks cold, though the paint scratches easily.",
    "The bottle is fine. Nothing special either way.",
    "I expected better insulation for this price.",
    "Lightweight, useful, and easy to wash.",
    "Not bad, but I would prefer a larger capacity.",
    "Excellent bottle. I use it every day.",
    "The lid broke after one week. Poor quality.",
    "It works as described and arrived on time.",
    "Nice color, but it does not keep drinks cold.",
    "Affordable and practical for carrying water.",
    "The bottle is okay; the handle is uncomfortable.",
    "Very happy with the purchase and the quality.",
    "The bottle is not easy to clean."  # Added for negation test (Lesson 3, Step 3)
]

topics = [
    "insulation", "portability", "design", "leakage", "cap/weight",
    "cleaning", "insulation", "general", "insulation/price", "cleaning/weight",
    "capacity", "general", "lid/quality", "general", "insulation",
    "price/portability", "handle", "general", "cleaning"
]

df = pd.DataFrame({
    "id": [f"R{i:02}" for i in range(1, 20)],
    "text": reviews,
    "topic": topics
})

# 2. Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

def label_sentiment(text):
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive", score
    if score <= -0.05:
        return "Negative", score
    return "Neutral", score

df[["sentiment", "compound"]] = df["text"].apply(
    lambda text: pd.Series(label_sentiment(text))
)

# 3. Streamlit Dashboard
st.set_page_config(page_title="Lab 8 | Feedback", layout="wide")
st.title("Product Feedback: A Text Summary")
st.caption("Practice data only. Automated labels need human review.")

left, right = st.columns(2)

with left:
    st.subheader("Comment counts by sentiment")
    order = ["Positive", "Neutral", "Negative"]
    counts = df["sentiment"].value_counts().reindex(order, fill_value=0)
    st.bar_chart(counts, horizontal=True)
    
    st.subheader("Comment counts by topic")
    topic_counts = df["topic"].value_counts()
    st.bar_chart(topic_counts, horizontal=True)

with right:
    st.subheader("Words used in the comments")
    
    # IMPROVEMENT: Added "water" to stopwords. 
    # Reason: "water" is highly frequent but uninformative since it is a water bottle. 
    # Retaining terms like "lid", "cold", and "clean" ensures readers notice specific 
    # functional feedback rather than just the generic product category.
    stop_words = set(STOPWORDS) | {"bottle", "drinks", "drink", "water"}
    
    cloud = WordCloud(
        width=900, height=500, background_color="white", 
        colormap="viridis", stopwords=stop_words, collocations=False
    ).generate(" ".join(df["text"]))
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)
    
    # Added caption explaining word size and its limits
    st.caption("Word size indicates relative frequency after removing common filler words. Position and color are arbitrary and do not represent sentiment.")

st.subheader("Review the original comments")
st.dataframe(df[["id", "text", "topic", "sentiment", "compound"]], use_container_width=True, hide_index=True)
st.caption("A compound score is a model output, not a confidence rating.")