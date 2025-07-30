import streamlit as st
import pandas as pd
from analysis import load_dataset, analyze_sentiments, detect_fake_reviews
import plotly.express as px
from datetime import datetime

# Configuration
st.set_page_config(layout="wide")
st.title("Amazon Review Analysis")

# Load data
df = load_dataset()

if not df.empty:
    # Convert date to datetime and extract year/month
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    
    # Analysis pipeline
    df = analyze_sentiments(df)
    df = detect_fake_reviews(df)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", len(df))
    col2.metric("Average Rating", f"{df['rating'].mean():.1f} ★")
    col3.metric("Positive Sentiment", 
                f"{len(df[df['sentiment_label'] == 'positive']) / len(df) * 100:.1f}%")
    col4.metric("Suspicious Reviews",
                f"{df['is_fake'].mean() * 100:.1f}%",
                delta_color="inverse")

    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(df, names='sentiment_label', title='Sentiment Distribution')
        st.plotly_chart(fig1, use_container_width=True)
        
        fig3 = px.histogram(df, x='rating', nbins=5, 
                           title='Rating Distribution',
                           color='sentiment_label')
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        fig2 = px.histogram(df, x='year_month', 
                           color='sentiment_label',
                           title='Reviews Over Time by Sentiment')
        st.plotly_chart(fig2, use_container_width=True)
        
        fig4 = px.scatter(df, x='sentiment', y='rating',
                         color='is_fake', title='Sentiment vs Rating')
        st.plotly_chart(fig4, use_container_width=True)

    # Sample reviews
    st.subheader("Review Samples")
    st.dataframe(df[['date', 'rating', 'sentiment_label', 'is_fake', 'text']]
                .sample(5)
                .sort_values('date', ascending=False)
                .style.set_properties(**{'text-align': 'left'}))

    # Download
    st.download_button(
        "Download Full Analysis",
        df.to_csv(index=False),
        "review_analysis.csv"
    )
else:
    st.error("Failed to load dataset. Please check the file path.")