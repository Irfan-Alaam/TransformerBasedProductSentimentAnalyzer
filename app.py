import streamlit as st
from model_utils import predict_sentiment, load_transformer_model, model, vocab, label_map

st.set_page_config(page_title="Sentiment Analysis", page_icon="🧠", layout="centered")
st.title("🧠 Sentiment Analysis with Transformer Playground")

# -----------------
# Model Selection & Hyperparameters
# -----------------
st.sidebar.header("⚙️ Model Settings")
checkpoint_choice = st.sidebar.selectbox(
    "Select model checkpoint",
    options=["transformer_checkpoint.pkl"],  # add more checkpoints if available
    index=0
)

st.sidebar.subheader("Hyperparameters")
num_layers = st.sidebar.number_input("Number of Encoder Layers", min_value=1, max_value=12, value=model.params["num_layers"])
d_model = st.sidebar.number_input("Hidden Dimension (d_model)", min_value=32, max_value=1024, value=model.params["d_model"])
num_heads = st.sidebar.number_input("Number of Attention Heads", min_value=1, max_value=16, value=4)
seq_len = st.sidebar.number_input("Sequence Length", min_value=16, max_value=512, value=model.params["seq_len"])

if st.sidebar.button("🔄 Reload Model"):
    override_params = {
        "num_layers": num_layers,
        "d_model": d_model,
        "num_heads": num_heads,
        "seq_len": seq_len
    }
    model, vocab, label_map = load_transformer_model(checkpoint_choice, override_params)
    st.sidebar.success("✅ Model reloaded with new hyperparameters")

# -----------------
# Review Input
# -----------------
text_input = st.text_area(
    "Review Text",
    height=150,
    placeholder="Type a review here..."
)

if st.button("🔍 Analyze Sentiment"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text")
        st.stop()

    with st.spinner("🤖 Analyzing sentiment..."):
        result = predict_sentiment(text_input.strip(), model, vocab, label_map)

    st.subheader("📊 Result")
    st.write(f"**Analyzed Text (first 100 chars):** {text_input[:100]}{'...' if len(text_input) > 100 else ''}")

    sentiment = result['sentiment']
    confidence = result['confidence'] * 100

    if sentiment == "Positive":
        st.success(f"**Sentiment:** {sentiment} 👍")
    elif sentiment == "Negative":
        st.error(f"**Sentiment:** {sentiment} 👎")
    else:
        st.warning(f"**Sentiment:** {sentiment} 😐")

    st.metric(label="Confidence", value=f"{confidence:.2f}%")

    st.subheader("📈 Probabilities")
    for label, prob in result['probabilities'].items():
        st.write(f"{label}: {prob:.4f}")
