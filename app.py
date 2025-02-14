import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import spacy
import PyPDF2
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
import io
import ollama  # Import Ollama for AI summarization
import numpy as np

# Download NLTK punkt tokenizer if not available
nltk.download('punkt')

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="📄 Document Analysis Agent with AI Chat", layout="wide")

st.title("📊 Document Analysis Agent + AI Chat 🤖")

# Description Box
st.markdown("""
### 📌 About This App
This AI-powered document analysis tool allows you to:
- 📄 **Upload documents** (PDF, TXT, CSV)  
- 🤖 **Get AI-powered summaries** using **local Ollama models**  
- 📊 **Visualize CSV data** with scatter plots, heatmaps, boxplots & line graphs  
- 💬 **Chat with AI** to ask questions about your document  
- 📥 **Download processed visualizations**  

**🚀 Powered by Streamlit, NLP, and Local AI Models**
""", unsafe_allow_html=True)

# Sidebar for Chat History
with st.sidebar:
    st.header("🗨️ Chat Assistant")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages in containers for structured view
    for msg in st.session_state.messages:
        with st.container():
            if msg["role"] == "user":
                st.markdown(f"👤 **You:** {msg['content']}")
            else:
                st.markdown(f"🤖 **AI:** {msg['content']}")

    # Input for user chat
    user_input = st.text_input("💬 Ask AI", key="user_query")

    if user_input:
        # Store user input
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Call local Ollama model
        response = ollama.chat(
            model="deepseek-r1:1.5b",  # Ensure this model is available in Ollama
            messages=st.session_state.messages,
        )

        # Get AI response
        ai_reply = response["message"]["content"]
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})

        # Refresh the sidebar chat display
        st.rerun()  # ✅ Corrected from st.experimental_rerun()


# File uploader
uploaded_file = st.file_uploader("Upload a file (PDF, TXT, or CSV)", type=["pdf", "txt", "csv"])

def read_pdf(file):
    """Extract text from PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_txt(file):
    """Read text from TXT file."""
    return file.getvalue().decode("utf-8")

def read_csv(file):
    """Read CSV file into a Pandas DataFrame."""
    return pd.read_csv(file, delimiter=";")

def summarize_with_ollama(text):
    """Summarizes text using a local Ollama model."""
    try:
        response = ollama.chat(
            model="deepseek-r1:1.5b",  # Change to your preferred model, e.g., 'llama3', 'mistral', etc.
            messages=[
                {"role": "system", "content": "Summarize the following text concisely:"},
                {"role": "user", "content": text}
            ]
        )
        return response["message"]["content"] if "message" in response else "Summarization failed."
    except Exception as e:
        return f"Error: {e}"

file_extension = None  # Ensure the variable exists before use
# Store extracted text
document_text = None


# Handle uploaded file
if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension == "pdf":
        text = read_pdf(uploaded_file)
    elif file_extension == "txt":
        text = read_txt(uploaded_file)
    elif file_extension == "csv":
        df = read_csv(uploaded_file)
        st.dataframe(df)  # Display CSV file
        text = None
    else:
        st.error("Unsupported file format")
        text = None

    if text:
        # Display text preview
        st.subheader("📄 Document Preview")
        st.text_area("Extracted Text:", text[:1000] + "..." if len(text) > 1000 else text, height=200)

        # AI Summarization using Ollama
        st.subheader("🧠 AI-Powered Summarization")
        if st.button("⚡ Generate AI Summary"):
            summary = summarize_with_ollama(text)
            st.write(summary)

        # Word Frequency Analysis
        st.subheader("📊 Word Frequency Analysis")
        words = [token.text.lower() for token in nlp(text) if token.is_alpha and not token.is_stop]
        word_counts = Counter(words)

        # Display word frequency table
        freq_df = pd.DataFrame(word_counts.items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)
        st.dataframe(freq_df.head(20))

        # Generate word cloud
        st.subheader("☁️ Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(words))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # Sentiment Analysis
        st.subheader("📈 Sentiment Analysis")
        sentiment = TextBlob(text).sentiment
        st.write(f"**Polarity:** {sentiment.polarity} (🔴 Negative ↔️ 🟢 Positive)")
        st.write(f"**Subjectivity:** {sentiment.subjectivity} (0: Objective, 1: Subjective)")

# CSV-Specific Visualization
def save_fig_as_bytes(fig):
    """Save Matplotlib figure to bytes for Streamlit download button."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

if file_extension == "csv":
    st.subheader("📊 CSV Data Visualization")

    # Choose numeric columns for graphing
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    if numeric_columns:
        x_axis = st.selectbox("Select X-axis", numeric_columns)
        y_axis = st.selectbox("Select Y-axis", numeric_columns)

        # Scatter Plot
        st.subheader("📌 Scatter Plot")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
        st.pyplot(fig)
        st.download_button("📥 Download Scatter Plot", save_fig_as_bytes(fig), "scatter_plot.png", "image/png")

        # Heatmap with Significant Signs
        if len(numeric_columns) > 1:  
            st.subheader("🔥 Correlation Heatmap (Significant Pairs Marked)")
            correlation_matrix = df[numeric_columns].corr()

            # Create mask for upper triangle
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

            fig, ax = plt.subplots(figsize=(8, 6))
            heatmap = sns.heatmap(
                correlation_matrix, 
                mask=mask, 
                annot=True, 
                fmt=".2f", 
                cmap="coolwarm", 
                linewidths=0.5
            )

            # Mark significant correlations with '*'
            for i in range(len(correlation_matrix.columns)):
                for j in range(i):
                    if abs(correlation_matrix.iloc[i, j]) > 0.7:
                        ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center", color="black", fontsize=14)

            st.pyplot(fig)
            st.download_button("📥 Download Heatmap", save_fig_as_bytes(fig), "heatmap.png", "image/png")

        else:
            st.warning("Need at least two numeric columns for a heatmap.")

        # Boxplot
        st.subheader("📦 Boxplot")
        selected_boxplot_col = st.selectbox("Select column for boxplot", numeric_columns)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df[selected_boxplot_col], ax=ax)
        st.pyplot(fig)
        st.download_button("📥 Download Boxplot", save_fig_as_bytes(fig), "boxplot.png", "image/png")

        # Line Graph
        st.subheader("📈 Line Graph")
        x_line = st.selectbox("Select X-axis for line graph", numeric_columns)
        y_line = st.selectbox("Select Y-axis for line graph", numeric_columns, index=1 if len(numeric_columns) > 1 else 0)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df, x=x_line, y=y_line, ax=ax, marker="o")
        st.pyplot(fig)
        st.download_button("📥 Download Line Graph", save_fig_as_bytes(fig), "line_graph.png", "image/png")

    else:
        st.warning("No numeric columns available for graphing.")

# AI Chat Section
st.subheader("🤖 AI Chat")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("Ask AI a question based on the document:")

if st.button("💬 Chat"):
    if user_query:
        with st.spinner("Generating response..."):
            response = ollama.chat(
                model="deepseek-r1:1.5b",
                messages=[
                    {"role": "system", "content": "You are an AI assistant analyzing documents."},
                    {"role": "user", "content": f"Context: {document_text}\nQuestion: {user_query}"}
                ]
            )
            ai_response = response["message"]["content"]
            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("AI", ai_response))
            st.write(f"**AI:** {ai_response}")


st.markdown("🚀 **Powered by Streamlit, Ollama, NLP & Data Science**")
