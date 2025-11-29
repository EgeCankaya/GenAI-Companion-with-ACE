"""Streamlit prototype for the IBM Gen AI Companion."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from genai_companion_with_ace.evaluation import ensure_default_dataset
from genai_companion_with_ace.rag import DocumentIngestionPipeline, IngestionConfig

st.set_page_config(page_title="IBM Gen AI Companion", layout="wide")
st.title("IBM Gen AI Companion (Preview UI)")
st.write(
    "This experimental interface complements the Rich CLI. "
    "Use it to inspect ingested documents and view evaluation coverage."
)

ingestion = DocumentIngestionPipeline(IngestionConfig(processed_dir=Path("data/processed")))

with st.sidebar:
    st.header("Controls")
    uploaded_files = st.file_uploader("Attach documents", accept_multiple_files=True)

    if uploaded_files:
        st.sidebar.success(f"{len(uploaded_files)} file(s) queued for inspection.")

if uploaded_files:
    st.subheader("Attachment Preview")
    for upload in uploaded_files:
        content = upload.read().decode("utf-8", errors="ignore")
        docs = ingestion.ingest_raw_content(content=content, source_name=upload.name, persist=False)
        excerpt = "\n\n".join(doc.page_content for doc in docs[:2])
        st.markdown(f"#### {upload.name}")
        st.text_area("Preview", value=excerpt, height=200)

st.subheader("Evaluation Dataset Coverage")
dataset_path = Path("data/eval/eval_questions_100.json")
dataset = ensure_default_dataset(dataset_path)
summary = dataset.summary()
st.bar_chart(summary)
