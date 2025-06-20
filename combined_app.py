import streamlit as st
import pandas as pd
import io
from deduplication_naive import deduplicate_products as basic_dedup
from deduplication_dbscan import deduplicate_with_dbscan
from deduplication_weighted_features import deduplicate_products_weighted

st.set_page_config(page_title="Combined methods Product Deduplication App", layout="wide")
st.title("Product Deduplication")

uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    method = st.sidebar.selectbox("Select Deduplication Method", [
        "Basic Semantic Similarity (Naive-Graph based)",
        "DBSCAN Clustering",
        "Weighting features"
    ])

    model_options = {
        "MiniLM": "all-MiniLM-L6-v2",
        "MPNet": "all-mpnet-base-v2",
        "Paraphrase-MiniLM": "paraphrase-MiniLM-L6-v2"
    }
    model_choice = st.sidebar.selectbox("Sentence Transformer Model", list(model_options.keys()))
    model_name = model_options[model_choice]
    auto_pick = st.sidebar.selectbox("Representative Selection", ["Lowest Price", "First Seen"])

    if method == "Basic Semantic Similarity (Naive-Graph based)":
        threshold = st.sidebar.slider("Similarity Threshold", 0.5, 0.99, 0.85, 0.01)
        if st.button("Run Deduplication"):
            try:
                df_deduped, pairs = basic_dedup(df.copy(), threshold=threshold, model_name=model_name, auto_pick=auto_pick)
                st.success(f"Removed {len(df) - len(df_deduped)} duplicates.")
                st.subheader("Deduplicated Data")
                st.dataframe(df_deduped)
                buffer = io.BytesIO()
                df_deduped.to_excel(buffer, index=False, engine='openpyxl')
                buffer.seek(0)
                st.download_button(
                    "Download Result",
                    data=buffer,
                    file_name="deduplicated_basic.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif method == "DBSCAN Clustering":
        eps = st.sidebar.slider("Eps (distance threshold)", 0.1, 0.9, 0.3, 0.05)
        min_samples = st.sidebar.slider("Min Samples", 2, 10, 2)
        if st.button("Run Deduplication"):
            try:
                df_deduped, pairs_df, analysis_info = deduplicate_with_dbscan(
                    df.copy(), model_name=model_name, auto_pick=auto_pick, eps=eps, min_samples=min_samples
                )
                st.success(f"Removed {analysis_info['products_removed']} duplicates.")
                st.subheader("Deduplicated Data")
                st.dataframe(df_deduped)
                buffer = io.BytesIO()
                df_deduped.to_excel(buffer, index=False, engine='openpyxl')
                buffer.seek(0)
                st.download_button(
                    "Download Result",
                    data=buffer,
                    file_name="deduplicated_dbscan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif method == "Weighting features":
        threshold = st.sidebar.slider("Similarity Threshold", 0.5, 0.99, 0.80, 0.01)
        title_w = st.sidebar.slider("Title Weight", 0.0, 1.0, 0.45, 0.05)
        attr_w = st.sidebar.slider("Attributes Weight", 0.0, 1.0, 0.35, 0.05)
        brand_w = st.sidebar.slider("Brand Weight", 0.0, 1.0, 0.15, 0.05)
        cat_w = st.sidebar.slider("Category Weight", 0.0, 1.0, 0.05, 0.05)
        total = title_w + attr_w + brand_w + cat_w
        if total == 0:
            total = 1
        weights = {
            'title': title_w / total,
            'attributes': attr_w / total,
            'brand': brand_w / total,
            'category': cat_w / total
        }

        if st.button("Run Deduplication"):
            try:
                df_deduped, pairs_df, analysis_info = deduplicate_products_weighted(
                    df.copy(), threshold=threshold, model_name=model_name, auto_pick=auto_pick, feature_weights=weights
                )
                st.success(f"Removed {analysis_info['products_removed']} duplicates.")
                st.subheader("Deduplicated Data")
                st.dataframe(df_deduped)
                buffer = io.BytesIO()
                df_deduped.to_excel(buffer, index=False, engine='openpyxl')
                buffer.seek(0)
                st.download_button(
                    "Download Result",
                    data=buffer,
                    file_name="deduplicated_weighted.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")