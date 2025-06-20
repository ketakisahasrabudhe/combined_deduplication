

import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
import uuid

def parse_json_field(field):
    try:
        return json.loads(field) if isinstance(field, str) else {}
    except:
        return {}

def get_text_for_embedding(row):
    title = str(row.get("title", "")) if "title" in row else ""
    brand = str(row.get("brand", "")) if "brand" in row else ""
    attributes = {}
    if "attributes" in row:
        if isinstance(row["attributes"], dict):
            attributes = row["attributes"]
        else:
            attributes = parse_json_field(row["attributes"])
    attr_string = " ".join([f"{k}:{v}" for k, v in attributes.items()])
    return f"{title} {brand} {attr_string}".strip()

def deduplicate_products(df, threshold=0.85, model_name="all-MiniLM-L6-v2", auto_pick="Lowest Price"):
    df = df.copy()
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise ValueError(f"Failed to load model '{model_name}': {e}")

    has_original_product_id = "product_id" in df.columns
    if has_original_product_id:
        df["original_product_id"] = df["product_id"]
    df["internal_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df["text_for_embedding"] = df.apply(get_text_for_embedding, axis=1)

    texts = df["text_for_embedding"].tolist()
    internal_ids = df["internal_id"].tolist()

    try:
        embeddings = model.encode(texts, convert_to_tensor=True)
        embeddings = [e.cpu().numpy() for e in embeddings]
    except:
        embeddings = model.encode(texts)

    pairs = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            try:
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if sim >= threshold:
                    pairs.append({
                        "internal_id_1": internal_ids[i],
                        "internal_id_2": internal_ids[j],
                        "similarity": sim
                    })
            except:
                continue

    G = nx.Graph()
    for row in pairs:
        G.add_edge(row["internal_id_1"], row["internal_id_2"])
    df_lookup = df.set_index("internal_id")
    to_keep = set()

    for component in nx.connected_components(G):
        group_ids = list(component)
        group_df = df[df["internal_id"].isin(group_ids)]
        if auto_pick == "Lowest Price" and "price" in df.columns:
            if pd.api.types.is_numeric_dtype(group_df["price"]) or all(pd.to_numeric(group_df["price"], errors='coerce').notna()):
                if not pd.api.types.is_numeric_dtype(group_df["price"]):
                    group_df["price"] = pd.to_numeric(group_df["price"], errors='coerce')
                best_product = group_df.loc[group_df["price"].idxmin()]
                best_pid = best_product["internal_id"]
            else:
                best_pid = group_ids[0]
        else:
            best_pid = group_ids[0]
        to_keep.add(best_pid)

    all_ids = set(df["internal_id"])
    to_keep.update(all_ids - set(G.nodes))
    df_result = df[df["internal_id"].isin(to_keep)].copy()

    if has_original_product_id:
        df_result["product_id"] = df_result["original_product_id"]
        df_result.drop(columns=["original_product_id", "internal_id", "text_for_embedding"], errors="ignore", inplace=True)
    else:
        df_result["product_id"] = df_result["internal_id"]
        df_result.drop(columns=["internal_id", "text_for_embedding"], errors="ignore", inplace=True)

    df_result = df_result.reset_index(drop=True)

    if pairs and has_original_product_id:
        id_mapping = dict(zip(df["internal_id"], df["original_product_id"]))
        for pair in pairs:
            pair["product_id_1"] = id_mapping.get(pair["internal_id_1"])
            pair["product_id_2"] = id_mapping.get(pair["internal_id_2"])
            pair.pop("internal_id_1")
            pair.pop("internal_id_2")
    elif pairs:
        for pair in pairs:
            pair["product_id_1"] = pair.pop("internal_id_1")
            pair["product_id_2"] = pair.pop("internal_id_2")

    return df_result, pd.DataFrame(pairs if pairs else [])
