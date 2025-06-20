

import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import numpy as np
import uuid
import warnings
warnings.filterwarnings('ignore')

def parse_json_field(field):
    try:
        return json.loads(field) if isinstance(field, str) else {}
    except:
        return {}

def get_text_for_embedding(row):
    title = str(row.get("title", ""))
    brand = str(row.get("brand", ""))
    attributes = {}
    if "attributes" in row:
        if isinstance(row["attributes"], dict):
            attributes = row["attributes"]
        else:
            attributes = parse_json_field(row["attributes"])
    attr_string = " ".join([f"{k}:{v}" for k, v in attributes.items()])
    return f"{title} {brand} {attr_string}".strip()

def deduplicate_with_dbscan(df, model_name="all-MiniLM-L6-v2", auto_pick="Lowest Price", eps=0.3, min_samples=2):
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

    try:
        embeddings = model.encode(texts, convert_to_tensor=True)
        embeddings = np.array([e.cpu().numpy() for e in embeddings])
    except:
        embeddings = np.array(model.encode(texts))

    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    cluster_labels = dbscan.fit_predict(embeddings)
    df["cluster"] = cluster_labels

    cluster_counts = pd.Series(cluster_labels).value_counts()
    duplicate_clusters = cluster_counts[cluster_counts > 1].index.tolist()
    if -1 in duplicate_clusters:
        duplicate_clusters.remove(-1)

    pairs = []
    for cluster_id in duplicate_clusters:
        members = df[df["cluster"] == cluster_id]["internal_id"].tolist()
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                idx_i = df[df["internal_id"] == members[i]].index[0]
                idx_j = df[df["internal_id"] == members[j]].index[0]
                sim = cosine_similarity([embeddings[idx_i]], [embeddings[idx_j]])[0][0]
                pairs.append({
                    "internal_id_1": members[i],
                    "internal_id_2": members[j],
                    "similarity": sim,
                    "cluster": cluster_id
                })

    to_keep = set()
    for cluster_id in duplicate_clusters:
        group_df = df[df["cluster"] == cluster_id]
        if auto_pick == "Lowest Price" and "price" in df.columns:
            if pd.api.types.is_numeric_dtype(group_df["price"]) or all(pd.to_numeric(group_df["price"], errors='coerce').notna()):
                if not pd.api.types.is_numeric_dtype(group_df["price"]):
                    group_df["price"] = pd.to_numeric(group_df["price"], errors='coerce')
                best_product = group_df.loc[group_df["price"].idxmin()]
                best_pid = best_product["internal_id"]
            else:
                best_pid = group_df.iloc[0]["internal_id"]
        else:
            best_pid = group_df.iloc[0]["internal_id"]
        to_keep.add(best_pid)

    non_duplicate_ids = df[~df["cluster"].isin(duplicate_clusters)]["internal_id"].tolist()
    to_keep.update(non_duplicate_ids)
    df_result = df[df["internal_id"].isin(to_keep)].copy()

    if has_original_product_id:
        df_result["product_id"] = df_result["original_product_id"]
        df_result.drop(columns=["original_product_id", "internal_id", "text_for_embedding", "cluster"], errors="ignore", inplace=True)
    else:
        df_result["product_id"] = df_result["internal_id"]
        df_result.drop(columns=["internal_id", "text_for_embedding", "cluster"], errors="ignore", inplace=True)

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

    pairs_df = pd.DataFrame(pairs if pairs else [])
    return df_result, pairs_df, {
        'method': 'dbscan',
        'eps': eps,
        'min_samples': min_samples,
        'n_clusters': len(duplicate_clusters),
        'n_noise_points': sum(cluster_labels == -1),
        'total_pairs_found': len(pairs),
        'products_removed': len(df) - len(df_result),
        'removal_percentage': (len(df) - len(df_result)) / len(df) * 100,
        'cluster_sizes': cluster_counts.to_dict()
    }
