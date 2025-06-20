import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
import uuid
import re
from difflib import SequenceMatcher

def parse_json_field(field):
    try:
        return json.loads(field) if isinstance(field, str) else {}
    except:
        return {}

def normalize_text(text):
    if pd.isna(text) or text == "":
        return ""
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def exact_match_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    return 1.0 if normalize_text(text1) == normalize_text(text2) else 0.0

def fuzzy_string_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio()

def semantic_similarity(text1, text2, model):
    if not text1 or not text2:
        return 0.0
    try:
        embeddings = model.encode([str(text1), str(text2)])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    except:
        return 0.0

def calculate_attribute_similarity(attr1, attr2, method="weighted"):
    try:
        if isinstance(attr1, str):
            attr1 = parse_json_field(attr1)
        elif not isinstance(attr1, dict):
            attr1 = {}
        
        if isinstance(attr2, str):
            attr2 = parse_json_field(attr2)
        elif not isinstance(attr2, dict):
            attr2 = {}
        
        if not attr1 or not attr2:
            return 0.0

        all_keys = set(attr1.keys()) | set(attr2.keys())
        if not all_keys:
            return 0.0

        if method == "exact":
            return 1.0 if attr1 == attr2 else 0.0
        elif method == "jaccard":
            set1 = set(f"{k}:{v}" for k, v in attr1.items())
            set2 = set(f"{k}:{v}" for k, v in attr2.items())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
        else:
            similarities = []
            weights = {
                'model': 3.0, 'model_number': 3.0, 'sku': 3.0,
                'brand': 2.5, 'manufacturer': 2.5,
                'category': 2.0, 'type': 2.0,
                'color': 1.5, 'colour': 1.5, 'size': 1.5,
                'material': 1.0, 'weight': 1.0, 'dimensions': 1.0,
                'description': 0.5, 'notes': 0.5
            }
            for key in all_keys:
                val1 = attr1.get(key, "")
                val2 = attr2.get(key, "")
                if val1 or val2:
                    sim = fuzzy_string_similarity(str(val1), str(val2)) if val1 and val2 else 0.0
                    weight = weights.get(key.lower(), 1.0)
                    similarities.append(sim * weight)
            return sum(similarities) / sum(weights.get(k.lower(), 1.0) for k in all_keys)
    except:
        return 0.0

def calculate_weighted_similarity(row1, row2, model, weights=None):
    try:
        if weights is None:
            weights = {'title': 0.45, 'attributes': 0.35, 'brand': 0.15, 'category': 0.05}
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        sims = {}

        if weights.get('title', 0) > 0:
            t1, t2 = row1.get('title', ''), row2.get('title', '')
            sims['title'] = 0.7 * semantic_similarity(t1, t2, model) + 0.3 * fuzzy_string_similarity(t1, t2) if t1 and t2 else 0.0

        if weights.get('attributes', 0) > 0:
            sims['attributes'] = calculate_attribute_similarity(row1.get('attributes', {}), row2.get('attributes', {}))

        if weights.get('brand', 0) > 0:
            b1, b2 = row1.get('brand', ''), row2.get('brand', '')
            exact = exact_match_similarity(b1, b2)
            fuzzy = fuzzy_string_similarity(b1, b2)
            sims['brand'] = 0.8 * exact + 0.2 * fuzzy if b1 and b2 else 0.0

        if weights.get('category', 0) > 0:
            c1, c2 = row1.get('category', ''), row2.get('category', '')
            sims['category'] = exact_match_similarity(c1, c2) if c1 and c2 else 0.0

        final_score = sum(weights[k] * sims.get(k, 0) for k in weights)
        return final_score, sims
    except:
        return 0.0, {'title': 0, 'attributes': 0, 'brand': 0, 'category': 0}

def deduplicate_products_weighted(df, threshold=0.85, model_name="all-MiniLM-L6-v2", auto_pick="Lowest Price", feature_weights=None):
    if feature_weights is None:
        feature_weights = {'title': 0.45, 'attributes': 0.35, 'brand': 0.15, 'category': 0.05}

    df = df.copy()
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        raise ValueError(f"Failed to load model '{model_name}': {e}")

    has_original_product_id = "product_id" in df.columns
    if has_original_product_id:
        df["original_product_id"] = df["product_id"]
    df["internal_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    pairs = []
    ids = df["internal_id"].tolist()

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            try:
                sim, breakdown = calculate_weighted_similarity(df.iloc[i].to_dict(), df.iloc[j].to_dict(), model, feature_weights)
                if sim >= threshold:
                    pairs.append({
                        "internal_id_1": ids[i],
                        "internal_id_2": ids[j],
                        "weighted_similarity": sim,
                        "title_similarity": breakdown.get('title', 0) or 0,
                        "attributes_similarity": breakdown.get('attributes', 0) or 0,
                        "brand_similarity": breakdown.get('brand', 0) or 0,
                        "category_similarity": breakdown.get('category', 0) or 0
                    })
            except:
                continue

    G = nx.Graph()
    for p in pairs:
        G.add_edge(p["internal_id_1"], p["internal_id_2"])

    to_keep = set()
    for component in nx.connected_components(G):
        group = list(component)
        group_df = df[df["internal_id"].isin(group)]
        if auto_pick == "Lowest Price" and "price" in df.columns:
            if pd.api.types.is_numeric_dtype(group_df["price"]) or all(pd.to_numeric(group_df["price"], errors='coerce').notna()):
                if not pd.api.types.is_numeric_dtype(group_df["price"]):
                    group_df["price"] = pd.to_numeric(group_df["price"], errors='coerce')
                best_product = group_df.loc[group_df["price"].idxmin()]
                to_keep.add(best_product["internal_id"])
            else:
                to_keep.add(group[0])
        else:
            to_keep.add(group[0])

    all_ids = set(df["internal_id"])
    to_keep.update(all_ids - set(G.nodes))
    df_result = df[df["internal_id"].isin(to_keep)].copy()

    if has_original_product_id:
        df_result["product_id"] = df_result["original_product_id"]
        df_result.drop(columns=["original_product_id", "internal_id"], errors="ignore", inplace=True)
    else:
        df_result["product_id"] = df_result["internal_id"]
        df_result.drop(columns=["internal_id"], errors="ignore", inplace=True)

    df_result = df_result.reset_index(drop=True)

    if pairs and has_original_product_id:
        id_mapping = dict(zip(df["internal_id"], df["original_product_id"]))
        for p in pairs:
            p["product_id_1"] = id_mapping.get(p["internal_id_1"])
            p["product_id_2"] = id_mapping.get(p["internal_id_2"])
            p.pop("internal_id_1")
            p.pop("internal_id_2")
    elif pairs:
        for p in pairs:
            p["product_id_1"] = p.pop("internal_id_1")
            p["product_id_2"] = p.pop("internal_id_2")

    pairs_df = pd.DataFrame(pairs if pairs else [])
    return df_result, pairs_df, {
        'method': 'weighted_similarity',
        'feature_weights': feature_weights,
        'threshold_used': threshold,
        'total_pairs_found': len(pairs),
        'products_removed': len(df) - len(df_result),
        'removal_percentage': (len(df) - len(df_result)) / len(df) * 100
    }