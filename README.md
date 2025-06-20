# Product Deduplication 

Three different approaches:

1. **Basic Semantic Similarity (Naive-Graph based)**
2. **DBSCAN Clustering**
3. **Weighting features**


---

## 1. Basic Semantic Similarity 

### **Underlying Logic**:

* Joins the product fields (title, brand, and attributes) into a single text string.
* It uses a **SentenceTransformer model** (eg: MiniLM, MPNet) to convert each products combined text into a vector representation (embedding).
* captures the **semantic meaning** of the text.
* For every pair of products, the **cosine similarity** between their embeddings is computed. This gives a value from -1 (opposite) to 1 (identical).
* If the similarity exceeds a manually set **threshold**, the pair is considered a match.
* A graph is constructed with nodes as products and edges between those exceeding the threshold.
* Connected components in the graph are considered duplicate clusters, and one representative product is selected per cluster (can pick between lowest price or first seen in the group).

### **Disadvantages/Possible Improvements**:

* no feature consideration.
* Threshold tuning is manual.

* But handles "transitive matching" well.

---

## 2. DBSCAN Clustering (Unsupervised)

### **Underlying Logic**:

* Instead of comparing every pair individually, what the **DBSCAN** clustering algorithm does is:

  * Uses **cosine distance** (1 - cosine similarity) between embeddings.
  * Products are grouped into clusters where each point (product) is close to at least `min_samples` other points within a distance `eps`(neighbours).
  * Products that don’t belong to any dense region are treated as noise (i.e., not duplicates).

### **Advantages**:

* No fixed threshold needed.

### **Disadvantages**:

* tuning of `eps` and `min_samples`.
* Doesn’t always catch transitive similarities.

---

## 3. Weighting features

### **Underlying Logic**:

 not all features of a product contribute equally. For example, even one different attribute can imply a completely different product


**Similarity calculation per feature**:

   * **Title**: Uses both SentenceTransformer embeddings (semantic similarity, capturing meaning) and fuzzy string similarity (for typos or spelling variations).
   * **Attributes**: Parsed into dictionaries, and compared field-wise. Weighted by importance. Fuzzy string similarity handles inconsistent formatting.
   * **Brand**: Prioritizes exact match (brands should match exactly) with a fuzzy score to tolerate typos.
   * **Category**: Exact match only, since categories are expected to be standardized.

**Weighted aggregation**:
   Each feature similarity is multiplied by a weight (e.g., title: 0.45, attributes: 0.35, brand: 0.15, category: 0.05). These are normalized to sum to 1. Final score is:

   ```
   final_score = Σ (feature_similarity × feature_weight)
   ```

**Duplicate Identification**:

   * If `final_score >= threshold`, the products are considered duplicates.
   * A graph is built from matched pairs and same logic follows...


### **Disadvantages/Possible Improvements**:
* Needs weights to be manually defined.
* Doesn’t yet learn these weights from data.


* We can make the model learn the weights where they are determined automatically using supervised learning on labeled product pairs (this would be necessary for training). 

---



## You can choose from the following Transformer Models:

* `all-MiniLM-L6-v2` 
* `paraphrase-MiniLM-L6-v2`
* `all-mpnet-base-v2` 

For multilingual requirement ( product titles in different languages), check out `distiluse-base-multilingual-cased` or `paraphrase-multilingual-MiniLM` or mBERT, whether they work well or not.

---

## For Evaluation (of these methods)


* Prepare a labelled dataset of `{product_1, product_2, is_duplicate}`.
* Use it to compute metrics: precision(out of all the product pairs that model predicted as duplicates, how many are actually duplicates), recall(out of all the actual duplicate pairs, how many did your model correctly detect), F1-score, ROC-AUC, etc.
* can apply classifiers like:

  * **Logistic Regression** (interpretable baseline).
  * **Random Forests or Gradient Boosting (XGBoost)** 
* Perform k-fold validation to check generalization.


---

## Feedback loop and Retraining 

 Train a supervised classifier using the labeled (duplicate or not) examples taken from the feedback.

* Use classifiers like XGboost, Random Forest/Logistic regression (good for feature importance that is weighting for our task) to:

   * Predict match probabilities directly (and threshold them), OR
   * Learn feature weights automatically.
Save updated weights or model to file (e.g., JSON or joblib).


---

(TF-IDF wont work here , it wont be semantically/contexually aware)


