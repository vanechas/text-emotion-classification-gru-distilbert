# 💬 Multiclass Text Emotion Classification: Bidirectional GRU vs. DistilBERT

An end-to-end Natural Language Processing (NLP) benchmark comparing deep sequential recurrent architectures (**Bidirectional GRU**) against a transformer-based transfer learning model (**DistilBERT**) for 6-class fine-grained emotion detection on over 250,000 text samples[cite: 10].

---

## 📌 Project Overview

* **Domain:** Natural Language Processing (NLP), Sentiment & Emotion Analysis, Deep Learning[cite: 10]
* **Task:** Multiclass text classification into 6 fine-grained emotional states: `anger`, `fear`, `joy`, `love`, `sadness`, and `surprise`[cite: 10]
* **Dataset:** `text_emotions.tsv` containing 250,084 labeled text records[cite: 10]
* **Key Challenge:** Severe class imbalance across emotion labels (e.g., high representation of `joy` and `sadness` versus low representation of `love` and `surprise`)[cite: 10]
* **Core Comparison:** Assessing whether a feature-engineered, class-weighted sequential neural network (Bi-GRU) can outperform a partially frozen pretrained transformer (`distilbert-base-uncased`) on domain-specific imbalanced emotional text[cite: 10]

---

## 🔍 Data Preprocessing & Pipeline

1. **Text Cleansing:** Standardized text by converting to lowercase, stripping URLs with regular expressions, filtering non-alphabetical characters, and removing English stopwords using NLTK[cite: 10].
2. **Dataset Partitioning:** Stratified splitting to preserve label distribution across subsets[cite: 10]:
   * **Train:** 160,053 samples (64%)[cite: 10]
   * **Validation:** 40,014 samples (16%)[cite: 10]
   * **Test:** 50,017 samples (20%)[cite: 10]
3. **Class Weighting:** Balanced class weights were computed via `sklearn.utils.class_weight.compute_class_weight` and integrated into training loss functions to penalize misclassifications on minority categories[cite: 10].
4. **Tokenization & Input Formatting:**
   * **For GRU:** Keras Tokenizer sequence conversion with post-padding to a fixed sequence length of 100 tokens, converted into one-hot encoded categorical vectors[cite: 10].
   * **For DistilBERT:** `DistilBertTokenizerFast` generating `input_ids` and `attention_mask` tensors with truncation and padding to a maximum sequence length of 128[cite: 10].
5. **High-Performance Data Loaders:** Streaming pipelines structured with `tf.data.Dataset` utilizing `.shuffle()`, `.batch()`, and `.prefetch(tf.data.AUTOTUNE)`[cite: 10].

---

## 🤖 Model Architectures

### 1. Stacked Bidirectional GRU (Class-Weighted)
* **Embedding Layer:** Vocabulary size mapped to 128-dimensional dense representations[cite: 10]
* **Recurrent Block 1:** Bidirectional GRU with 256 units (`return_sequences=True`) capturing bidirectional sequential context[cite: 10]
* **Regularization:** Spatial Dropout layer (rate = 0.3)[cite: 10]
* **Recurrent Block 2:** Bidirectional GRU with 128 units returning the final latent vector[cite: 10]
* **Output Head:** Dense layer with Softmax activation over 6 classes[cite: 10]
* **Optimization:** Adam optimizer with categorical cross-entropy loss[cite: 10]

### 2. DistilBERT Transformer (Pretrained & Fine-Tuned)
* **Backbone:** `distilbert-base-uncased` via Hugging Face Transformers (`TFDistilBertForSequenceClassification`)[cite: 10]
* **Transfer Learning Strategy:** Frozen lower transformer layers to preserve foundational language representations, unfreezing the final transformer block layer and classification head for domain adaptation[cite: 10]
* **Optimization:** Adam optimizer with sparse categorical cross-entropy loss[cite: 10]

---

## 📊 Experimental Results & Comparison

Both models were evaluated on the identical unseen test set of 50,017 samples[cite: 10]:

| Metric | Bidirectional GRU (Class Weighted) | DistilBERT (Partially Frozen) |
| :--- | :---: | :---: |
| **Overall Accuracy** | **94.00%** | **59.07%**[cite: 10] |
| **Macro Average Precision** | **0.88** | **0.59**[cite: 10] |
| **Macro Average Recall** | **0.95** | **0.41**[cite: 10] |
| **Macro Average F1-Score** | **0.91** | **0.42**[cite: 10] |
| **Weighted Average F1-Score** | **0.94** | **0.55**[cite: 10] |

### Detailed Performance by Emotion Class (Bidirectional GRU)

| Emotion Class | Support | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **sadness** | 14,543 | 0.99 | 0.95 | **0.97**[cite: 10] |
| **joy** | 16,928 | 1.00 | 0.90 | **0.95**[cite: 10] |
| **anger** | 6,878 | 0.93 | 0.94 | **0.94**[cite: 10] |
| **fear** | 5,725 | 0.89 | 0.91 | **0.90**[cite: 10] |
| **love** | 4,146 | 0.76 | 1.00 | **0.86**[cite: 10] |
| **surprise** | 1,797 | 0.73 | 1.00 | **0.84**[cite: 10] |

The class-weighted Bidirectional GRU achieved strong recognition across all 6 classes, particularly maintaining 100% recall on the minority classes `love` and `surprise`[cite: 10]. In contrast, the partially frozen DistilBERT model suffered from bias toward dominant classes (`joy` and `sadness`), yielding low recall on minority emotion labels[cite: 10].

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.x[cite: 10]
* **Deep Learning Framework:** TensorFlow 2.x, Keras[cite: 10]
* **Pretrained Transformers:** Hugging Face `transformers` (`DistilBertTokenizerFast`, `TFDistilBertForSequenceClassification`)[cite: 10]
* **NLP & Text Processing:** NLTK, Regular Expressions (`re`)[cite: 10]
* **Machine Learning & Metrics:** scikit-learn (`classification_report`, `LabelEncoder`, `train_test_split`, `compute_class_weight`)[cite: 10]
* **Data Handling & Visualization:** Pandas, NumPy, Matplotlib[cite: 10]

---

## 📁 Repository Structure

```text
├── data/
│   └── text_emotions.tsv      # Dataset (250k labeled records)
├── notebooks/
│   └── emotion_classification.ipynb  # End-to-end preprocessing, modeling, & benchmark
├── README.md                  # Project documentation
```
