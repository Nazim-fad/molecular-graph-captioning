# Molecular Graph Captioning – ALTEGRAD 2025–26 Data Challenge

This repository contains my solutions for the **ALTEGRAD 2025–26 Kaggle Data Challenge**. The goal is to develop models that translate a **molecular graph** (atoms + bonds) into a **natural-language description** of the molecule.

## 1. Problem Overview

Each molecule is represented as a graph structure:
* **Nodes (Atoms):** 9 categorical features (atomic number, chirality, degree, formal charge, attached hydrogens, radical electrons, hybridization, aromatic flag, ring flag).
* **Edges (Bonds):** 3 categorical features (bond type, stereo, conjugation).

The objective is to learn a model $f: G \rightarrow S$ that generates a caption describing the molecule's structural and functional characteristics. Evaluation is performed on a hidden test set using text-generation metrics: **BLEU-4** and **BERTScore** (using RoBERTa-base).

### Data Splits
* **Train:** ~31k graphs (with descriptions)
* **Validation:** 1k graphs (with descriptions)
* **Test:** 1k graphs (without descriptions)

The graphs are stored as `PyTorch Geometric Data` objects in pickle files:
* `train_graphs.pkl`
* `validation_graphs.pkl`
* `test_graphs.pkl`

---

## 2. Baseline Retrieval Model

`data_baseline`

### 2.1. High-Level Idea

The baseline approach frames the task as a **joint-embedding retrieval system**:

1. **Graph Encoder:** A simple GCN maps the molecular graph to a fixed-size vector .
2. **Text Encoder:** A pre-trained BERT encodes the caption into a vector.
3. **Training:** The GCN is trained using **MSE Loss** to minimize the distance between matching graph-caption pairs.
4. **Inference:** For a given test graph, the model retrieves the **nearest neighbor** caption from the training set based on cosine similarity.


### 2.2. How to Run

From the `data_baseline` folder, run the following:


#### Step 1: Install Dependencies

```bash
pip install -r requirements.txt

```

#### Step 2: Prepare Data

Ensure your graph pickle files are placed inside the `data/` directory:

* `data/train_graphs.pkl`
* `data/validation_graphs.pkl`
* `data/test_graphs.pkl`

#### Step 3: Generate Text Embeddings

Pre-compute the BERT embeddings for the training captions:

```bash
python generate_description_embeddings.py

```

*Output:* `data/train_embeddings.csv`, `data/validation_embeddings.csv`

#### Step 4: Train the Retriever

Train the GCN encoder (simple topology-only GCN with shared node embeddings):

```bash
python train_gcn.py

```

*Output:* `model_checkpoint.pt`

#### Step 5: Inference (Retrieval)

Compute embeddings for test graphs and retrieve the closest training captions:

```bash
python retrieval_answer.py

```

*Output:* `test_retrieved_descriptions.csv` (Submission file with `ID` and `description` columns).

## 3. Solution 1: Improved Retrieval (GINE + Contrastive Learning)


### 3.1. High-Level Idea
We improve upon the baseline by introducing a **GINE-based encoder** and a **Contrastive Learning** objective.
* **Encoder:** Instead of a simple GCN, we use **GINEConv** to explicitly model edge features (bond type, stereo, conjugation) and atom features.
* **Loss Function:** We replace MSE with a **symmetric InfoNCE loss** (CLIP-style), which forces the model to maximize the similarity of positive pairs while pushing away negative examples in the batch.
* **Matching Head:** We train an auxiliary MLP to re-rank the top retrieved candidates for better precision.

### 3.2. How to Run

1.  **Requirement:**
Make sure all dataset and checkpoint paths are correctly configured in the corresponding YAML files located in the `config/` directory.


1.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Analysis (Optional):**
    Check dataset statistics and graph properties.
    ```bash
    python scripts/analysis/check_description_stats.py
    python scripts/analysis/inspect_graph_data.py
    ```

3.  **Precompute Text Embeddings:**
    Generate BERT embeddings for the training captions.
    ```bash
    python scripts/preprocess/generate_description_embeddings.py
    ```

4.  **Train the Model:**
    Train the GINE encoder and matching head.
    ```bash
    python scripts/train/train_gine.py
    ```

5.  **Inference:**
    Retrieve the best captions for the test set.
    ```bash
    python scripts/inference/retrieval_answer.py
    ```

---

## 4. Solution 2: Retrieval-Augmented Generation (RAG)


### 4.1. High-Level Idea
This approach combines retrieval with the generative power of Large Language Models (LLMs).
* **Retrieval:** We use the trained GINE model (from Solution 1) to retrieve similar captions for the test molecule.
* **Prompting:** We construct a prompt containing a **rule-based graph summary** (atom counts, charge, aromaticity, etc.) and the **retrieved captions** as few-shot examples.
* **Generation:** We use **Mistral-7B-Instruct-v0.2** (quantized to 4-bit) to generate a final description that integrates the factual graph details with the style of the retrieved text.

### 4.2. How to Run

0.  **Requirement:**
    To build the prompt dataset, you must first train the GINE encoder (Solution 1), as its learned weights are required for retrieval.

    Make sure all dataset and checkpoint paths are correctly configured in the corresponding YAML files located in the `config/` directory.



1.  **Build the Dataset:**
    Construct the prompts by combining graph summaries and retrieved context.
    ```bash
    python scripts/preprocess/build_rag_dataset.py
    ```

2.  **Run Inference:**
    Generate descriptions using the LLM, you can change the model in the `llm_rag.yaml` file.
    ```bash
    python scripts/inference/generate_llm_rag.py
    ```

---

## 5. Solution 3: End-to-End Generation (MolT5)


### 5.1. High-Level Idea
This is a pure generative approach that translates graphs directly to text without relying on retrieval during inference.
* **Encoder:** A **GATv2** (Graph Attention Network) encoder with a **Virtual Node** to capture global graph context and long-range dependencies.
* **Decoder:** We use the **MolT5-base** decoder, initialized from weights pre-trained on chemical SMILES and text.
* **Training:** The model is trained end-to-end with a composite loss: **Cross-Entropy** (generation) + **Contrastive Loss** (latent alignment).

### 5.2. How to Run

0.  **Requirement:**
    Make sure all dataset and checkpoint paths are correctly configured in the corresponding YAML files located in the `config/` directory.


1.  **Train the Model:**
    Train the MolT5 architecture from scratch or fine-tune.
    ```bash
    python scripts/train/train_molt5.py
    ```

2.  **Inference:**
    Generate descriptions for the test set using Beam Search.
    ```bash
    python scripts/inference/generate_molt5.py
    ```

---

## 6. Results

| Model | Approach | Kaggle Score |
| :--- | :--- | :--- |
| **Baseline** | Simple GCN + MSE Retrieval | 0.48 |
| **Solution 1** | GINE + Contrastive Retrieval | **0.61** |
| **Solution 2** | RAG (Mistral-7B) | 0.57 |
| **Solution 3** | MolT5 End-to-End | 0.58 |