# 🧠 Frequency-Resolved Brain Network GNN Mini-Challenge

This mini-challenge focuses on **graph-level classification of human brain networks** derived from resting-state fMRI, using **frequency-resolved multilayer connectivity** and **graph neural networks (GNNs)**.

Participants are asked to predict whether a subject belongs to a **high-performance** or **low-performance** group with respect to **Fluid Intelligence (PMAT24_A_CR)**, based solely on graph-structured brain network data.

The dataset and feature construction are based on the methodology introduced in **Neşe et al. (2024)**.

---

## 🧪 Scientific Background

Resting-state BOLD signals contain meaningful information beyond the conventional 0.01–0.1 Hz range.  
Following **Neşe et al. (2024)**, we analyze intrinsic brain connectivity across a broader frequency spectrum using **phase-based connectivity** and **multilayer network modeling**.

---

## 📊 Frequency Bands

The BOLD signal (0.01–0.23 Hz) is divided into **7 equal-width frequency bands**:

| Band | Frequency Range (Hz) |
|-----|----------------------|
| fb1 | 0.011 – 0.038 |
| fb2 | 0.043 – 0.071 |
| fb3 | 0.076 – 0.103 |
| fb4 | 0.109 – 0.136 |
| fb5 | 0.141 – 0.168 |
| fb6 | 0.174 – 0.201 |
| fb7 | 0.206 – 0.233 |

---

## 🧩 Dataset Construction

- **Subjects:** 96 healthy participants (Human Connectome Project)
- **Parcellation:** 400 cortical parcels
- **Connectivity:** Phase-consistency–based functional connectivity
- **Network type:** Multilayer (one layer per frequency band)
- **Group labels:** High vs Low Fluid Intelligence  
  *(median split of PMAT24_A_CR)*

---

<img width="1297" height="529" alt="Data" src="https://github.com/user-attachments/assets/24cacc20-9b57-4b3e-96d3-194a9901d6d8" />

**Figure:**  
(a) Graph structure with node-level features,  
(b) adjacency matrix \(A\),  
(c) node feature matrix \(X \in \mathbb{R}^{D \times N}\),  
(d) edge feature matrix \(E \in \mathbb{R}^{D_E \times E}\).

> The baseline model uses only node-level features (X).  
> Graph structure (A) and edge attributes (E) are provided for advanced models.


## 🧠 Network Metrics (Node Features)

Computed using a **generalized modularity algorithm** for multilayer networks:

- Normalized Participation Coefficient (**PCnorm**) × 7  
- Within-module degree **z-score** × 7  
- **Flexibility** × 1  
- Multilayer Betweenness Centrality (**MBC**) × 1  

➡️ **Total node features per parcel: 16**

---

## 🧠 Canonical Graph Representation

Each subject is represented as **one graph**:

- **Nodes:** 400 brain parcels  
- **Edges:**  
  - Common, group-level topology  
  - Thresholded functional connectivity (**top 30%**)  
- **Node features:** 16 features per node  
- **Edge features:** 7 frequency-band–specific weights  
- **Graph label:** Binary (High vs Low PMAT performance)

👉 **This is a graph-level binary classification problem.**

---

## 🎯 Task Description

**Goal:**  
Predict whether a subject belongs to the **high-performance** or **low-performance** group based on their brain network.

Participants may use **any graph neural network architecture**  
(GCN, GAT, Graph Transformer, MPNN, etc.).

---

## 🏁 Challenge Levels

### 🔹 Level A — Node-Feature-Only Classification

- **Input:** 16 node features per node
- **Task:** Classification using **node features only**

**Data location:**
```text
data/level_a/
├── train.csv
├── test.csv
```
---

### 🔹 Level B — Full Graph Learning (Node + Edge Features)

- **Edges:** Fixed topology with **7-band edge weights**
- **Nodes:** 16 node features per node
- **Task:** Graph-level classification using **node + edge features**

---

**Data location:**
```text
data/level_b/
├── edge_index.mat          # Shared graph topology
├── node_labels.mat         # Shared region names
├── train/
│   ├── node_features.mat   # (400 × 16 × nTrain)
│   ├── edge_attr.mat       # (E × 7 × nTrain)
│   ├── y_train.mat         # Training labels
└── test/
    ├── node_features.mat   # (400 × 16 × nTest)
    ├── edge_attr.mat       # (E × 7 × nTest)
```
---

### 📂 Level B File Descriptions

- **`edge_index.mat`** — Edge list defining the common graph topology
- **`node_labels.mat`** — Parcel / network labels (optional, for interpretation)
- **`edge_attr.mat`** — Frequency-resolved edge weights (7 per edge)
- **`node_features.mat`** — Node features
- **`y_train.mat`** — Training labels (0 = low, 1 = high)

### 🧮 Evaluation
- **Metric:** Macro-averaged F1 score
- **Task type:** Binary classification
- **Evaluation level:** Graph-level
- **Test labels:** Hidden (organizer-only)

