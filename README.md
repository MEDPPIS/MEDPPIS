# MED-PPIS: Multi-order Moments External Graph Attention Network

This repository contains the official source code and pre-trained models for the paper: "MED-PPIS: Multi-order Moments External Graph Attention Network with Dual-Axis Attention for Protein-Protein Interaction Site prediction".

## Abstract

Predicting protein-protein interaction sites (PPIS) occupies a pivotal position in biological sciences and bioinformatics, as this capability facilitates a deeper mechanistic understanding of complex cellular processes, molecular mechanisms, and genomic research paradigms; while graph neural networks (GNNs) have driven remarkable advancements in PPIS prediction, two critical challenges remain unresolved: the oversight of external graph information—most notably inter-graph correlations and the loss of critical feature distribution information arising from simplistic statistical operations (e.g., mean, max, sum) in neighbor feature aggregation, a practice prevalent in most GNN methodologies. To address these limitations, we present MED-PPIS, a novel framework integrating four key innovations: an mLSTM-based matrix memory mechanism to capture long-range dependencies in the sequential information embedded in input features; a graph external attention mechanism that establishes implicit connections across graphs via a set of learnable external units functioning as shared memory, where these external units comprising node-level and edge-level key-value pairs—capture and propagate inter-graph structural and semantic correlations through interactions with graph nodes; a novel higher-order moment graph neural network that comprehensively characterizes neighborhood feature distributions via multiple statistical moments from probability theory, thereby overcoming the limitations of single-statistic aggregation.

---

## Repository Structure

-   `/Dataset`: Contains the `.pkl` files for the training and testing datasets, such as `Train_335.pkl` and `Test_60.pkl`.
-   `/Feature`: Contains all the pre-calculated feature files (e.g., in subfolders like `pssm`, `hmm`, `dssp`, etc.) required by the model.
-   `/Log/model`: Contains all the pre-trained model weights (`.pkl` files) managed by Git LFS. This includes models for each fold and final epoch models.
-   `MEDPPIS_model.py`: The core implementation of the MED-PPIS model architecture.
-   `test.py`: The script to evaluate the performance of our pre-trained models on the test datasets.
-   `requirements.txt`: A list of all Python dependencies required to run the code.
-   `.gitattributes`: Configuration file for Git LFS to track `.pkl` files.
-   `.gitignore`: Specifies files to be ignored by Git.

---

## Requirements and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MEDPPIS/MEDPPIS.git
    cd MEDPPIS
    ```

2.  **Install Git LFS:**
    Make sure you have Git LFS installed to handle the large model files.
    ```bash
    git lfs install
    ```

3.  **Pull LFS files:**
    Download the large model files from the LFS storage.
    ```bash
    git lfs pull
    ```

4.  **Install Python dependencies:**
    We recommend using a virtual environment. All required packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage: Evaluating Pre-trained Models

To reproduce the results reported in our paper, you can simply run the `test.py` script. This will load our pre-trained models from the `/Log/model` directory and evaluate them on the `Test_60`, `Test_315-28`, and `UBtest_31-6` datasets.

```bash
python test.py