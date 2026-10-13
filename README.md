# ML Root node prediction in free tree

**Machine Learning course project** · Erasmus Mundus BDMA, Universitat Politècnica de Catalunya (UPC), 2025

Team: Stefanos Kypritidis, Kristóf Balázs · Evaluated through the course's Kaggle competition
 
Every sentence can be represented as a syntactic dependency tree, where nodes are words and edges are grammatical dependencies, and one word acts as the root. In this project the trees are given without that information: each sentence comes as a list of undirected edges (a *free tree*), and the task is to predict which word is the root. The data come from a parallel corpus of 21 languages. The training set has 500 sentences per language (10,500 trees) and the test set 495 per language (10,395 trees), scored on Kaggle as the fraction of sentences whose root is identified correctly. A random guess is right with probability 1/n, i.e. about 5% for a sentence of average length (18.8 words).
 
We framed the problem as node-level binary classification (root vs. non-root). For each tree we computed 13 centrality measures per node with NetworkX (degree, closeness, betweenness, PageRank, Laplacian, current-flow betweenness, and others) and normalised them within each sentence, so that nodes from short and long sentences are comparable. At prediction time, the node with the highest predicted root probability in each sentence is chosen. Because sentence length and tree structure vary strongly across languages, we trained a separate model for each of the 21 languages. All validation was grouped by sentence (GroupShuffleSplit for an 80/20 hold-out and StratifiedGroupKFold for 5-fold tuning), so neither the nodes of one tree nor the translations of one sentence end up on both sides of a split, and a custom scorer reproduced the Kaggle metric.
 
We compared a linear pipeline (StandardScaler + PCA + logistic regression), random forests and XGBoost. XGBoost handled the strong class imbalance through `scale_pos_weight`, used L1/L2 regularisation and early stopping against overfitting, and ran on GPU to make larger hyperparameter grids feasible. We also tested several additional feature sets: Node2Vec embeddings, subtree-balance metrics, normalised word position, centrality differences between a node and its neighbours, and 1-hop neighbour aggregates. None justified their cost; the best of them, neighbour aggregates, added only 1–2 points. As an exploratory direction, we treated root prediction as graph source localisation (finding the origin of a diffusion process) and tested GraphSL methods on a Polish subset, where LPSI gave the best validation AUC; scaling this up was beyond our compute.
 
**Main findings.** Per-language XGBoost was the best model, reaching 54.9% root accuracy on the sentence-grouped hold-out, compared with 29.7% for the linear pipeline. Feature importance showed Laplacian and current-flow betweenness centrality among the strongest predictors across many languages, but no single feature set worked for all of them, which supports training one model per language. The hold-out estimate was close to the score on Kaggle's public test subset. We also found that the provided edge lists kept their original direction, so picking the node with zero in-degree recovers the root with 100% accuracy. We verified this leak in a separate notebook but did not use it: every submitted model treats the trees as undirected.
 
## Results
 
| Model (one per language) | Validation | Root accuracy |
|---|---|---|
| StandardScaler + PCA + logistic regression | 20% sentence-grouped hold-out | 29.7% |
| Random forest | Out-of-bag estimate | 41.1% |
| **XGBoost (final model)** | 20% sentence-grouped hold-out | **54.9%** |
 
The random forest score comes from out-of-bag estimation rather than the shared hold-out set, so it is not directly comparable with the other two.
 
## Documents

- [Final Report](https://github.com/stef4k/ml-project-root-node-in-free-tree/blob/main/project-docs/final_report.pdf)
- [Appendix](https://github.com/stef4k/ml-project-root-node-in-free-tree/blob/main/project-docs/appendix.pdf), containing all the visualizations
- [Complete assignment description](https://github.com/stef4k/ml-project-root-node-in-free-tree/blob/main/project-docs/project_guidelines.pdf)

## Installation and Execution
Follow these steps to set up the environment and jupyter kernel:

1. Create a virtual environment:
   ```sh
   python -m venv myenv
   ```

2. Activate virtual environment:
   ```sh
   myenv\Scripts\activate
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

4. Add the virtual environment as a new Jupyter kernel
   ```sh
   python -m ipykernel install --user --name=myenv --display-name "Python ML-project(myenv)"
   ```
