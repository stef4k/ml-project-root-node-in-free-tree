# ML Root node prediction in free tree

## Description

This project was developed for the Machine Learning course in the Master’s in BDMA at UPC. The goal is to predict the root node in syntactic dependency trees derived from sentences in a 21-language parallel corpus. Each tree is represented as a set of undirected edges, and the root prediction task is framed as a binary classification problem at the node level. Centrality measures such as degree, closeness, betweenness, PageRank, and more are extracted and used as node features. The project involves preprocessing the data, constructing an expanded node-level dataset, engineering features, addressing class imbalance and dependency issues, and evaluating several classification models. The repository includes code for preprocessing, modeling, and evaluation.

Check the important pdf files:
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
