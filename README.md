# Link-prediction-in-graphs

# Description:
Link prediction aims to predict missing or future edges between nodes in a graph. It's useful in recommendation systems, social networks, and biological networks — for instance, suggesting new friends or predicting protein interactions. In this project, we’ll build a model to determine whether an edge exists between two nodes using graph embeddings.

# 🧪 Python Implementation (Link Prediction using GCN on Cora Dataset)
We’ll use PyTorch Geometric again, this time for link prediction by training a GCN to learn node embeddings, then compute similarity scores for predicting links.

# ✅ Prerequisites:
pip install torch-geometric

# ✅ What It Does:
* Splits edges into train/test sets for supervised learning.

* Learns node embeddings using a GCN encoder.

* Uses dot product as a similarity score for predicting if an edge exists between two nodes.

* Evaluates the model using ROC-AUC, a common metric for binary classification.
