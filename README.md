# AI-Driven Coin Analysis

This repository contains the code for my Bachelor thesis:  
**"From Computer Vision to Network Analysis: Comparing AI Approaches for Understanding Celtic Coin Image Similarities"**  

The project focuses on leveraging artificial intelligence, particularly **deep learning** and **multimodal embeddings**, to analyze and compare ancient coins. It employs **feature extraction**, **similarity retrieval**, and **network analysis** to enhance numismatic research.

---

## **Project Overview**
Ancient coins hold valuable historical and cultural significance, but their classification and analysis pose challenges due to inconsistencies, wear, and subjective human interpretations. This project aims to improve **coin similarity detection** using **deep learning** and **multimodal embeddings** by integrating **computer vision** and **natural language processing** techniques.

The project includes:
- **Feature extraction** using deep learning models like **ResNet-50** and **CLIP**.
- **Multimodal fusion** of image and text embeddings via an **MLP-based approach**.
- **Similarity retrieval** to find the most similar coins based on extracted features.
- **Network analysis** to visualize the relationships between coins.

---

## **Installation**
To set up the project:

```bash
git clone https://github.com/KenanKhauto/AI-driven-coin-analysis.git
cd AI-driven-coin-analysis
```

### **Required Dependencies**
```Python 3.10.0```
The project primarily uses **PyTorch**, **OpenAI CLIP**, and other ML libraries:

```txt

Torch == 2.5.1+cu124
Torchvision == 0.20.1+cu124
NumPy == 1.26.3
Pandas == 2.2.3
Matplotlib == 3.10.0
Scikit-Learn == 1.6.0
Pillow (PIL) == 10.2.0
NetworkX == 3.2.1
openai-clip == 1.0.1 
```

For **GPU acceleration**, ensure you have an appropriate CUDA version installed.

---

## **Usage**
### **1. Feature Extraction**
The first step in the pipeline is extracting features from images using **ResNet-50** and **CLIP**:
- Run `feature_extraction.ipynb` to extract and save image embeddings.
- `clip.ipynb` to extract the features for the multimodal approach.

### **2. Similarity Retrieval**
- The extracted embeddings can be used to compute **cosine similarity** between coin images.
- In `similarites_new.ipynb` you can load and show similar and dissimilar images.

### **3. Network Analysis**
- The project also generates **graph-based visualizations** of coin relationships.
- Run `network.ipynb` to create **similarity networks** of coins.

### **4. Interactive Graph Visualizations**
The project includes **interactive similarity graphs**, where coins are represented as nodes, and edges indicate similarity relationships. These graphs help visualize how different coins are connected based on their extracted features.

You can explore the **interactive graphs** here:

- [Obverse Coin Similarity Graph (Fused Embeddings)](https://kenankhauto.github.io/AI-driven-coin-analysis/obverse_fused_graph_0.9995.html)
- [Reverse Coin Similarity Graph (Fused Embeddings)](https://kenankhauto.github.io/AI-driven-coin-analysis/reverse_fused_graph_0.9995.html)
- [Obverse Coin Similarity Graph (Image-Only)](https://kenankhauto.github.io/AI-driven-coin-analysis/obverse_img_graph_0.95.html)
- [Reverse Coin Similarity Graph (Image-Only)](https://kenankhauto.github.io/AI-driven-coin-analysis/reverse_img_graph_0.95.html)

These graphs allow for **interactive exploration**, helping to identify patterns in coin similarities across different embedding techniques.

---
### **Feature Files (.npy and .txt)**
The extracted features are stored as `.npy` and `.txt` files:
- **`features.npy`** → Contains the **feature vectors** for each coin.
- **`filenames.txt`** → Stores the corresponding **image filenames**.

These files can be loaded and used for **similarity retrieval**.

---

### **Loading Features and Computing Similarity**
You can use the following Python code to:
1. **Load the extracted features and filenames.**
2. **Compute the cosine similarity matrix.**
3. **Retrieve the most and least similar images for a given query.**

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_txt(file_path):
    """Loads a .txt file containing filenames."""
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]

# Load feature vectors and filenames
features = np.load("features.npy")  # Example: "obverse_features_gray.npy"
filenames = load_txt("filenames.txt")  # Example: "obverse_filenames_gray.txt"

# Compute the similarity matrix (Cosine Similarity)
similarity_matrix = cosine_similarity(features)

# Function to retrieve the most and least similar images
def find_similar_images(query_index, top_n=3):
    """
    Given an image index, find the top N most and least similar images.
    
    :param query_index: Index of the query image in filenames list.
    :param top_n: Number of similar/dissimilar images to retrieve.
    :return: List of most and least similar image filenames.
    """
    # Get similarity scores for the query image
    similarity_scores = similarity_matrix[query_index]

    # Get the indices of most similar (excluding itself) and least similar images
    most_similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]  # Descending order
    least_similar_indices = np.argsort(similarity_scores)[:top_n]  # Ascending order

    # Retrieve corresponding filenames
    most_similar = [(filenames[i], similarity_scores[i]) for i in most_similar_indices]
    least_similar = [(filenames[i], similarity_scores[i]) for i in least_similar_indices]

    return most_similar, least_similar

# Example usage: Find top 3 most and least similar images for the first image in the dataset
query_index = 0  # Change this index to test with other images
most_similar, least_similar = find_similar_images(query_index)

print("Query Image:", filenames[query_index])
print("\nTop 3 Most Similar Images:")
for img, score in most_similar:
    print(f"{img} (Similarity: {score:.5f})")

print("\nTop 3 Least Similar Images:")
for img, score in least_similar:
    print(f"{img} (Similarity: {score:.5f})")
```

---


## **Current State of the Repository**
- The included **notebooks and scripts** are primarily used for **testing, feature extraction, and model training**.
- A **comprehensive Jupyter notebook** will be added soon to:
  - Load extracted features.
  - Search for **similar images and coins**.
  - Provide an **interactive similarity retrieval interface**.

---

## **Citation**
If you use this work, please cite:

```
@thesis{Khauto2025,
  author = {Kenan Khauto},
  title = {From Computer Vision to Network Analysis: Comparing AI Approaches for Understanding Celtic Coin Image Similarities},
  school = {Goethe-Universität Frankfurt am Main},
  year = {2025}
}
```

---

## **License**
Creative Commons Lizenzvertrag Content is licensed under a Creative Commons Attribution - NonCommercial - ShareAlike 3.0 Germany License. http://creativecommons.org/licenses/by-nc-sa/3.0/de/

---

## **Acknowledgments**
Special thanks to **Dr. Karsten Tolle** , **Sebastian Gampe** and the **Goethe-Universität Frankfurt am Main** for their guidance and support.
