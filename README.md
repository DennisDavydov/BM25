## Overview
This project implements a simple yet powerful text search engine using the inverted index data structure and the BM25 ranking algorithm. It is designed to efficiently index documents and rank search results based on query relevance, demonstrating fundamental concepts in information retrieval systems.

The core of the project is split into two main components:

inverted_index.py: Builds an inverted index from a collection of documents. It utilizes the BM25 algorithm to calculate relevance scores between documents and queries.
evaluate.py: Evaluates the effectiveness of the inverted index by comparing search results against a benchmark dataset, using metrics such as precision at K, recall, and average precision.

## Usage
First, you need to build the inverted index from your dataset:

```bash
python inverted_index.py <path-to-your-dataset>
```

To evaluate the performance of your search engine:

```bash
python evaluate.py <path-to-your-dataset> <path-to-benchmark-data>
```
