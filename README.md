# TF-IDF N-grams Cosine Similarity

Small educational script that compares **three texts** using **TF-IDF** features built from **n-grams** and computes **pairwise cosine similarity**.

## What it does
- Reads `text1.txt`, `text2.txt`, `text3.txt`
- Preprocesses text (tokenize → lowercase → keep alphabetic tokens)
- Builds **1-gram**, **2-gram**, and **3-gram** representations
- Computes a term-frequency matrix and applies **IDF** and **TF-IDF**
- Prints cosine similarity for:
  - each n separately (1, 2, 3)
  - union of 1+2+3 n-grams
