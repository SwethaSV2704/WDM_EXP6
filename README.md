### EX6 Information Retrieval Using Vector Space Model in Python
### DATE: 24-11-2026
### AIM: To implement Information Retrieval Using Vector Space Model in Python.
### Description: 
<div align = "justify">
Implementing Information Retrieval using the Vector Space Model in Python involves several steps, including preprocessing text data, constructing a term-document matrix, 
calculating TF-IDF scores, and performing similarity calculations between queries and documents. Below is a basic example using Python and libraries like nltk and 
sklearn to demonstrate Information Retrieval using the Vector Space Model.

### Procedure:
1. Define sample documents.
2. Preprocess text data by tokenizing, removing stopwords, and punctuation.
3. Construct a TF-IDF matrix using TfidfVectorizer from sklearn.
4. Define a search function that calculates cosine similarity between a query and documents based on the TF-IDF matrix.
5. Execute a sample query and display the search results along with similarity scores.

### Program:
```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

documents = {
    "doc1": "This is the first document.",
    "doc2": "This document is the second document.",
    "doc3": "And this is the third one.",
    "doc4": "Is this the first document?"
}

doc_ids = list(documents.keys())
texts = list(documents.values())

count_vectorizer = CountVectorizer(stop_words="english",
    vocabulary=["document", "first", "one", "second", "third"])

tf_matrix = count_vectorizer.fit_transform(texts).toarray()
terms = count_vectorizer.get_feature_names_out()

print("\n--- Term Frequencies (TF) ---\n")

tf_table = [[doc_ids[i]] + list(tf_matrix[i])
            for i in range(len(doc_ids))]

print(tabulate(tf_table, headers=["Doc ID"] + list(terms),
               tablefmt="grid"))

tfidf_vectorizer = TfidfVectorizer(
    stop_words="english",
    vocabulary=list(terms))

tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

df = np.sum(tf_matrix > 0, axis=0)
idf = tfidf_vectorizer.idf_

print("\n--- Document Frequency (DF) and Inverse Document Frequency (IDF) ---\n")

df_table = [[terms[i], df[i], round(idf[i], 4)]
            for i in range(len(terms))]

print(tabulate(df_table, headers=[
    "Term", "Document Frequency (DF)",
    "Inverse Document Frequency (IDF)"], tablefmt="grid"))

print("\n--- TF-IDF Weights ---\n")

weights = tfidf_matrix.toarray()

weight_table = [[doc_ids[i]] + list(np.round(weights[i], 4))
                for i in range(len(doc_ids))]

print(tabulate(weight_table, headers=["Doc ID"] + list(terms),
               tablefmt="grid"))

def search(query, tfidf_matrix, tfidf_vectorizer):
    query_vector = tfidf_vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    results = []

    for i, doc_id in enumerate(doc_ids):
        dot_product = np.dot(
            query_vector.toarray()[0], weights[i])

        query_mag = np.linalg.norm(query_vector.toarray())
        doc_mag = np.linalg.norm(weights[i])

        results.append([
            doc_id, documents[doc_id],
            round(dot_product, 4),
            round(query_mag, 4),
            round(doc_mag, 4),
            round(scores[i], 4)
        ])

    results.sort(key=lambda x: x[5], reverse=True)
    return results, query_vector

query = input("\nEnter your query: ")

search_results, query_vector = search(
    query, tfidf_matrix, tfidf_vectorizer)

print("\n--- Search Results and Cosine Similarity ---\n")

print(tabulate(search_results, headers=[
    "Doc ID", "Document", "Dot Product",
    "Query Magnitude", "Doc Magnitude",
    "Cosine Similarity"], tablefmt="grid"))

print("\n--- Query TF-IDF Weights ---\n")

query_weights = query_vector.toarray()[0]

query_table = [[terms[i], round(query_weights[i], 4)]
               for i in range(len(terms))
               if query_weights[i] > 0]

print(tabulate(query_table, headers=[
    "Term", "Query TF-IDF Weight"], tablefmt="grid"))

print("\n--- Ranked Documents ---\n")

ranked_table = [[i + 1, result[0], result[1], result[5]]
                for i, result in enumerate(search_results)]

print(tabulate(ranked_table, headers=[
    "Rank", "Document ID", "Document Text",
    "Cosine Similarity"], tablefmt="grid"))

highest = search_results[0]

print("\nThe highest rank cosine score is:",
      highest[5], f"(Document ID: {highest[0]})")
```

### Output:
<img width="1122" height="687" alt="image" src="https://github.com/user-attachments/assets/e4508b6a-9937-43ab-8ad2-2574f3fa6e4b" />
<img width="1302" height="645" alt="image" src="https://github.com/user-attachments/assets/2bda34b0-4c78-4ac3-823d-b5c5b0f4fa97" />
<img width="962" height="533" alt="image" src="https://github.com/user-attachments/assets/1f0eb53e-400e-4a4a-ac3f-f7b5f58d11ae" />

### Result:
Thus, the Information Retrieval system using the Vector Space Model was successfully implemented in Python, and documents were ranked based on cosine similarity using TF-IDF weighting.
