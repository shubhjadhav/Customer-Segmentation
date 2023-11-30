import pandas as pd
import os
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

stop_words = [
    'can', 'its', 'each', 'with', 'their', 'is', 'into',
    'are', 'as', 'of', 'by', 'those', 'just', 'only', 'after',
    'a', 'to', 'they', 'through', 'he', 'for', 'him', 'in', 'be',
    'other', 'an', 'most', 'has', 'all', 'but', 'from', 'been',
    "it's", 'over', 'own', 'not', 'that', 'out',
    'and', 'his', 'on', 'the', 'about'
]


data_path = "../Data/"


def get_file_path(file_name):
    dir_path = os.getcwd()
    print("Current work Directory", dir_path)
    file_path = dir_path + os.sep + file_name
    print("File Path is ", file_path)
    return file_path


def create_df(file_name):
    file_path = get_file_path(file_name)
    invoice_df = pd.read_csv(data_path + file_name + '.csv', encoding='unicode_escape')
    invoice_df['InvoiceDate'] = pd.to_datetime(invoice_df['InvoiceDate'])
    return invoice_df


def print_observation(text):
    print("OBSERVATION: ", text)


# Remove unwanted text
def get_alphanum(sentences):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    res = []
    matches = pattern.findall(sentences.lower())
    res.append(" ".join(matches))
    return res


# Remove Stop words
def remove_stop_words(sentences):
    results = []
    for text in sentences:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))
    return results


def clean_text(text):
    cleaned_text_file = get_alphanum(text)
    corpus = remove_stop_words(cleaned_text_file)
    return corpus


def create_word_mappings_from_corpus(corpus, top=0):

    # tokenize
    word_list = []
    for obs in corpus:
        word_list.extend(obs.split(' '))

    # Create BOW vocabulary
    counter = Counter(word_list)
    
    high_freq = [i[0] for i in counter.most_common(3)]
    
    if top == 0:
        vocab = [i[0] for i in counter if i[0] not in high_freq]
    else:
        vocab = [i[0] for i in counter.most_common(top) if i[0] not in high_freq]

    # Create word mapping file
    word_to_id = {}
    for i, token in enumerate(vocab):
        word_to_id[token] = i

    return counter, vocab, word_to_id

def create_word_mappings_from_vocab(vocab, top=0):

    # Create word mapping file
    word_to_id = {}
    for i, token in enumerate(vocab):
        word_to_id[token] = i

    return vocab, word_to_id


def create_embeddings(word_to_id, corpus, embed_dim):
    X = []

    for desc in corpus:
        one_hot_encoding = np.zeros(embed_dim)

        for word in desc.split(' '):
            if word in word_to_id:
                one_hot_encoding[word_to_id[word]] = 1

        X.append(one_hot_encoding)

    return np.asarray(X)


def get_kmeans(X, feature_names, max_clusters, title, plot = True ):
    
    X = X[feature_names]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  
    
    wcss = []
    silhouette_scores = []
    all_clusters = []

    for i in range(3, max_clusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)

        wcss.append(kmeans.inertia_)
        clusters = kmeans.predict(X_scaled)
        all_clusters.append(clusters)
        silhouette_avg = silhouette_score(X_scaled, clusters)
        silhouette_scores.append(round(silhouette_avg, 2))

        print("For n_clusters =", i, "The average silhouette_score is :", round(silhouette_avg,2))
    
    if plot: 
        # Plot the elbow graph to find the optimal number of clusters
        plt.plot(range(3, max_clusters), wcss)
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS')
        plt.show()
        
    clusters = int(input("Give best Cluster"))
    plt.hist(all_clusters[clusters-3], alpha=0.5)
    plt.title(title)
    plt.xlabel('Cluster Number')
    plt.ylabel('count')
    plt.show()
