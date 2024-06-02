from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from functions import *
import numpy as np
import math
from sklearn.datasets import load_iris
import random
from sklearn import preprocessing

#import random
random_state=None
seed = None if random_state is None else int(random_state)
rng = np.random.default_rng(seed=seed)

with open("dataset/shakespeare.txt") as data:
    text_data = data.read().lower()

tokens = word_tokenize(text_data)[0:500]
vocabulary = list(set(tokens))

print("size tokens: ", len(tokens))

print("size vocabulary: ", len(vocabulary))
print("size vector vocabulary: ", len(vocabulary[0]))

# Generate one-hot encoded vectors for each word in the vocabulary
one_hot_encoded = []
for word in tokens:
    # Create a list of zeros with the length of the vocabulary
    encoding = [0] * len(tokens)
    
    # Get the index of the word in the vocabulary
    index = list(tokens).index(word)
    
    # Set the value at the index to 1 to indicate word presence
    encoding[index] = 1.0
    one_hot_encoded.append((word, encoding))
