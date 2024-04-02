import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """ 
    Example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi","hello","I","you"]
    bog = [0,1,0,1]
    """

    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag



#----------Tokenizer--------------------
# a = "Writers like Shakespeare and Jane Austen have crafted timeless stories that continue to captivate readers worldwide."
# print(a)
# a = tokenize(a)
# print(a)

#----------Stem---------------------------
# words = ['Writers', 'like', 'Shakespeare', 'and', 'Jane', 'Austen', 'have', 'crafted', 'timeless', 'stories', 'that', 'continue', 'to', 'captivate', 'readers', 'worldwide', '.']
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)

