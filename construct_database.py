import pickle
import os
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as ps


# defining the punctuation list
# ord() and chr()
punc_marks = ['?', ',', '!', '.', '-', ':', ';', '/',\
    chr(8216), chr(8217), chr(8220), chr(8221), '(', ')']

# constructing the docID matrix first
docids = dict()
_id = 1
with os.scandir("files/") as entries:
    for entry in entries:
        docids[_id] = entry.name
        _id += 1

# total document #, namely C
tot_doc = _id - 1

# construction of the term database
def make_the_database():
    # the dict data holder
    term_database = dict()

    for i in range(1,_id):
        path_new = 'files/' + docids[i]
        # opening and reading the txt file content
        with open(path_new, encoding="utf8") as f:
            temp_store = f.read()
        # tokenization
        temp_store = word_tokenize(temp_store)

        # removing stop words
        stop_words = list(stopwords.words("english"))
        filtered_store = [element for element in temp_store\
            if element.casefold() not in stop_words]

        # removing punctuations
        filtered_store = [element for element in filtered_store\
            if element not in punc_marks]
        
        # stemming (using Porter stemmer)
        stemmer = ps()
        stemmed_store = [stemmer.stem(element) for element in filtered_store]

        # creating the inverted matrix 
        for element in stemmed_store:
            if element not in list(term_database.keys()):
                term_database[element] = np.array([[i, 1]])
            else:
                if term_database[element][-1][0] == i:
                    term_database[element][-1][1] += 1
                else:
                    term_database[element] =\
                        np.append(term_database[element],[[i,1]], 0)
        
        # the final dictionary is not sorted alphabetically 

    # pickled database with term details
    with open('term_database_dict', 'wb') as outfile:
        pickle.dump(term_database, outfile)

    # pickled database for doc ids vs file names
    with open("doc_ids", 'wb') as outfile:
        pickle.dump(docids, outfile)

# processing the query, returns the stemmed bag of words
# shall be invoked in main.py
def process_query(query):

    # tokenization
    tokenized_query = word_tokenize(query)

    # removing stop words and punctuations 
    stop_words = list(stopwords.words("english"))
    stpwrd_punc_removed = [element for element in tokenized_query\
        if element not in stop_words and element not in punc_marks]
        
    # stemming (using Porter stemmer)
    stemmer = ps()
    stemmed_store = [stemmer.stem(element) for element in stpwrd_punc_removed]

    return stemmed_store