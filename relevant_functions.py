import numpy as np
import pickle
from construct_database import tot_doc, process_query, make_the_database

with open('term_database_dict', 'rb') as infile:
        data = pickle.load(infile)

def show_database(read_database):
    """ prints the term database if exists; creates one if not
    """
    term_database = read_database()
    print(term_database)
    

def calculate_tf_idf(term, docid, term_database):
    try:
        tot_doc_no = tot_doc
        term_details = term_database[term]
        doc_freq = np.size(term_details, 0)
        i = np.where(term_details[:, 0] == docid)[0][0]
        return\
            term_details[i][1] * np.log(1.0*tot_doc_no/doc_freq)/np.log(10)
    except:
        return 0

def find_doc_vector(words, docid, database):
    vector = np.array([])
    for element in words:
        tf_idf = calculate_tf_idf(element, docid, database)
        vector = np.append(vector, tf_idf)
    return vector 

def vectorize_query(query_words):
    words = dict()
    for element in query_words:
        if element not in words:
            words[element] = 1
        else:
            words[element] += 1
    return [np.array([i for i in words.values()])\
        , words.keys()]

def normalize(vector):
    val = np.sqrt(np.sum(vector**2))
    if val == 0:
        return vector
    else:
        return vector / val
        

def similarity_value(vec1, vec2):
    #vec1 = normalize(vec1)
    #vec2 = normalize(vec2)
    return np.sum(vec1 * vec2)
