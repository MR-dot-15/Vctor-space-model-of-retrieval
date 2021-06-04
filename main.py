import construct_database as cd
cd.make_the_database()  

import relevant_functions as rf
import numpy as np
import pickle

def read_database():
    with open('term_database_dict', 'rb') as infile:
        term_database = pickle.load(infile)
    return term_database 

with open('doc_ids', 'rb') as infile:
    docids = pickle.load(infile)

def init_search():
    query = input()
    query_words = cd.process_query(query)
    database = read_database()
    [vec_query, word_bag_to_check] = \
        rf.vectorize_query(query_words)
    ranking_value = np.zeros((cd.tot_doc,2))

    for i in range(1, cd.tot_doc + 1):
        vec_doc = rf.find_doc_vector(word_bag_to_check, i, database)
        sim_rank = rf.similarity_value(vec_query, vec_doc)
        ranking_value[i-1][0], ranking_value[i-1][1]=\
            i, sim_rank

    ranking_value =\
        ranking_value[ranking_value[:,1].argsort()[::-1]]
    best_val = int(0.6 * cd.tot_doc)
    return ranking_value[:3,:]


while True:
    print(" 1. search\n 2. print the database\n 3. exit")
    inp = int(input())
    if inp == 1:
        result, i = init_search(), 0
        while i < np.size(result, 0):
            print(docids[result[i][0]],\
                ":similarity index = %.3f"%result[i][1])
            i+=1
    elif inp == 2:
        rf.show_database(read_database)
    elif inp == 3:
        exit()