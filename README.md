# Vctor-space-model-of-retrieval
Part of 2021 summer project. A basic implementation of vector space paradigm in IR.
---
## Components
1. main.py <br />
Takes user unput and initiates searching process based on a generic query.
2. construct_database.py <br />
Assigns doc IDs to files inside the directory named *files*, creates a term database (a dict) of the following format. <br/> 
{term<sub>i</sub> : ndarray((ndarray(docID<sub>i</sub></sub>, tf<sub>term<sub>i</sub>, docID<sub>i</sub></sub>)...)...}.
3. relelevant_functions.py <br />
calculates tf-idf, vectorizes the query as well as the documents, finds their similarity coefficient (dot prod).
---
## Requirements
1. Python 3.x
2. nltk
3. numpy
---
## How to use
* Create a new directory and keep the above mentioned files inside the same.
* Create ```/files``` storing the documents to perform the retrieval task on.
* Run ```main.py``` from the main directory. ```1 + <enter>``` to search.
---
## Fixes needed
* Porter stemmer
* More refined ranking
