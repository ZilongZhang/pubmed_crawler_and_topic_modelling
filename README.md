# pubmed_crawler_and_topic_modelling
A tool consist of a pubmed's abstracts crawler and a topic modelling pipeline. Could be helpful to people who want to have some ideas about what's going on in a specific medical field. 
# Packages used:
Python 3.8

Spacy (need to be installed via conda by `conda install -c conda-forge spacy
==2.3.4`)

A english model for spacy, installed by `python -m spacy install en_web_core_sm`

request    2.24.0

beautifulsoup4    4.9.1

jupyter    1.0.0

scikit-learn    0.23.2

tqdm

# Input and output:
Input: list of keywords 

Output: a topic model trained on abstracts about the defined keywords

# How to run this script:
Go to the pubmed_crawler_lda.ipynb and change the list of keywords(searching terms) and run the rest of notebook.

# Parameters you could play with:
  * page_number in function `fetch_abstract`, will change the number of crawled pages. The default is 10 and each page contains 10 papers, so it will crawl abstracts from 100 papers.
  * topic_num in function `lda_pipeline`, will change the number of output topics
  * iter_num in main script, the meaning of the parameter is explained below:
  
# Iterative stopwords generation process:
Because all abstract are from the searching result of the same keywords list, so certain related words in that field will appear frequently, for example, machine learning related papers always have word like classifer, classify, model etc. this result in meaningless repetitive topics. So this script will run topic modelling for many times(default = 5) and define words that appeared at more than 20% of topics as new stopwords(sw). Hopefully it will produce more sensible topics. The parameter iter_num is about how many time this process will run
