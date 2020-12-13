import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import pickle
import numpy as np

prefix = 'https://pubmed.ncbi.nlm.nih.gov/'

def get_tmp_url(kw_list):
    new_list = []
    for kw in kw_list:
        kw=kw.replace(' ','%20')
        kw='(' + kw + ')'
        new_list.append(kw)
    term_str = 'term=(' + '%20AND%20'.join(new_list) + ')'
    search_url = 'https://pubmed.ncbi.nlm.nih.gov/?' + term_str + '&sort=&page='
    return search_url

def custom_sw(sw_list):
    import spacy.lang.en
    import copy
    spacy_sw = spacy.lang.en.stop_words.STOP_WORDS
    sw = copy.deepcopy(spacy_sw)
    for word in sw_list: sw.add(word)
    return sw
    
def batch_lemma(text_list):
    nlp = spacy.load("en_core_web_sm")
    lemmatized_text = []
    for doc in nlp.pipe(text_list, batch_size=10,n_process=6, disable=['parser', 'ner']):
        tmp_doc = [tok.lemma_ for tok in doc]
        tmp_doc = ' '.join(tmp_doc)
        lemmatized_text.append(tmp_doc)
    return lemmatized_text

def produce_links(index,search_url):
    link_list = []
    tmp_url = search_url + str(index)
    #print(tmp_url)
    page = requests.get(tmp_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    href_divs = soup.findAll("div", {"class": "docsum-content"})
    for p_job in href_divs:
        link = p_job.find('a')['href']
        link_list.append(link)
    return link_list
    
def fetch_abstract(kw_list,page_number = 10):
    search_url = get_tmp_url(kw_list)
    abstract_list = []
    for i in range(1,page_number+1):
        tmp_link_list = produce_links(i,search_url)
        for doc_id in tqdm(tmp_link_list):
            url = prefix + doc_id
            page = requests.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')
            abs_divs = soup.findAll("div", {"class": "abstract-content selected"})
            for p_job in abs_divs:
                abstract = p_job.find('p')
                abstract_list.append(str(abstract))
                time.sleep(0.5)
    return abstract_list
    
def lda_pipeline(text_list,topic_num=20,do_lemma=False,ngram=1,sw_list=[],suggest_sw = False,print_model = False):
    sw = custom_sw(sw_list)
    vectorizer = CountVectorizer(ngram_range = (1,ngram),stop_words = sw,max_df=0.8, min_df=3,token_pattern = '[A-Za-z]{3,}')
    if not do_lemma:
        matrix = vectorizer.fit_transform(text_list)
    else:
        print('Lemmatizing texts ...')
        lemma_text_list = batch_lemma(text_list)
        matrix = vectorizer.fit_transform(lemma_text_list)
        pickle.dump(lemma_text_list,open('lemma_text_list.pkl','wb'))
        print('Lemmatized text was restored in lemma_text_list.pkl')
    lda = LatentDirichletAllocation(n_components=topic_num, learning_method='batch',n_jobs=5,max_iter = 20)
    lda = lda.fit(matrix)
    topics = lda.transform(matrix)
    topic_count = np.sum(topics,axis=0)
    topic_sort = np.argsort(topic_count)
    rev_sort = []
    for i in range(len(topic_sort)-1,-1,-1):
        rev_sort.append(topic_sort[i])
    representative_words,representative_weight = print_topic_word(lda,vectorizer,matrix,word_num=10,weight_list = rev_sort,print_model=print_model)
    topic_count = np.sum(topics,axis=0)
    sw = custom_sw(sw_list)
    topic_count = topic_count / np.sum(topic_count)
    log = 'Topic {}: {:0.3f} '
    if print_model:
        for i in np.argsort(-topic_count):
            print(log.format(i,topic_count[i]))
    if not suggest_sw:
        return lda
    else:
        index_2_word_dict = vectorizer.get_feature_names()
        new_sw = []
        for ind in set(representative_words.flatten()):
            if np.sum(representative_words==ind) >= 0.2 * np.shape(representative_words)[0]:
                new_sw.append(index_2_word_dict[ind])
        return lda,new_sw
                
        
    
def print_topic_word(topic_model,vectorizer,count_vec,word_num=20,weight_list=None,print_model = False):
    tf = np.sum(count_vec,axis=0)
    tf = np.squeeze(np.asarray(tf))
    index_2_word_dict = vectorizer.get_feature_names() #reverse_dict(vectorizer.vocabulary_)
    topic_num = np.shape(topic_model.components_)[0]
    feature_num = len(vectorizer.vocabulary_)
    norm_score = topic_model.components_ / topic_model.components_.sum(axis=1)[:, np.newaxis]
    representative_words = np.zeros((topic_num,word_num),dtype=np.int32)
    representative_weight = np.zeros((topic_num,word_num))
    if not weight_list:
        for i in range(topic_num):
            tmp_score = norm_score[i,:]#*tf
            representative_words[i,:] = np.argsort(tmp_score)[-word_num:]
            pdb.set_trace()
            representative_weight[i,:] = np.sort(tmp_score)[-word_num:]
            line = 'Topic '+str(i)
            for j in range(word_num-1,0,-1):
                line += ' ' + index_2_word_dict[representative_words[i,j]]+','
            if print_model:
                print(line+'\n')
    else:
        #import pdb; pdb.set_trace()
        for i in weight_list:
            tmp_score = norm_score[i,:]#*tf
            representative_words[i,:] = np.argsort(tmp_score)[-word_num:]
            representative_weight[i,:] = np.sort(tmp_score)[-word_num:]
            #import pdb;pdb.set_trace()
            line = 'Topic '+str(i)
            for j in range(word_num-1,0,-1):
                line += ' ' + index_2_word_dict[representative_words[i,j]]+','
            if print_model:
                print(line+'\n')
    return representative_words,representative_weight
  