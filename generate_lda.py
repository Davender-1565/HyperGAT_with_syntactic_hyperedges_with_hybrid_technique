import argparse
import pickle
from sklearn.utils import class_weight
from nltk.corpus import stopwords
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from utils import clean_str, clean_str_simple_version, show_statisctic, clean_document
import sys
from nltk import tokenize
import collections
from collections import Counter
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def display_topics(model, feature_names, no_top_words):
	keywords_dic = {}
	for topic_idx, topic in enumerate(model.components_):
		print("Topic %d:" % (topic_idx))
		klist = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
		print(" ".join(klist))
		for k in klist:
			if not k in keywords_dic:
				keywords_dic[k] = []
			keywords_dic[k].append(topic_idx)
	return keywords_dic

def Generate_LDA(dataset):
	doc_content_list = []
	doc_sentence_list = []
	f = open('data/' + dataset + '_corpus.txt', 'rb')
	for line in f.readlines():
	    doc_content_list.append(line.strip().decode('latin1'))
	    doc_sentence_list.append(tokenize.sent_tokenize(clean_str_simple_version(doc_content_list[-1], dataset)))
	f.close()

	# Remove the rare words
	doc_content_list = clean_document(doc_sentence_list, dataset)
	
	doc_list = []
	for doc in doc_content_list:
		temp = ''
		for sen in doc:
			temp += (' '.join(sen) + ' ')
		doc_list.append(temp)

	vectorizer = CountVectorizer(stop_words='english',max_df=0.98)
	vector = vectorizer.fit_transform(doc_list)
	feature_names = vectorizer.get_feature_names_out()  # Updated for scikit-learn >= 1.0
	lda = LatentDirichletAllocation(n_components=args.topics,learning_method='online',learning_offset=50.,random_state=0).fit(vector)

	keywords_dic = display_topics(lda, feature_names, args.topn)
	print(len(keywords_dic))
	print(keywords_dic)

	pickle.dump(keywords_dic, open('data/' + dataset + '_LDA.p', "wb" ) )


#for syntactic hyperedges
def generate_syntactic_hyperedges(doc_content_list, embedding_model, similarity_threshold=0.5):
    import spacy
    nlp = spacy.load('en_core_web_sm')
    syntactic_hyperedges = collections.defaultdict(list)
    
    for doc_id, doc in enumerate(doc_content_list):
        for sentence in doc:
            text = ' '.join(sentence)
            parsed = nlp(text)
            dependencies = [(token.head.lemma_, token.dep_, token.lemma_) 
                           for token in parsed if token.dep_ not in ['punct']]
            
            for head, dep, child in dependencies:
                try:
                    head_vec = embedding_model[head]
                    child_vec = embedding_model[child]
                    sim = np.dot(head_vec, child_vec)/(np.linalg.norm(head_vec)*np.linalg.norm(child_vec))
                    if sim > similarity_threshold:
                        edge_id = f"syn_{doc_id}_{head}_{child}"
                        syntactic_hyperedges[head].append(edge_id)
                        syntactic_hyperedges[child].append(edge_id)
                except KeyError:
                    continue
    return syntactic_hyperedges

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='20ng', 
                       help='dataset name: 20ng/R8/R52/ohsumed/mr/StackOverflow/Tweet/GoogleNews')
    parser.add_argument('--topn', type=int, default=10, help='top n keywords')
    parser.add_argument('--topics', type=int, help='number of topics')
    args = parser.parse_args()
    print(args)

    Generate_LDA(args.dataset)
	