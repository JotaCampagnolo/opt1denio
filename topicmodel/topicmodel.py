#link https://goo.gl/8h8qj5
import numpy as np ## Matrix package
import nltk ## methods for natural language processing

## package to deal with stop words
from nltk.corpus import stopwords
## lemmatizer package
from nltk.stem.wordnet import WordNetLemmatizer
## punctuation
import string
## gensim is a package with a lot of topic modeling stuffs
# link https://goo.gl/8h8qj5
import gensim
import gensim.models.ldamodel as glda
from gensim import corpora
fdoc = open('en09062011_snippets.txt')
docs = fdoc.readlines()
## show the first document
print(docs[0])
## transform all documents to lower case
docs=[d.lower() for d in docs]
## show again the first document
print(docs[0])
## stop is a set of stop-words
stop = set(stopwords.words('english'))
## punct is a set of punctuation (.,?! etc)
punct = set(string.punctuation)
## object to deal with stemming
stporter=nltk.stem.PorterStemmer()
## object to deal with lemmatization
lemma = WordNetLemmatizer()
## the following function cleans one document (parameter)
def clean_doc(doc):
	# remove stop words
	doc=' '.join([w for w in doc.split() if w not in stop])
	# remove punctuation
	doc=''.join(ch for ch in doc if ch not in punct)
	# apply porter stemming
	doc=' '.join(stporter.stem(w) for w in doc.split())
	# apply lemmatization
	doc=' '.join(lemma.lemmatize(w) for w in doc.split())
	## returns the document as a list
	return doc
###
### Now, we can prepare our collection (clean it)
docs_clean=[clean_doc(d).split() for d in docs]
print('Numero de documentos',len(docs_clean))
# build the dictionary
dict = corpora.Dictionary(docs_clean)
## remove some words by frequency
dict.filter_extremes(no_below=2, no_above=0.8, keep_n=1000)
## transform the words in documents into id and frequency
doc_ids=[dict.doc2bow(doc) for doc in docs_clean]
## show first document as a bag-of-words
print('Ids Documento 0',doc_ids[0])
## build the model using LDA (20 topics and default hyper parameters)
lda = glda.LdaModel(doc_ids,num_topics=7,id2word=dict)
## the first 5 words (id) from topic 0
w_top=lda.get_topic_terms(0,topn=5)
print(w_top[0]) # print the first word id and its probability
print(w_top[0][0]) # only the id
print(w_top[0][1]) # only the probability
## retrieve all top 10 words from all topics
tpcs=[]
for t in range(7): ## 20 = número de tópicos
	words=lda.get_topic_terms(t,topn=10)
	tpcs.append([dict.get(w[0]) for w in words])
## print topic 0's words
for i in range(len(tpcs)):
	print(tpcs[i])
# print the topics and their probabilities for doc 0
print('Topics and probabilities for doc 0\n',lda.get_document_topics(doc_ids[0],minimum_probability=0))
## evaluate the topics using u_mass metric
import gensim.models.coherencemodel as cm
##
mycm=cm.CoherenceModel(model=lda,corpus=dict,texts=docs_clean,coherence='u_mass')
print('U_MASS score for all topics:',mycm.get_coherence())
## evaluate the topics using c_uci metric
mycm=cm.CoherenceModel(model=lda,corpus=dict,texts=docs_clean,coherence='c_uci')
print('C_UCI score for all topics:',mycm.get_coherence())
## get the C_UCI score for topic 0
mycm=cm.CoherenceModel(topics=[tpcs[0]],dictionary=dict,texts=docs_clean,coherence='c_uci')
print('Topic 0 score:',mycm.get_coherence())
