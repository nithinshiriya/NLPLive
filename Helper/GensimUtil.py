from gensim.corpora import Dictionary, MmCorpus
from gensim.models.word2vec  import LineSentence

def generate_dictionary(input_file_path, applyExtreem = True, no_below=5, no_above=0.4):
    lineSentence = LineSentence(input_file_path)
    dictionary  =  Dictionary(lineSentence) 
    if applyExtreem:
        dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    dictionary.compactify() 
    return dictionary  

def save_dictionary(dictionary, file_path):
    dictionary.save(file_path)

def generate_corpus(input_file_path, applyExtreem = True, no_below=5, no_above=0.4):
    dictionary = generate_dictionary(input_file_path, applyExtreem, no_below, no_above)
    return [dictionary.doc2bow(text) for text in LineSentence(input_file_path)]

def save_corpus(corpus, corpus_path):
    MmCorpus.save_corpus(corpus_path, corpus)

def get_corpus(corpus_path):
    return MmCorpus(corpus_path)

def get_dictionary(dictionary_path):
    return Dictionary.load(dictionary_path)


def add_doc_to_dictionary(dictionary, new_dictionary):
    tokens = []
    doc = []
    [tokens.append(token) for token in new_dictionary.itervalues()]        
    doc.append(tokens)
    dictionary.add_documents(doc)  