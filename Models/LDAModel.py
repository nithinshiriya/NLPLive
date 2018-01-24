import warnings
import os
from Helper import Config
from gensim import models, similarities

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=UserWarning, module='utils')

class LDAModel:
    __dictionary = None
    __corpus = None

    def __init__(self):
        pass

    def setCorpus(self, corpus, dictionary):
        self.__corpus = corpus
        self.__dictionary = dictionary

    ## Generate the model, save and return the model
    def generate_model(self, model_save_path, topics=50):
        model = self.create_model(self.__dictionary, self.__corpus, topics)
        model.save(model_save_path)
        return self.get_model(model_save_path)

    def create_model(self, dictionary, corpus, topics=50):
        return self.prepare_model(corpus, dictionary, topics)
    
    def generate_index_matrix(self, index_save_path, topics=50):
        model = self.prepare_model(self.__corpus, self.__dictionary, topics)
        index = similarities.MatrixSimilarity(model[self.__corpus])
        index.save(index_save_path)
        return self.get_index_matrix(index_save_path)

    def generate_index(self, index_name, index_save_path, topics=50):
        model = self.prepare_model(self.__corpus, self.__dictionary, topics)
        index = similarities.Similarity(index_name, model[self.__corpus], topics)
        index.save(index_save_path)
        return self.get_index(index_save_path)    

    ## Load model from the file path
    def get_model(self, model_path):
        return  models.LdaModel.load(model_path)        

    ## Get Sim index file
    def get_index_matrix(self, index_path):
        return similarities.MatrixSimilarity.load(index_path) 

    def get_index(self, index_path):
        return similarities.Similarity.load(index_path)

    ## Prepare Model
    def prepare_model(self, corpus, dictionary, topics):
        return models.LdaModel(corpus, id2word=dictionary, num_topics=topics)
    
    ## Save Model to file
    def __save_model(self, corpus, dictionary, model_save_path, topics):
        model = self.prepare_model(corpus, dictionary, topics)
        model.save(model_save_path)

    def add_documents(self, model, corpus):
        model.update(corpus)


