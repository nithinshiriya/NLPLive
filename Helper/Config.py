import os

dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
root_directory = os.path.join(dir, "Files")
lda_model = os.path.join(root_directory, "lda_model.model")
lda_model_index_name ="lda_model"
lda_model_index = os.path.join(root_directory, lda_model_index_name + ".index")
lda_model_index_path = os.path.join(root_directory, lda_model_index_name)

dictionary_file = os.path.join(root_directory, "dictionary.dict")
corpus_file = os.path.join(root_directory, "corpus.mm")

bigram_model = os.path.join(root_directory, "bigram_model")
trigram_model = os.path.join(root_directory, "trigram_model")

training_file = os.path.join(root_directory, "tarining_file.txt")

