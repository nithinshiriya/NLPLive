from gensim.corpora import Dictionary, MmCorpus
from gensim.models.word2vec  import LineSentence
from Helper import Config, LFUtil, GensimUtil, Trigram
from Models import LDAModel
import os


# Helper method for LDA object
lda = LDAModel.LDAModel()

# Load all existing model, dictionary, index , bigram and trigram
lda_model = lda.get_model(Config.lda_model)
lda_index = lda.get_index(Config.lda_model_index)
dictionary =  GensimUtil.get_dictionary(Config.dictionary_file)
bigram= Trigram.get_model(Config.bigram_model)
trigram= Trigram.get_model(Config.trigram_model)


TrainingPhrase ="Knowing about the progress and performance of a model, as we train them, could be very helpful in understanding it’s learning process and makes it easier to debug and optimize them. In this notebook, we will learn how to visualize training statistics for LDA topic model in gensim. To monitor the training, a list of Metrics is passed to the LDA function call for plotting their values live as the training progresses."

# Write the training phrase to text file.
LFUtil.create_file(Config.training_file, TrainingPhrase)

# Get dictionary, corpus and lda model from tarining phrase
new_dictionary = GensimUtil.generate_dictionary(Config.training_file, applyExtreem=False)
new_corpus = GensimUtil.generate_corpus(Config.training_file, applyExtreem = False)
new_ldaModel =  lda.create_model(new_dictionary, new_corpus)

# ========================================
# Train the new document with existing one
# ========================================
GensimUtil.add_doc_to_dictionary(dictionary, new_dictionary)
lda_model.update(new_corpus)
lda_index.add_documents(new_ldaModel[new_corpus])


#======================================================
# Let's test the Similarity 
vec_bow = dictionary.doc2bow(TrainingPhrase.split(" "))
vec_model =  lda_model[vec_bow]
sims = lda_index[vec_model]
sims = sorted(enumerate(sims), key=lambda item: -item[1])

# print the result
results = []
for sim in sims:
    if sim[1] >= 0.90:
        results.append(sim[0])
print(results)


