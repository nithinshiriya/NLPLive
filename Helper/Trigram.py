from Helper import Config
import os
import codecs
from gensim.models.word2vec  import LineSentence
from gensim.models import Phrases

def __gram_to_text(sentences, model, filepath):
    with codecs.open(filepath, 'w', encoding=u'utf_8') as f:
        for sentence in sentences:
            model_sentence = u' '.join(model[sentence])
            f.write(model_sentence + '\n')


def generate_trigram(input_file_path, bigram_save_path, trigram_save_path, final_file_path):
    unigram_sentence  = LineSentence(input_file_path)
    bigram = Phrases(unigram_sentence)
    bigram.save(bigram_save_path)

    # temp strore bigram sentance to get the trigram
    temp_file_path = os.path.join(Config.temp_directory,  "temp_bigram.txt")
    __gram_to_text(unigram_sentence, bigram, temp_file_path)

    bigram_sentences = LineSentence(temp_file_path)
    trigram = Phrases(bigram_sentences)
    trigram.save(trigram_save_path)
    __gram_to_text(bigram_sentences, trigram, final_file_path)

def get_model(file_path):
    return Phrases.load(file_path)

def apply_trigram_model(sentence, bigram_model, trigram_model):
        sentence_stream = sentence.split()
        bigram =  bigram_model[sentence_stream]
        return trigram_model[bigram]    

def get_trigram_model_file(sentence, bigram_model, trigram_model, file_path):
        sentence_stream = sentence.split()
        bigram =  bigram_model[sentence_stream]
        trigram = trigram_model[bigram]
        with codecs.open(file_path, 'w', encoding=u'utf_8') as f:
            f.write(u' '.join(trigram))