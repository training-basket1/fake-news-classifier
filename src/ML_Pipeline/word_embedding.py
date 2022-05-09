from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import one_hot
import pandas as pd
import numpy as np
from ML_Pipeline.constants import *



def extractGlovefile(glove_dir='../input/glove/'):
    glove_dir = resource_dir+'glove/'
    #if ~pathlib.Path(glove_dir).exists():
    os.makedirs(glove_dir,exist_ok=True)
    file_zip = pathlib.Path(glove_dir + "glove.6B.zip")
    if file_zip.exists():
        with ZipFile(glove_dir + 'glove.6B.zip', 'r') as zip:
            # printing all the contents of the zip file
            zip.printdir()
            # extracting all the files
            print('Extracting all the files now...')
            zip.extractall(glove_dir)
            print('Done!')
    else:
        print("glove pretrained model not exists..downloading start..", )

        wget.download('http://nlp.stanford.edu/data/glove.6B.zip', out=glove_dir)
        # opening the zip file in READ mode
        with ZipFile(glove_dir + 'glove.6B.zip', 'r') as zip:
            # printing all the contents of the zip file
            zip.printdir()
            # extracting all the files
            print('Extracting all the files now...')
            zip.extractall(glove_dir)
            print('Done!')



def read_glove_embedings(glove_file_path=glove_file_path):
    word_vec = pd.read_table(glove_file_path, sep=r"\s", header=None, engine='python', encoding='iso-8859-1', error_bad_lines=False)
    word_vec.set_index(0, inplace=True)
    print('Found %s word vectors.' % len(word_vec))
    # print('politics',word_vec.head())
    return word_vec

# Golve embedding use tokenizer for 
# word index, vocab size
def glove_embedings(tokenizer):
    embeddings_index = read_glove_embedings()
    embedding_matrix = np.zeros((vocab_size, emb_dim))

    #embedding_weights = np.zeros((10000, 50))
    index_n_word = [(i, tokenizer.index_word[i]) for i in range(1, len(embedding_matrix)) if
                    tokenizer.index_word[i] in embeddings_index.index]
    idx, word = zip(*index_n_word)
    embedding_matrix[idx, :] = embeddings_index.loc[word, :].values

    return embedding_matrix

def onehot_embedding(tokenizer):
    onehot_vec =  [one_hot(words, (len(tokenizer.word_counts) +1)) for words in tokenizer.word_index.keys()]
    embedded_docs = pad_sequences(onehot_vec, padding='pre', maxlen=max_text_length)
    return embedded_docs

def build_embeddings(tokenizer):
    vocab_len = vocab_size
    print(" vocab_len ", vocab_size)
    
    if embedding_type=='glove':
        embedding_matrix =  glove_embedings(tokenizer)
        # print(" Encoded word sequence:: ",embedding_matrix[0:10])
        embeddingLayer = Embedding(input_dim=vocab_len, output_dim=emb_dim, input_length=max_text_length,
                                   weights=[embedding_matrix], trainable=False)
    # elif embedding_type=='fasttext':
    #     embedding_matrix =  fasttext_embedings()
    #     embeddingLayer = Embedding(input_dim=vocab_len, output_dim=emb_dim, input_length=max_text_length,
    #                                weights=[embedding_matrix], trainable=False)
    else:
        embedding_matrix = onehot_embedding(tokenizer)
        embeddingLayer = Embedding(input_dim=vocab_len, output_dim=emb_dim, input_length=max_text_length,
                                   trainable=False)

    return embeddingLayer