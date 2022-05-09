from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import io
import json
from ML_Pipeline.constants import *





def save_tokenizer(tokenizer,num_words=vocab_size,model_dir='../output/models/', filename=None):
    if filename == None:
        filepath = model_dir + 'tokenizer_'+str(num_words)+'.json'
    else:
        filepath = model_dir + filename
    with io.open(filepath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer.to_json(), ensure_ascii=False))
    f.close()
    print(f"Tokenizer saved in {filename}")
    return

def build_tokenizer(df_train,num_words=vocab_size):
    if num_words is None:
        tokenizer = Tokenizer(oov_token=oov_token)
    else:
        tokenizer = Tokenizer(oov_token=oov_token,num_words=num_words)
    tokenizer.fit_on_texts(df_train)
    word_index = tokenizer.word_index
    print(" Word Index length ", len(word_index))
    print(" Number of Words:  ", tokenizer.num_words)
    # save_tokenizer(tokenizer,num_words)
    return tokenizer, word_index

def prepare_seqence_data(df,tokenizer):
    # Transforms each text in texts to a sequence of integers.
    # Create Sequence
    print(" Create Sequence of tokens ")
    text_sequences = tokenizer.texts_to_sequences(df)

#     # Missing words in Glove vectors
#     words_used = [tokenizer.index_word[i] for i in range(1, vocab_size)]
#     missing_words = set(words_used) - set(word_vec.index.values)
#     print(len(missing_words))
#     missing_word_index = [tokenizer.word_index[word] for word in missing_words]
    
#     # Deleting above missing words
#     text_sequences = [[word for word in sentence if word not in missing_word_index] for sentence in text_sequences]
    print("Text to sequence of Id:: ", text_sequences[0:1])
    return text_sequences

def pad_sequence_data(text_sequences,max_text_length):
    # Pad the Sequences, because the sequences are not of the same length,
    # so let’s pad them to make them of similar length
    text_padded = pad_sequences(text_sequences, maxlen=max_text_length, padding=padding_type,
                                          truncating=trunction_type)
    return text_padded


