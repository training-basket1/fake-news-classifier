from ML_Pipeline.utils import read_data
from ML_Pipeline.text_statistics import text_statistics
from ML_Pipeline.clean_data import clean_data
from ML_Pipeline.data_preprocessing import preparing_datasets
from ML_Pipeline.text_tokenizer import save_tokenizer, build_tokenizer, prepare_seqence_data, pad_sequence_data
from ML_Pipeline.constants import *
from ML_Pipeline.word_embedding import build_embeddings
from ML_Pipeline.build_model import build_network_GRU, build_network_RNN, build_network_lstm
from ML_Pipeline.train_model import train_model,store_model
from ML_Pipeline.evaluate_model import performance_history, model_evaluation, performance_report

news_df= read_data("../input/train.csv")
test = read_data('../input/test.csv')
submit_test = read_data('../input/submit.csv')
test['label'] = submit_test.label

text_stats = text_statistics(news_df,'text')
title_stats = text_statistics(news_df,'title')

df = news_df.copy()

## data cleaning
df = clean_data(df, remove_column_names=remove_columns)
df_test = clean_data(test, remove_column_names=remove_columns)

## preprocessing datasets
print(" Training data preprocessing ")
X,y = preparing_datasets(df, text_features=text_features)
print(" Test data preprocessing ")
X_test,y_test = preparing_datasets(df_test,text_features=text_features)
X_train,y_train = X,y

## tokenization
tokenizer, word_index = build_tokenizer(X_train)
save_tokenizer(tokenizer)

train_text_seq = prepare_seqence_data(X_train,tokenizer)
test_text_seq = prepare_seqence_data(X_test,tokenizer)

train_text_padded = pad_sequence_data(train_text_seq,max_text_length)
test_text_padded = pad_sequence_data(test_text_seq,max_text_length)

print("Padded Sequence :: ", test_text_padded[0:1])
print(" Tokenizer detail :: ", tokenizer.document_count)
print('Vocabulary size:', len(tokenizer.word_counts))
print('Shape of data padded:', train_text_padded.shape)

## embedding
embeding_layer = build_embeddings(tokenizer)

## model building

# model_rnn = build_network_RNN(embeding_layer)
# model_rnn,history = train_model(model_rnn,train_text_padded,y_train,test_text_padded, y_test)
# performance_history(history)
# store_model(model_rnn,file_name='rnn_1')
# score = model_evaluation(model_rnn,test_text_padded,y_test)
# total_cost_df = performance_report(model_rnn,test_text_padded,y_test,'rnn_1')

# model_lstm = build_network_lstm(embeding_layer,lstm_size)
# model_lstm,history_lstm = train_model(model_lstm,train_text_padded,y_train,test_text_padded, y_test)
# performance_history(history_lstm)
# store_model(model_lstm,file_name='lstm_1')
# score = model_evaluation(model_lstm,test_text_padded,y_test)
# total_cost_df = performance_report(model_lstm,test_text_padded,y_test,'lstm_1')


model_gru = build_network_GRU(embeding_layer)
model_gru,history_gru = train_model(model_gru,train_text_padded,y_train,test_text_padded, y_test)
performance_history(history_gru)
store_model(model_gru,file_name='gru_1')
score = model_evaluation(model_gru,test_text_padded,y_test)
total_cost_df = performance_report(model_gru,test_text_padded,y_test,'gru_1')