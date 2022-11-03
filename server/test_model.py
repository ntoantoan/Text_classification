import tensorflow as tf
import re
import os
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pickle
from vncorenlp import VnCoreNLP
import underthesea
import matplotlib.pyplot as plt



def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\,\?]+$-()!*=._", "", row)
    row = row.replace(",", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("*", " ")\
        .replace("=", " ").replace("(", " ")\
        .replace(")", " ").replace("_", " ").replace(".", " ")
    row = row.strip().lower()
    return row

def get_data_line_decode_token(v_token, max_len, max_line):
    data_line = []
    data_token = []
    cnt_line = 0
    cnt_code = 0
    for code in v_token:
        if cnt_line == max_line:
            return data_line
        
        if cnt_code == max_len:
            data_line.append(data_token)
            cnt_line += 1
            cnt_code = 0
            data_token = []

        data_token.append(code)
        cnt_code += 1
    if len(data_line) < max_line:
        data_line.append(data_token)
    if len(data_line) < max_line:
        for i in range(0, max_line - len(data_line)):
            data_line.append([])
    return data_line

def get_pho_bear_feature(v_pho_bert, data_token, max_len_word):
    v_tokenized = []
    v_tokenized.append(data_token)
    padded = np.array([i + [1] * (max_len_word - len(i)) for i in v_tokenized])

    # Đánh dấu các từ thêm vào = 0 để không tính vào quá trình lấy features
    attention_mask = np.where(padded == 1, 0, 1)

    # Chuyển thành tensor
    padded = torch.tensor(padded).to(torch.long)
    attention_mask = torch.tensor(attention_mask)

    # Lấy features dầu ra từ BERT
    with torch.no_grad():
        last_hidden_states = v_pho_bert(input_ids=padded, attention_mask=attention_mask)

    # print(last_hidden_states)
    # print(last_hidden_states[0][:, 0, :].numpy().shape)
    v_features = last_hidden_states[0][:, 0, :].numpy().T
    # print(v_features.shape)

    return v_features

def get_text_feature(sentence, v_pho_bert, v_tokenizer, max_line, max_len_word):
    data_feature = []
    word_segmented_text = underthesea.word_tokenize(sentence)
    
    line = " ".join(word_segmented_text)
    line = underthesea.word_tokenize(line, format="text")
    v_token = v_tokenizer.encode(line)


    data_line_token = get_data_line_decode_token(v_token, max_len_word, max_line)


    for data_token in data_line_token:
        feature_token = get_pho_bear_feature(v_pho_bert, data_token, max_len_word)
        data_feature.append(feature_token)
    
    # print(np.array(data_feature).shape)

    data_feature = np.array(data_feature)
    
    return data_feature



def load_pho_bert():
    pho_bert = AutoModel.from_pretrained("vinai/phobert-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    return pho_bert, tokenizer

pho_bert, v_token = load_pho_bert()

max_len_word = 100
cnt_post = 1000
line = 3
output_shape = (3, 768, 1)

text = "Công an Nam Định thành lập Phòng An ninh mạng và phòng, chống tội phạm sử dụng công nghệ cao"
text_post = standardize_data(text)
v_feat = get_text_feature(text_post, pho_bert, v_token, line, max_len_word)

print(v_feat.shape)
v_feat = v_feat.reshape(1,3,768,1)


model = tf.keras.models.load_model('cnn.h5')

y_predict = model.predict(v_feat)
y_target = np.argmax(y_predict, axis=1)

print(y_predict)
print(y_target)