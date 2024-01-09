import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Dense, Attention
from tensorflow.keras.models import Model


class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Encoder, self).__init__()
        # Embedding Layer（params: input_dim（词典长度）, output_dim（嵌入词向量维度）, input_length（输入语句长度））
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        # Encode LSTM Layer（params: units（输出维度）, input_dim（输入维度）, return_sequences（返回控制类型，True时返回整个序列，否则仅返回输出序列的最后一个输出））
        self.encoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, name="encode_lstm")
        
    def call(self, inputs):
        encoder_embed = self.embedding(inputs)
        encoder_outputs, state_h, state_c = self.encoder_lstm(encoder_embed)
        return encoder_outputs, state_h, state_c


class Decoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Decoder, self).__init__()
        # Embedding Layer
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        # Decode LSTM Layer
        self.decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, name="decode_lstm")
        # Attention Layer
        self.attention = Attention()
    
    def call(self, enc_outputs, dec_inputs, states_inputs):
        decoder_embed = self.embedding(dec_inputs)
        dec_outputs, dec_state_h, dec_state_c = self.decoder_lstm(decoder_embed, initial_state=states_inputs)
        attention_output = self.attention([dec_outputs, enc_outputs])
        
        return attention_output, dec_state_h, dec_state_c


def Seq2Seq(maxlen, embedding_dim, hidden_units, vocab_size):
    """
    seq2seq model
    """
    # Input Layer
    encoder_inputs = Input(shape=(maxlen,), name="encode_input")
    decoder_inputs = Input(shape=(None,), name="decode_input")
    # Encoder Layer
    encoder = Encoder(vocab_size, embedding_dim, hidden_units)  # 初始化encoder
    enc_outputs, enc_state_h, enc_state_c = encoder(encoder_inputs)  # 前向传播
    # decoder接收的状态输入数组[enc_state_h, enc_state_c]
    dec_states_inputs = [enc_state_h, enc_state_c]
    # Decoder Layer
    decoder = Decoder(vocab_size, embedding_dim, hidden_units)  # 初始化decoder
    attention_output, dec_state_h, dec_state_c = decoder(enc_outputs, decoder_inputs, dec_states_inputs)  # 前向传递
    # Dense Layer（全连接层）
    dense_outputs = Dense(vocab_size, activation='softmax', name="dense")(attention_output)
    # seq2seq model
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=dense_outputs)
    
    return model


def read_vocab(vocab_path):
    vocab_words = []
    with open(vocab_path, "r", encoding="utf8") as f:
        for line in f:
            vocab_words.append(line.strip())
    return vocab_words


def read_data(data_path):
    datas = []
    with open(data_path, "r", encoding="utf8") as f:
        for line in f:
            words = line.strip().split()
            datas.append(words)
    return datas


def process_data_index(datas, vocab2id):
    data_indexs = []
    for words in datas:
        line_index = [vocab2id[w] if w in vocab2id else vocab2id["<UNK>"] for w in words]
        data_indexs.append(line_index)
    return data_indexs


vocab_words = read_vocab("data/vocab.txt")
special_words = ["<PAD>", "<UNK>", "<GO>", "<EOS>"]
vocab_words = special_words + vocab_words
vocab2id = {word: i for i, word in enumerate(vocab_words)}
id2vocab = {i: word for i, word in enumerate(vocab_words)}

num_sample = 50000
source_data = read_data("data/source_min.txt")[:num_sample]
source_data_ids = process_data_index(source_data, vocab2id)
target_data = read_data("data/target_min.txt")[:num_sample]
target_data_ids = process_data_index(target_data, vocab2id)

print("vocab test: ", [id2vocab[i] for i in range(10)])
print("source test: ", source_data[10])
print("source index: ", source_data_ids[10])
print("target test: ", target_data[10])
print("target index: ", target_data_ids[10])


def process_input_data(source_data_ids, target_indexs, vocab2id):
    source_inputs = []
    decoder_inputs, decoder_outputs = [], []
    for source, target in zip(source_data_ids, target_indexs):
        source_inputs.append([vocab2id["<GO>"]] + source + [vocab2id["<EOS>"]])
        decoder_inputs.append([vocab2id["<GO>"]] + target)
        decoder_outputs.append(target + [vocab2id["<EOS>"]])
    return source_inputs, decoder_inputs, decoder_outputs

source_input_ids, target_input_ids, target_output_ids = process_input_data(source_data_ids, target_data_ids, vocab2id)
print("encoder inputs: ", source_input_ids[:2])
print("decoder inputs: ", target_input_ids[:2])
print("decoder outputs: ", target_output_ids[:2])


maxlen = 15
source_input_ids = keras.preprocessing.sequence.pad_sequences(source_input_ids, padding='post', maxlen=maxlen)
target_input_ids = keras.preprocessing.sequence.pad_sequences(target_input_ids, padding='post',  maxlen=maxlen)
target_output_ids = keras.preprocessing.sequence.pad_sequences(target_output_ids, padding='post',  maxlen=maxlen)
print(source_data_ids[:5])
print(target_input_ids[:5])
print(target_output_ids[:5])


K.clear_session()

maxlen = 15
embedding_dim = 50
hidden_units = 128
vocab_size = len(vocab2id)

model = Seq2Seq(maxlen, embedding_dim, hidden_units, vocab_size)
model.summary()


# 训练模型

epochs = 5
batch_size = 32
val_rate = 0.2

loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss_fn, optimizer='adam')
model.fit([source_input_ids, target_input_ids], target_output_ids, 
          batch_size=batch_size, epochs=epochs, validation_split=val_rate)


# 保存模型

model.save_weights("data/seq2seq_attention_weights.h5")
del model


# 载入模型

K.clear_session()

model = Seq2Seq(maxlen, embedding_dim, hidden_units, vocab_size)
model.load_weights("data/seq2seq_attention_weights.h5")
print(model.summary())


def encoder_infer(model):
    encoder_model = Model(inputs=model.get_layer('encoder').input, 
                        outputs=model.get_layer('encoder').output)
    return encoder_model

encoder_model = encoder_infer(model)
print(encoder_model.summary())


def decoder_infer(model, encoder_model):
    encoder_output = encoder_model.get_layer('encoder').output[0]
    maxlen, hidden_units = encoder_output.shape[1:]
    
    dec_input = model.get_layer('decode_input').input
    enc_output = Input(shape=(maxlen, hidden_units), name='enc_output')
    dec_input_state_h = Input(shape=(hidden_units,), name='input_state_h')
    dec_input_state_c = Input(shape=(hidden_units,), name='input_state_c')
    dec_input_states = [dec_input_state_h, dec_input_state_c]

    decoder = model.get_layer('decoder')
    dec_outputs, out_state_h, out_state_c = decoder(enc_output, dec_input, dec_input_states)
    dec_output_states = [out_state_h, out_state_c]

    decoder_dense = model.get_layer('dense')
    dense_output = decoder_dense(dec_outputs)

    decoder_model = Model(inputs=[enc_output, dec_input, dec_input_states], 
                          outputs=[dense_output]+dec_output_states)
    return decoder_model

decoder_model = decoder_infer(model, encoder_model)
print(decoder_model.summary())


import numpy as np

maxlen = 10

def infer_predict(input_text, encoder_model, decoder_model):
    text_words = input_text.split()[:maxlen]
    input_id = [vocab2id[w] if w in vocab2id else vocab2id["<UNK>"] for w in text_words]
    input_id = [vocab2id["<GO>"]] + input_id + [vocab2id["<EOS>"]]
    if len(input_id) < maxlen:
        input_id = input_id + [vocab2id["<PAD>"]] * (maxlen-len(input_id))

    input_source = np.array([input_id])
    input_target = np.array([vocab2id["<GO>"]])
    
    # 编码器encoder预测输出
    enc_outputs, enc_state_h, enc_state_c = encoder_model.predict([input_source])
    dec_inputs = input_target
    dec_states_inputs = [enc_state_h, enc_state_c]

    result_id = []
    result_text = []
    for i in range(maxlen):
        # 解码器decoder预测输出
        dense_outputs, dec_state_h, dec_state_c = decoder_model.predict([enc_outputs, dec_inputs]+dec_states_inputs)
        pred_id = np.argmax(dense_outputs[0][0])
        result_id.append(pred_id)
        result_text.append(id2vocab[pred_id])
        if id2vocab[pred_id] == "<EOS>":
            break
        dec_inputs = np.array([[pred_id]])
        dec_states_inputs = [dec_state_h, dec_state_c]
    return result_id, result_text


input_text = "你 在 干 什么 呢"
result_id, result_text = infer_predict(input_text, encoder_model, decoder_model)

print("Input: ", input_text)
print("Output: ", result_text, result_id)

