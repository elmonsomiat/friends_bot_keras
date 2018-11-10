from keras import regularizers
from keras.layers import multiply
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.metrics import sparse_top_k_categorical_accuracy
from keras.losses import sparse_categorical_crossentropy
from keras.models import Model
from keras.layers import Embedding, Input, Dense, LSTM, Dropout, RepeatVector
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers.wrappers import TimeDistributed



def define_model(max_seq_len, vocab_dim, vocab_out_dim, emb_weights, emb_len=100, lstm_units=200, dropout=0.2, reg=0, attention=False, simple_attention=True):
    inp = Input(shape=(max_seq_len,))
    emb = Embedding(vocab_dim, emb_len, weights=[emb_weights], 
                    input_length=max_seq_len, trainable=False, mask_zero=True)(inp)
    # lstm_in = LSTM(200, dropout=0, return_sequences=True)(emb)
    lstm_in = LSTM(lstm_units, dropout=dropout)(emb)

    rep_vec = RepeatVector(max_seq_len)(lstm_in)
    rep_vec = TimeDistributed(Dense(vocab_out_dim, activation='relu', W_regularizer=regularizers.l2(reg)))(rep_vec)

    if attention:
        lstm_in_2 = LSTM(lstm_units, dropout=dropout)(emb)
        rep_vec_2 = RepeatVector(max_seq_len)(lstm_in)
        rep_vec_2 = TimeDistributed(Dense(vocab_out_dim, activation='tanh', W_regularizer=regularizers.l2(reg)))(rep_vec_2)
        rep_vec = multiply([rep_vec, rep_vec_2])

    elif simple_attention:
        lstm_in_2 = LSTM(lstm_units, dropout=dropout)(emb)
        rep_vec_2 = Dense(vocab_out_dim, activation='tanh', W_regularizer=regularizers.l2(reg))(lstm_in_2)
        rep_vec = multiply([rep_vec, rep_vec_2])

    lstm_out = LSTM(lstm_units, dropout=dropout, return_sequences=True)(rep_vec)
    # lstm_out = LSTM(lstm_units, dropout=0, return_sequences=True)(lstm_out)

    out = TimeDistributed(Dense(vocab_out_dim, activation='softmax', W_regularizer=regularizers.l2(reg)))(lstm_out)
    return inp, out

def compile_model(inp, out, metrics=['accuracy']):
    model = Model(inputs=inp, outputs=out)
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(lr=0.001), metrics=metrics)
    return model