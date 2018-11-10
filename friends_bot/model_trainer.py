from keras import regularizers
from keras.layers import multiply
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.metrics import sparse_top_k_categorical_accuracy
from keras.losses import sparse_categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Embedding, Input, Dense, LSTM, Dropout, RepeatVector
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping



def define_model(max_seq_len, vocab_dim, vocab_out_dim, emb_len=100, lstm_units=200, dropout=0.2, reg=0, attention=False, simple_attention=True)
    inp = Input(shape=(max_seq_len,))
    emb = Embedding(vocab_dim, emb_len, weights=[glove_unk_friends_df], 
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
        rep_vec_2 = Dense(vocab_out_dim, activation='tanh', W_regularizer=regularizers.l2(reg))(lstm_in_2)
        rep_vec = multiply([rep_vec, rep_vec_2])

    lstm_out = LSTM(lstm_units, dropout=dropout, return_sequences=True)(rep_vec)
    # lstm_out = LSTM(lstm_units, dropout=0, return_sequences=True)(lstm_out)

    out = TimeDistributed(Dense(vocab_out_dim, activation='softmax', W_regularizer=regularizers.l2(reg)))(lstm_out)
    return out

def compile_model(inp, out, metrics=['accuracy']):
    earlystop = EarlyStopping(monitor='acc', min_delta=0.00001, patience=5)
    reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.4, patience=5, min_lr=0.0005, verbose=1)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=metrics)