from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
import numpy as np

class TokenizerCustom(Tokenizer):
    def __init__(self, voc, max_len=15, *args, **kwargs):
        super(TokenizerCustom, self).__init__(*args, **kwargs)
        self.max_len = max_len
        self.word_index = voc
        self.oov_token = '<unk>'
        self.filters = '#$%&()*+-/=@[\]^_`{|}~.,'
    def pad_string(self, x):
        return pad_sequences(x, maxlen=self.max_len)
    
    def tokenize_string(self, x):
        tok_str = self.texts_to_sequences(pd.Series(x).values)
        return self.pad_string(tok_str)[0]


def remap_words_overall(y, times, myid, voc_dic_inv):
    len_shape = len(voc_dic_inv)
    new_index = len_shape
    i = 0
    row, col = np.where(y==myid)
    if times>0:
        for pos in range(times-1):
            row_loop, col_loop = row[i:i+int(len(row)/times)], col[i:i+int(len(row)/times)]
            y[row_loop, col_loop] = new_index
            voc_dic_inv[new_index] = voc_dic_inv[myid]
            new_index += 1
            i += int(len(row)/times)

    return y, voc_dic_inv