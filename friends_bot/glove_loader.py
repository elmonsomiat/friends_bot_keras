import numpy as np
import pandas as pd

class GloveLoader(object):

    def load_glove_model(self, glove_file):
        print("Loading Glove Model")
        f = open(glove_file,'r')
        model = {}
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            model[word] = embedding
        print( "Done.",len(model)," words loaded!")
        return model

    def glove_to_df(self, arr):
        glove_df = pd.DataFrame(arr).T
        return glove_df

    def append_glove_pad_unk(self, glove_df, pad, unk, begin, end):
        unk_pad_df = pd.DataFrame(columns=glove_df.columns)
        if pad:
            unk_pad_df.loc['<PAD>'] = np.zeros(glove_df.shape[1])
        if unk:
            unk_pad_df.loc['<UNK>'] = glove_df.mean()
        if begin:
            unk_pad_df.loc['<BEGIN>'] = np.ones(glove_df.shape[1])
        if end:    
            unk_pad_df.loc['<END>'] = -np.ones(glove_df.shape[1])
        glove_unk_df = pd.concat([unk_pad_df,glove_df])
        return glove_unk_df


    def run_glove_loader(self, glove_file, pad=True, unk=True, begin=True, end=True):
        model = self.load_glove_model(glove_file)
        glove_df = self.glove_to_df(model)
        glove_df = self.append_glove_pad_unk(glove_df, pad, unk, begin, end)
        return glove_df