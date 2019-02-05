import re

from friends_bot.cleaner_utils import cleaner_dict

class DialogueCleaner(object):
    """docstring for DialogueCleaner"""
    def __init__(self, words_replace_dict=None):
        if words_replace_dict is None:
            self.words_replace_dict = cleaner_dict
        else:
            self.words_replace_dict = words_replace_dict


    def delete_blank(self, x):
        if x=='':
            return None
        else:
            return x


    def separate_punctuation(self, x):
        x = re.sub(r'\.',' ', x)
        x = re.sub(r'\,',' ', x)
        x = re.sub(r'\!',' ', x)
        x = re.sub(r'\?',' ', x)
        return x

    def delete_large_spaces(self, x):
        return re.sub(r'\s{2,}', ' ', x)

    def change_words(self, x):
        for i in self.words_replace_dict:
            x = re.sub(i, words_replace_dict[i], x)
        return x
    
    def delete_no_dialogue(self, x):
        if ':' in x:
            return x

    def delete_names(self, x):
        x = re.sub(r'[a-z]{2,}:','', x)
        return x

    def delete_semicol(self, x):
        x = re.sub(r'\:','', x)
        return x

    def run_dialogue_cleaner(self, dialogue):
        x = dialogue.astype(str)
        x = x.map(self.delete_blank)
        x = x.str.lower()
        x = x.map(self.delete_large_spaces)
        x = x.map(self.change_words)
        x = x.map(self.delete_no_dialogue)
        x = x.dropna()
        x = x.map(self.delete_names)
        x = x.map(self.delete_semicol)
        x = x.dropna().reset_index(drop=True)
        return x