import re

class DialogueCleaner(object):
    """docstring for DialogueCleaner"""
    def __init__(self, words_replace_dict=None):
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
        if self.words_replace_dict:
            for i in self.words_replace_dict:
                x = re.sub(i, words_replace_dict[i], x)
            return x
        x = re.sub("let's", 'let us', x)
        x = re.sub("let’s", 'let us', x)
        x = re.sub("c'mon", 'come on', x)
        x = re.sub("c’mon", 'come on', x)
        x = re.sub("there’s", 'there is', x)
        x = re.sub("there's", 'there is', x)
        x = re.sub("you're", 'you are', x)
        x = re.sub("you’re", 'you are', x)
        x = re.sub("we're", 'we are', x)
        x = re.sub("we’re", 'we are', x)
        x = re.sub("i'm", 'i am', x)
        x = re.sub("i’m", 'i am', x)
        x = re.sub("y'", 'you', x)
        x = re.sub("y’", 'you', x)
        x = re.sub("how'd", 'how did', x)
        x = re.sub("how’d", 'how did', x)
        x = re.sub("\'ll", ' will', x)
        x = re.sub("\’t", ' not', x)
        x = re.sub("\'t", ' not', x)
        x = re.sub("\'s", '  is', x)
        x = re.sub("\’s", '  is', x)
        x = re.sub("\'re", '  are', x)
        x = re.sub("\’re", '  are', x)
        x = re.sub("\'", ' ', x)
        x = re.sub('\"', ' ', x)
        x = re.sub('-', ' ', x)
        x = re.sub('pheebs', 'phoebe', x)
        x = re.sub('wasn', 'was not', x)
        x = re.sub('noo', 'no', x)
        x = re.sub("didn", 'did', x)
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