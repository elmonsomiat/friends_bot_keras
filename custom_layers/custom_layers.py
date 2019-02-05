from keras import backend as K
from keras.layers import Layer, Activation

class Att(Layer):
  '''Attention layer following:
         ---Attention is all you need, A.Vaswani et al.---
  '''

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Att, self).__init__(**kwargs)

    def build(self, input_shape):
        self.q = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1], 1),
                                      initializer='uniform',
                                      trainable=True)
        self.k = self.add_weight(name='kernel', 
                                      shape=(1, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Att, self).build(input_shape)  

    def call(self, x):
        att = K.dot(self.q, self.k)/np.sqrt(self.output_dim)
        att = Activation('softmax')(att)
        x_att = K.dot(x, att)
        return x_att

    def compute_output_shape(self, input_shape):
        return (1, input_shape[0], self.output_dim)