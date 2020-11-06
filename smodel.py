import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences


class sModel:

    def __init__(self):
        self.model = None
        self.xs, self.ys, self.ins, self.outs, self.outs_cln = None, None, None, None, None
        self.index2word, self.word2index, self.tokenizer = None, None, None
        self._load_data()
        self._vectorized()
        self._model2()
        self.status = 'loaded'

    def train(self):
        self.model.fit(self.xs, self.ys, epochs=10)

    def prediction(self, sentence):
        vec = self._to_vec(sentence)
        ret = self.model.predict(vec)

        return ret

    def _model1(self):
        self.model = Sequential([keras.layers.Dense(units=10, input_shape=[1, 20])])
        self.model.compile(optimizer='sgd', loss='mean_squared_error')

    def _model2(self):
        self.model = Sequential([keras.layers.Flatten(input_shape=(1, 20)),
                                 keras.layers.Dense(128, activation=tf.nn.relu),
                                 keras.layers.Dense(20, activation=tf.nn.softmax)])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy')



    def _load_data(self):
        self.index2word = pickle.load(open('index2word_dict.pickle', 'rb'))
        self.word2index = pickle.load(open('word2index_dict.pickle', 'rb'))
        self.tokenizer = pickle.load(open('tokenizer_object.pickle', 'rb'))
        self.ins = pickle.load(open('encoder_inputs.pickle', 'rb'))
        self.outs = pickle.load(open('decoder_inputs.pickle', 'rb'))
        self.outs_cln = [x[6:-6] for x in self.outs]

    def _vectorized(self):
        self.xs = self._to_vec(self.ins)
        self.ys = self._to_vec(self.outs_cln)

    def _to_vec(self, text):
        tok = self.tokenizer.texts_to_sequences([text])
        pad = pad_sequences(tok, maxlen=20, dtype='int32', padding='post', truncating='post')
        """
        retval = []
        for i, val in enumerate(pad):
            retval += val
            if i > 100:
                break
        """
        return pad


if __name__ == "__main__":
    sm = sModel()
    sm.train()
    sm.prediction("Hi there")
    print("done")
