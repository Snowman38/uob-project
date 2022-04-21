import keras
import tensorflow as tf

from keras.layers import Dense, Dropout, BatchNormalization, CuDNNGRU, CuDNNLSTM, Activation
from .sentimental_rnn import SentimentalRNN

ARCHITECTURES = {
    "DEFAULT": [("LSTM", 128), ("GRU", 128), ("Dense", 32)]
}

class BitcoinRNN(tf.keras.Model):
    def __init__(self, in_shape_main, in_shape_sent=None, **kwargs):
        super().__init__()

        out_neurons = kwargs.pop("out_neurons", 1)
        out_neurons_sent = kwargs.pop("out_neurons_sent", 4)
        architecture = kwargs.pop("architecture", "DEFAULT")
        self.sent_arch = kwargs.pop("sent_arch", "DENSE")
        last_activation = kwargs.pop("last_activation", None)
        self.use_sentimental = kwargs.pop("use_sentimental", True)
        self.look_back = kwargs.pop("look_back", 30)

        architecture = ARCHITECTURES[architecture]
        in_shape = in_shape_main

        if self.use_sentimental:
            in_shape = (in_shape_main[0], in_shape_main[1] + out_neurons_sent * 2)
            self.search_rnn = SentimentalRNN(in_shape_sent, out_neurons=out_neurons_sent, architecture=self.sent_arch)
            self.similarity_rnn = SentimentalRNN(in_shape_sent, out_neurons=out_neurons_sent, architecture=self.sent_arch)

        self.main = keras.models.Sequential()
        self.main.add(keras.Input(shape=in_shape))

        for i in range(len(architecture)):
            for layer in SentimentalRNN.make_layers(i, architecture):
                self.main.add(layer)
        
        self.fc = Dense(out_neurons, activation=last_activation)
        

    def call(self, x, **kwargs):
        if self.use_sentimental:
            x_searches = x["x_searches"]
            x_similarities = x["x_similarities"]

            x_search = []
            x_similarity = []

            for i in range(len(x_searches[0])):
                similarity = x_similarities[:, i]
                search = x_searches[:, i]

                bs = len(similarity)

                similarity_seq = []
                search_seq = []

                similarity = tf.reshape(similarity, (bs * self.look_back, similarity.shape[2]))
                search = tf.reshape(search, (bs * self.look_back, search.shape[2]))

                if self.sent_arch in ["SIMPLE", "DEFAULT"]:
                    similarity = similarity[:, None, :]
                    search = search[:, None, :]

                similarity = self.similarity_rnn(similarity)
                search = self.search_rnn(search)

                similarity = tf.reshape(similarity, (bs, self.look_back, similarity.shape[1]))
                search = tf.reshape(search, (bs, self.look_back, search.shape[1]))

                similarity_seq.append(similarity)
                search_seq.append(search)

                # for j in range(len(similarity[0])):
                #   similarity_seq.append(self.similarity_rnn(similarity[:, j]))
                #   search_seq.append(self.search_rnn(search[:, j]))
                
                x_similarity.append(similarity_seq)
                x_search.append(search_seq) 

            x_search = tf.reduce_mean(x_search, axis=0)[0]
            x_similarity = tf.reduce_mean(x_similarity, axis=0)[0]

            #x_search = tf.transpose(x_search, [1, 0, 2])
            #x_similarity = tf.transpose(x_similarity, [1, 0, 2])

            x = tf.concat([x["x_bitcoin"], x_similarity, x_search], axis=2)
            print(x.shape)
        else:
            x = x["x_bitcoin"]
        
        x = self.main(x)
        out = self.fc(x)

        return out