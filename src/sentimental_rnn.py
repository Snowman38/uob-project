import keras
import tensorflow as tf

from keras.layers import Dense, Dropout, BatchNormalization, CuDNNGRU, CuDNNLSTM, Activation

ARCHITECTURES = {
    "DEFAULT": [("LSTM", 128), ("GRU", 128), ("Dense", 32), ("Dense", 4)],
    "DENSE": [("Dense", 256), ("Dense", 64), ("Dense", 16), ("Dense", 4)]
}

class SentimentalRNN(tf.keras.Model):

    def __init__(self, input_shape, architecture="SIMPLE", out_neurons=4, last_activation=None):
        super().__init__()

        self.main = keras.models.Sequential()
        self.main.add(keras.Input(shape=input_shape))
        architecture = ARCHITECTURES[architecture]

        for i in range(len(architecture)):
            for layer in self.make_layers(i, architecture, drop=0.3):
                self.main.add(layer)
        
        self.fc = Dense(out_neurons, activation=last_activation)
        self.bn = BatchNormalization()

    @staticmethod
    def make_layers(i, architecture, drop=0.2):
        name, param = architecture[i]

        rnn_params = {}
        fc_params = {"activation": "relu"}
        layers = []

        if i + 1 < len(architecture):
            return_sequences = architecture[i+1][0] in ["GRU", "LSTM"]
            rnn_params["return_sequences"] = return_sequences            

        if name == "LSTM":
            layers.append(CuDNNLSTM(param, **rnn_params))
        elif name == "GRU":
            layers.append(CuDNNGRU(param, **rnn_params))
        elif name == "Dense":
            layers.append(Dense(param, **fc_params))
            layers.append(BatchNormalization())
        
        #layers.append(BatchNormalization())
        layers.append(Dropout(drop))
        
        return layers

    def call(self, x, **kwargs):
        x = self.main(x)
        out = self.fc(x)

        return self.bn(out)