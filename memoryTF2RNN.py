#  #################################################################
#  This file contains the main RROO operations, including building RNN, 
#  Storing data sample, Training RNN, and generating quantized binary offloading decisions.


from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print(tf.__version__)

# RNN network for memory
class MemoryRNN:
    def __init__(
        self,
        num_units,
        learning_rate=0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False
    ):
        self.num_units = num_units
        self.training_interval = training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        
        self.enumerate_actions = []
        self.memory_counter = 1
        self.cost_his = []

        # initialize memory
        self.memory = np.zeros((self.memory_size, self.num_units))

        # build the RNN model
        self._build_net()

    def _build_net(self):
        self.model = keras.Sequential([
            layers.LSTM(self.num_units, return_sequences=True),
            layers.LSTM(self.num_units),
            layers.Dense(self.num_units, activation='relu'),
            layers.Dense(self.num_units, activation='sigmoid')
        ])

        self.model.compile(optimizer=keras.optimizers.Adam(lr=self.lr), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])

    def remember(self, h, m):
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = h

        self.memory_counter += 1

    def encode(self, h, m):
        self.remember(h, m)
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = batch_memory[:, :]
        m_train = batch_memory[:, :]

        hist = self.model.fit(h_train, m_train, verbose=0)
        self.cost = hist.history['loss'][0]
        assert(self.cost > 0)
        self.cost_his.append(self.cost)

    def decode(self, h, k=1, mode='OP'):
        h = h[np.newaxis, :]

        m_pred = self.model.predict(h)

        if mode == 'OP':
            return self.knm(m_pred[0], k)
        elif mode == 'KNN':
            return self.knn(m_pred[0], k)
        else:
            print("The action selection must be 'OP' or 'KNN'")
    
    def knm(self, m, k=1):
        m_list = []
        m_list.append(1*(m > 0.5))
        
        if k > 1:
            m_abs = abs(m - 0.5)
            idx_list = np.argsort(m_abs)[:k-1]
            for i in range(k-1):
                if m[idx_list[i]] > 0.5:
                    m_list.append(1*(m - m[idx_list[i]] > 0))
                else:
                    m_list.append(1*(m - m[idx_list[i]] >= 0))

        return m_list
    
    def knn(self, m, k=1):
        if len(self.enumerate_actions) == 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.num_units))))

        sqd = ((self.enumerate_actions - m)**2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]
        
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()
