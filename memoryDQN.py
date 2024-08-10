from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from collections import deque
import random

print(tf.__version__)

class DQNDNN:
    def __init__(
        self,
        net,
        learning_rate=0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        gamma=0.99,
        target_update_interval=100,
        output_graph=False
    ):
        self.net = net
        self.training_interval = training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.gamma = gamma
        self.target_update_interval = target_update_interval

        self.memory_counter = 1
        self.memory = deque(maxlen=self.memory_size)
        self.cost_his = []

    def build_dqn(self):
        # Create a Sequential model
        self.model = keras.models.Sequential()

        # Add layers to the model
        for layer in self.net:
            self.model.add(keras.layers.Dense(layer, activation='relu'))

        # Add a final layer with linear activation
            self.model.add(keras.layers.Dense(1, activation='linear'))

        # Compile the model
            self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                        loss='mse',
                        metrics=['accuracy'])
        self.build_dqn()

    # def build_dqn(self):
    #     self.model = keras.Sequential([
    #         keras.layers.Dense(120, activation="relu"),
    #         keras.layers.Dense(80, activation="relu"),
    #         keras.layers.Dense(10, activation="linear") # Linear activation for Q-values
    #     ])

    #     self.model.compile(optimizer=keras.optimizers.Adam(lr=self.lr), loss='mse')

        # Target network for stability
        # self.target_model = keras.models.clone_model(self.model)
        # self.target_model.set_weights(self.model.get_weights())

        # self.model.build(input_shape=(1, 10))
    
        # self.model(tf.random.normal((1, 10)))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.memory_counter += 1

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(next_state)[0])
                target[0][action] = reward + self.gamma * Q_future
            states.append(state)
            targets.append(target)

        states = np.vstack(states)
        targets = np.vstack(targets)

        hist = self.model.fit(states, targets, epochs=1, verbose=0)
        self.cost_his.append(hist.history['loss'][0])

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def encode(self, state):
        # Use the DQN to make predictions
        return self.model.predict(state)

    def decode(self, state, k=1, mode='OP'):
        q_values = self.encode(state)

        if mode == 'OP':
            return self.knm(q_values[0], k)
        elif mode == 'KNN':
            return self.knn(q_values[0], k)
        else:
            print("The action selection must be 'OP' or 'KNN'")

    def knm(self, q_values, k=1):
        # Use Q-values for order-preserving binary actions
        m_list = [1 if q_value > 0 else 0 for q_value in q_values]
        return [m_list]

    def knn(self, q_values, k=1):
        # Use Q-values for k-nearest neighbors binary offloading actions
        idx = np.argsort(q_values)
        return [list(map(int, format(action, '0' + str(self.net[0]) + 'b'))) for action in idx[:k]]

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()

