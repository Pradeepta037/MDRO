#  #################################################################
#  This file contains the main DROO operations, including building DNN, 
#  Storing data sample, Training DNN, and generating quantized binary offloading decisions.



from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print(tf.__version__)



# DNN network for memory
# class MemoryDNN:
#     def __init__(
#         self,
#         net,
#         learning_rate=0.01,
#         training_interval=10, 
#         batch_size=50, 
#         memory_size=1000,
#         output_graph=False
#     ):
#         self.net = net  # the size of the DNN architecture
#         self.training_interval = training_interval  # learn every #training_interval
#         self.lr = learning_rate
#         self.batch_size = batch_size
#         self.memory_size = memory_size
        
#         # store all binary actions
#         self.enumerate_actions = []

#         # stored # memory entry
#         self.memory_counter = 1

#         # store training cost
#         self.cost_his = []

#         # initialize zero memory [h, m]
#         self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

#         # construct memory network
#         self._build_net()

#     def _build_net(self):
#         model = keras.Sequential()

#         # Add the input layer
#         model.add(layers.Dense(self.net[0], activation='relu'))

#         # Add hidden layers
#         for units in self.net[1:-1]:
#             model.add(layers.Dense(units, activation='relu'))

#         # Add the output layer
#         model.add(layers.Dense(self.net[-1], activation='sigmoid'))

#         # Compile the model
#         model.compile(optimizer=keras.optimizers.Adam(lr=self.lr),
#                       loss=tf.losses.binary_crossentropy,
#                       metrics=['accuracy'])

#         self.model = model

class MemoryDNN:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval=10, 
        batch_size=50, 
        memory_size=1000,
        output_graph = False
    ):

        self.net = net  # the size of the DNN
        self.training_interval = training_interval      # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size
        
        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 1

        # store training cost
        self.cost_his = []

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # construct memory network
        self._build_net()

    def _build_net(self):
        self.model = keras.Sequential([
                    layers.Dense(self.net[1], activation='relu'),  # the first hidden layer
                    layers.Dense(self.net[2], activation='relu'),  # the second hidden layer
                    layers.Dense(self.net[-1], activation='sigmoid')  # the output layer
                ])

        self.model.compile(optimizer=keras.optimizers.Adam(lr=self.lr), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])


    def remember(self, h, m):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        # encoding the entry
        self.remember(h, m)
        # train the DNN every 10 step
        #if self.memory_counter> self.memory_size / 2 and self.memory_counter % self.training_interval == 0:
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        
        h_train = batch_memory[:, 0: self.net[0]]
        m_train = batch_memory[:, self.net[0]:]
        
        # print(h_train)          # (128, 10)
        # print(m_train)          # (128, 10)

        # train the DNN
        hist = self.model.fit(h_train, m_train, verbose=0)
        self.cost = hist.history['loss'][0]
        assert(self.cost > 0)
        self.cost_his.append(self.cost)

    def decode(self, h, k = 10, mode = 'GA'):
        # to have batch dimension when feed into tf placeholder
        h = h[np.newaxis, :]

        m_pred = self.model.predict(h)

        if mode == 'OP':
            return self.knm(m_pred[0], k)
        elif mode == 'KNN':
            return self.knn(m_pred[0], k)
        elif mode == 'GA':
            return self.genetic_algorithm(m_pred[0], k)
        else:
            print("The action selection must be 'OP' or 'KNN'")

    def genetic_algorithm(self, m, k=10):
        # Genetic algorithm for binary action selection
        pop_size = 100
        num_genes = len(m)
        num_parents = 50
        num_offspring = 50
        num_generations = 100
        mutation_rate = 0.01

        # Initialize population
        population = np.random.randint(2, size=(pop_size, num_genes))

        for _ in range(num_generations):
            fitness = self.evaluate_fitness(population, m)
            parents = self.select_parents(population, fitness, num_parents)
            offspring = self.crossover(parents, num_offspring)
            offspring = self.mutate(offspring, mutation_rate)
            population = np.vstack((population, offspring))

        best_solution = population[np.argmax(fitness)]
        return best_solution


    # def genetic_algorithm(self, m, k=10):
    # # Genetic algorithm for binary action selection
    #     pop_size = 100
    #     num_genes = len(m)
    #     num_parents = 50
    #     num_offspring = 50
    #     num_generations = 100
    #     mutation_rate = 0.01

    # # Initialize population
    #     population = np.random.randint(2, size=(pop_size, num_genes))

    #     for _ in range(num_generations):
    #         fitness = self.evaluate_fitness(population, m)
    #         parents = self.select_parents(population, fitness, num_parents)
    #         offspring = self.crossover(parents, num_offspring)
    #         offspring = self.mutate(offspring, mutation_rate)
    #         population = np.vstack((population, offspring))

    #     best_solution = population[np.argmax(fitness)]

    # # Convert the best solution to the desired format
    #     binary_actions = [int(bit) for bit in best_solution]
    #     return binary_actions


    def evaluate_fitness(self, population, m):
        fitness = np.zeros(len(population))
        for i, ind in enumerate(population):
            fitness[i] = -np.linalg.norm(ind - m)  # Negative Euclidean distance
        return fitness

    def select_parents(self, population, fitness, num_parents):
        selected_parents = []
        for _ in range(num_parents):
            candidates = np.random.choice(len(population), 3, replace=False)
            winner = candidates[np.argmax(fitness[candidates])]
            selected_parents.append(population[winner])
        return np.array(selected_parents)

    def crossover(self, parents, num_offspring):
        offspring = []
        for _ in range(num_offspring):
            crossover_point = np.random.randint(1, len(parents[0]))
            child = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))
            offspring.append(child)
        return np.array(offspring)

    def mutate(self, offspring, mutation_rate):
        for i in range(len(offspring)):
            for j in range(len(offspring[i])):
                if np.random.rand() < mutation_rate:
                    offspring[i][j] = 1 - offspring[i][j]  # Flip the bit
        return offspring
    
    def knm(self, m, k = 10):
        # return k order-preserving binary actions
        m_list = []
        # generate the ﬁrst binary ofﬂoading decision with respect to equation (8)
        m_list.append(1*(m>0.5))
        
        if k > 1:
            # generate the remaining K-1 binary ofﬂoading decisions with respect to equation (9)
            m_abs = abs(m-0.5)
            idx_list = np.argsort(m_abs)[:k-1]
            for i in range(k-1):
                if m[idx_list[i]] >0.5:
                    # set the \hat{x}_{t,(k-1)} to 0
                    m_list.append(1*(m - m[idx_list[i]] > 0))
                else:
                    # set the \hat{x}_{t,(k-1)} to 1
                    m_list.append(1*(m - m[idx_list[i]] >= 0))

        return m_list
    
    def knn(self, m, k = 10):
        # list all 2^N binary offloading actions
        if len(self.enumerate_actions) == 0:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        # the 2-norm
        sqd = ((self.enumerate_actions - m)**2).sum(1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]
        

    def plot_cost(self):
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        mpl.style.use('seaborn')
        fig, ax = plt.subplots(figsize=(15,8))

        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()
