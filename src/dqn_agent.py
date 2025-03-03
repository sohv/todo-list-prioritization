import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Use deque with fixed size
        self.gamma = 0.95  # discount rate (slightly reduced)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995  # slower decay
        self.learning_rate = 0.001  # increased learning rate
        self.model = self._build_model()
        self.target_model = self._build_model()  # Target network
        self.update_target_counter = 0
        self.target_update_freq = 10  # Update target network every 10 episodes
    
    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.state_size,)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))  # Larger network
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        # Copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0][0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3][0] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        # Predict current Q-values and next Q-values
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update Q-values for the actions taken
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        # Return loss for monitoring
        return history.history['loss'][0]