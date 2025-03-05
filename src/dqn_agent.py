import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # Increased memory size
        self.priority_memory = deque(maxlen=1000)  # Special buffer for high-reward experiences
        self.gamma = 0.99  # Increased discount factor
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997  # Slower decay
        self.learning_rate = 0.0005  # Reduced learning rate
        self.tau = 0.001  # Soft update parameter
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        model = tf.keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='huber'  # More robust than MSE
        )
        return model
    
    def update_target_model(self):
        # Soft update target model
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        # Store high-reward experiences in priority buffer
        if reward > np.mean([x[2] for x in list(self.memory)[-100:]]):
            self.priority_memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        # Mix regular and priority experiences
        n_priority = batch_size // 4  # 25% priority experiences
        n_regular = batch_size - n_priority
        
        minibatch = random.sample(self.memory, n_regular)
        if self.priority_memory and n_priority > 0:
            minibatch.extend(random.sample(self.priority_memory, 
                                         min(n_priority, len(self.priority_memory))))
        
        states = np.array([i[0][0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3][0] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        # Double DQN update
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        targets = self.model.predict(states, verbose=0)
        
        for i in range(len(minibatch)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * next_q_values[i][next_actions[i]]
        
        # Train with gradient clipping
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        self.model.compile(optimizer=optimizer, loss='huber')
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=32)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Soft update target network
        self.update_target_model()
        
        return history.history['loss'][0]