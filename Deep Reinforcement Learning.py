import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define the environment (simple grid world)
class GridWorld:
    def __init__(self):
        self.state = 0
        self.n_states = 4
        self.n_actions = 2
        self.transitions = [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 1)]
        self.rewards = [0, 1, 0, 1]
        
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        for s, next_s, reward in self.transitions:
            if s == self.state and action == 1:
                self.state = next_s
                return next_s, self.rewards[next_s], False
        return self.state, 0, True  # No transition, stay in the same state

# Define the policy network
class PolicyNetwork(tf.keras.Model):
    def __init__(self, n_states, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(n_actions, activation='softmax')
    
    def call(self, state):
        x = self.fc1(state)
        return self.fc2(x)

# Initialize environment and policy network
env = GridWorld()
policy_net = PolicyNetwork(env.n_states, env.n_actions)
optimizer = Adam(learning_rate=0.01)

# Training loop
def train_policy_net(env, policy_net, optimizer, n_episodes=1000):
    for episode in range(n_episodes):
        state = env.reset()
        state = np.array([state], dtype=np.float32)
        done = False
        
        while not done:
            state_tf = tf.convert_to_tensor(state, dtype=tf.float32)
            probs = policy_net(state_tf)
            action = np.random.choice(env.n_actions, p=probs.numpy().flatten())
            
            next_state, reward, done = env.step(action)
            next_state = np.array([next_state], dtype=np.float32)
            
            # Compute loss and update policy network
            with tf.GradientTape() as tape:
                probs = policy_net(state_tf)
                log_prob = tf.math.log(probs[0, action])
                loss = -log_prob * reward  # Maximize entropy + reward
            grads = tape.gradient(loss, policy_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))
            
            state = next_state

# Run the training
train_policy_net(env, policy_net, optimizer)

# Test the trained policy
state = env.reset()
done = False
while not done:
    state_tf = tf.convert_to_tensor(np.array([state], dtype=np.float32))
    probs = policy_net(state_tf)
    action = np.argmax(probs.numpy())
    next_state, reward, done = env.step(action)
    print(f"State: {state}, Action: {action}, Reward: {reward}")
    state = next_state
