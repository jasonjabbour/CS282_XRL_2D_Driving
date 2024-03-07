import gymnasium as gym
import pandas as pd 
import argparse
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.optimizers import Adam
from collections import deque
from scipy.special import softmax
from helper import target_train_helper, save_collected_data, collect_data, plot_collected_data



class DeepQNetworkAgent():
    # To implement the algorithm in the nalysis of Explainable Goal-Driven Reinforcement Learning
    # in a Continuous Simulated Environment, you need to implement the DQN model yourself
    # then make modificaiton to the DQN class. In order to learn how to create a DQN model from scratch, 
    # I used the following source:
    # Source: https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
    # This source was independent from the authors of the paper. 

    def __init__(self, env):
        # OpenAI Gym Environment
        self._env = env
        # Initialize Memory Queue
        self._memory = deque(maxlen=1000)

        # Gamma
        self._gamma = .95
        # Fraction of time we will dedicate to exploring
        # Start by lots of exploration at the beginning
        self._epsilon = 1.0
        # Smallest epsilon allowed after decaying (lots of exploitation)
        self._epsilon_min = 0.01
        # Decay epsilon by this amount
        self._epsilon_decay = 0.9999
        # Step size 
        self._learning_rate = 0.001
        # Batch Size
        self._batch_size = 4
        # Tau
        self._tau = .125

        # Initialize Q Function
        self._model = self.create_DQN_model()
        # Initialize Q Function Objective
        self._target_model = self.create_DQN_model()
        # Initialize P Function
        self._P_model = self.create_P_DQN_model()
        # Initialize P Function Objective
        self._P_target_model = self.create_P_DQN_model()

    def create_DQN_model(self):

        # Get shape of observation space
        observation_shape = self._env.observation_space.shape
        action_shape = self._env.action_space.shape
    
        # Find the shape of the action space
        if len(action_shape) == 0:
            action_shape = 1
        else:
            action_shape = action_shape[0]

        # Define the input shape of the observation space
        input_shape = (96, 96, 3)

        # Define the model architecture according to the paper
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(5, activation='linear'))

        # Compile the model
        model.compile(optimizer=Adam(lr=self._learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def create_P_DQN_model(self):


        # Define the input shape of the observation space
        input_shape = (96, 96, 3)

        # Define the model architecture according to the paper
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(5, activation='softmax'))

        # Compile the model
        model.compile(optimizer=Adam(lr=self._learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model
    

    def save_memory(self, obs, action, reward, new_obs, terminated):
        '''Add memory to queue during training'''
        self._memory.append([obs, action, reward, new_obs, terminated])
    
    def replay(self):
        '''Maximum reward if any possible action was taken.'''
        batch_size = self._batch_size

        # Don't take sample unless batch size is large enough
        if len(self._memory) < batch_size: 
            return 

        # Get samples from memory
        samples = random.sample(self._memory, batch_size)

        for sample in samples:
            # Get transition from the sample
            obs = sample[0]
            action = sample[1]
            reward = sample[2]
            new_obs = sample[3]
            terminated = sample[4]

            # Make a prediction y_j
            target = self._target_model.predict(obs, verbose=0)

            # Make a prediction y_pj
            p_target = self._P_target_model.predict(obs, verbose=0)

            # Bellman equation to learn optimal Q-value
            if terminated:
                # update Q model with respect to y_j 
                target[0][action] = reward
        
                # update P model with respect to y_pj 
                p_target[0][action] = reward
            else:
                # Calcualte the future Q value
                Q_f = max(self._target_model.predict(new_obs,  verbose=0)[0])
                target[0][action] = reward + Q_f * self._gamma

                # Calcualte the future Q value
                Q_f_p = max(self._P_target_model.predict(new_obs,  verbose=0)[0])
                p_target[0][action] = reward + Q_f_p * self._gamma
            
            # Fit Q Model
            self._model.fit(obs, target, epochs=1, verbose=0)

            # Fit P Model
            self._P_model.fit(obs, p_target, epochs=1, verbose=0)
    
    def act(self, obs):
        # Decay epsilon. Update e with e-decay
        self._epsilon *= self._epsilon_decay

        # Check epislon doesn't go past minimum
        if self._epsilon < self._epsilon_min:
            self._epsilon = self._epsilon_min
        
        # Choose random number
        rand_num = np.random.random() 
        if rand_num < self._epsilon:
            # Explore if random number is smaller than epsilon
            return self._env.action_space.sample()

        # Exploit and use model prediction if rand num > epsilon
        # use argmax to find position where probability is the greatest
        return np.argmax(self._model.predict(obs, verbose=0)[0])
    
    def proba_act(self, obs):      
        # Exploit and use model prediction if rand num > epsilon
        # use argmax to find position where probability is the greatest
        logits =  self._P_model.predict(obs, verbose=0)[0]
        #apply softmax to get the probability
        return np.max(softmax(logits))

    def act_P(self, obs, action):
        return self._P_model.predict(obs, verbose=0)[0][action]

    def save_Q_model(self, file_name):
        self._model.save(file_name)
    
    def save_P_model(self, file_name):
        self._P_model.save(file_name)
        
    def set_model(self, model):
        self._model = model

    def set_target_model(self, target_model):
        self._target_model = target_model
    
    def set_P_model(self, P_model):
        self._P_model = P_model
    
    def set_P_target_model(self, P_target_model):
        self._P_target_model = P_target_model

    def set_tau(self, tau):
        self._tau = tau
    
    def get_model(self):
        return self._model 

    def get_target_model(self):
        return self._target_model
    
    def get_P_model(self):
        return self._P_model
    
    def get_P_target_model(self):
        return self._P_target_model
    
    def get_tau(self):
        return self._tau

def main(timesteps=50, episodes=8, render_mode=None):
    '''Train a car racing environment based agent'''
    
    # Create a CarRacing Gym Environment
    env = gym.make("CarRacing-v2", render_mode=render_mode, continuous=False)

    # Initialize the Models Q and P according to the paper
    DQN_model = DeepQNetworkAgent(env)

    print("**** Training in Progress ... *******")

    # Start training agent
    for ep in range(episodes):

        # Reset the environment and get initial observations
        observation, info = env.reset(seed=6)

        # Add dimension
        observation = np.expand_dims(observation, axis=0)

        # Count reward per episode
        ep_reward = 0
        count = 0

        for timestep in range(timesteps):
            # action = env.action_space.sample()

            # Select an action a_t according to s_t using policy e-greedy
            action = DQN_model.act(observation)
            proba = DQN_model.proba_act(observation)
            action_p = DQN_model.act_P(observation, action)

            # Step environment using action predicted by policy
            # Take action a_t. Observe r_t and s_t+1
            new_obs, reward, terminated, truncated, info = env.step(action)

            # Save Memory
            new_obs = np.expand_dims(new_obs, axis=0)
            # Store transition
            DQN_model.save_memory(observation, action, reward, new_obs, terminated)
            
            # Replay
            # Take random sample of transitions
            DQN_model.replay()

            # Train Q Target
            target_model_updated = target_train_helper(DQN_model.get_model(),
                                                        DQN_model.get_target_model(),
                                                        DQN_model.get_tau())
            DQN_model.set_target_model(target_model_updated)

            # Train P Target
            P_target_model_updated = target_train_helper(DQN_model.get_P_model(),
                                                        DQN_model.get_P_target_model(),
                                                        DQN_model.get_tau())
            DQN_model.set_P_target_model(P_target_model_updated)

            # Get observation ready for next step
            observation = new_obs
            
            # Save the information from this step
            collect_data(action, reward, action_p, timestep)

            save_collected_data()

            ep_reward += reward
            count +=1

            if terminated or truncated:
                # Start new episode
                break
            
        print(f"\n\n\n Completed episode {ep} ")
        print(f" --- Average Reward {ep_reward/count}")
        DQN_model.save_Q_model("training_Q_model")
        DQN_model.save_P_model("training_P_model")
        

    
    # Save Trained Model
    DQN_model.save_Q_model("trained_Q_model") 
    DQN_model.save_P_model("trained_P_model") 

    env.close()


if __name__ == "__main__":
    #Input commands
    arg_parser = argparse.ArgumentParser()
    # To visualize: --render_mode human
    arg_parser.add_argument('--render_mode', dest='render_mode', type=str, default=None)
    arg_parser.add_argument('--mode', dest='mode', type=str, default='train')
    # Part Commands
    args = arg_parser.parse_args()

    # Train Agent
    if args.mode == 'train':
        main(render_mode=args.render_mode)
    elif args.mode == 'plot':
        plot_collected_data()
        
