import pandas as pd 
import matplotlib.pyplot as plt
ACTION_LIST = []
REWARD_LIST = []
PROBA_LIST = []
TIMESTEP_LIST = []
data_collected = pd.DataFrame()

def target_train_helper(model, target_model, tau):
    '''Copy weights from main model to target model
    
    Source: https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
    '''
    W = model.get_weights()
    target_W = target_model.get_weights()

    for i in range(len(target_W)):
        target_W[i] = W[i] * tau + target_W[i] * (1-tau)
    
    target_model.set_weights(target_W)

    return target_model


def collect_data(action, reward, proba, timestep):
    '''Add data to global lists'''
    global REWARD_LIST, PROBA_LIST, TIMESTEP_LIST

    ACTION_LIST.append(action)
    REWARD_LIST.append(reward)
    PROBA_LIST.append(proba)
    TIMESTEP_LIST.append(timestep)

def save_collected_data():
    '''Save the collecterd data into a dataframe'''
    data_collected = pd.DataFrame({'actions':ACTION_LIST, 
                                   'rewards': REWARD_LIST, 
                                   'probability': PROBA_LIST, 
                                   'timestep':TIMESTEP_LIST})
    data_collected.to_csv('data/data_collected2.csv', index=False)


def plot_collected_data():
    '''Plot the Probability aglortihm across timesteps'''

    # Read the CSV file into a DataFrame
    data_collected = pd.read_csv('data/data_collected.csv')

    # Get the rewards, probability, and timestep lists from the DataFrame
    REWARD_LIST = data_collected['rewards']
    PROBA_LIST = data_collected['probability']
    TIMESTEP_LIST = list(range(len(data_collected['timestep'])))

    # Create a figure and axis objects for the reward plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Plot the rewards and timestep lists as a red line
    ax1.plot(TIMESTEP_LIST, REWARD_LIST, color='red')

    # Set the axis labels and title
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward vs Timestep')

    # Plot the probability and timestep lists as a blue line
    # ax2.plot(TIMESTEP_LIST, PROBA_LIST, color='red')

    import numpy as np
    
    # Fit a polynomial regression line to the data
    polyfit = np.polyfit(TIMESTEP_LIST, PROBA_LIST, deg=3)
    polyline = np.poly1d(polyfit)

    # Generate x values for the line
    x_values = range(min(TIMESTEP_LIST), max(TIMESTEP_LIST) + 1)

    # Plot the scatter points and the line of best fit
    plt.plot(x_values, polyline(x_values), 'r')

    # Set the y-axis limits for the probability plot
    ax2.set_ylim([0, 1.1])

    # Set the axis labels and title
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Probability')
    ax2.set_title('Probability vs Timestep')

    # Adjust the spacing between the subplots
    plt.subplots_adjust(hspace=0.5)

    # Save the plots to files
    fig.savefig('captures/reward_plot.png')
    fig.savefig('captures/probability_plot.png')

    # Display the plots
    plt.show()


