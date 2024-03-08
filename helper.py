import matplotlib.pyplot as plt
from IPython import display

"""
This file is for Plotting the graph for the model 
to understand the training metrics better
"""
plt.ion()

def plot(scores, mean_scores):
    """Using Matplotlib to plot No of Games and Scores and update them as each game gets over"""
    display.clear_output(wait=True) # Clear Ouptut Cell
    display.display(plt.gcf()) #Create a figure if none found
    plt.clf() # Clear Existing Plot
    # Display labels
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores) # Plot game score
    plt.plot(mean_scores) # Plot mean score of entire training process
    plt.ylim(ymin=0)
    # Display text of scores
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)