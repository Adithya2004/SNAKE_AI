import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# Make the model and its parameters
class Linear_QNet(nn.Module):
    """
    Initialize the model with all the required States 
    In this use case the following are 
    Input - 11
    Hidden - 256
    Output - 3
    Create two linear layers to get the above required model
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    # Forward function that can take input and run model on it to train or predict
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    # Function to save the model whenever required
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)



"""
Trainer Class to be able to control model from agent Class 
Abstracts Training process into easily understandable segment here.
"""
class QTrainer:
    # Initialize all parameters required for Training Model 
    # such as Loss, Optimization , gamma and Learning Rate
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    #  Calculate Current State and trains to get next prediction
    #  using Loss function MSE = Qnew = R + Y*Qmax
    def train_step(self, state, action, reward, next_state, done):
        #Get current state and arguement values for training
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        # Format above if not in correct shape
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        # Loss and optimizer Calculations  
        loss.backward()
        self.optimizer.step()


