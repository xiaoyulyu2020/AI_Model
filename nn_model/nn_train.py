from pathlib import Path
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
try:
    from .nn_model_app import LinearModel  # Works in Django
except ImportError:
    from nn_model_app import LinearModel


MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = '01_pytorch_model.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
DATA_PATH = Path('./data/Linear_data.csv')

def ai_model_train():
    """
        Train the model using data from CSV.
        """
    # Load data from CSV
    df = pd.read_csv(DATA_PATH)

    # Convert to PyTorch tensors
    X = torch.tensor(df["X"].values, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(df["Y"].values, dtype=torch.float32).view(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model_nn = LinearModel()
    loss_nn = nn.MSELoss()
    optimizer = torch.optim.Adam(model_nn.parameters(), lr=0.001)

    epochs = 1000
    for epoch in range(epochs):
        '''
        Training the model
        '''
        model_nn.train()
        for x_batch, y_batch in dataloader:
            y_predict = model_nn(x_batch)
            loss = loss_nn(y_predict, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(obj=model_nn.state_dict(), f=MODEL_SAVE_PATH)

if __name__ == "__main__":
    ai_model_train()
