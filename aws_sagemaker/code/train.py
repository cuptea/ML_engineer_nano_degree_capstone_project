import argparse
import json
import os
import pickle
import sys
#import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader

from model import NCF

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NCF(model_info['embedding_dim'])

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    # Load the saved token_to_idx.
    token_dict_path = os.path.join(model_dir, 'token_to_idx.pickle')
    with open(token_dict_path, 'rb') as f:
        model.token_to_idx = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model


class KKbox_set(Dataset):
    def __init__(self, df, transformation=None):
        super(KKbox_set).__init__()
        self.data = df
        self.data = list(self.data.T.to_dict().values())
        self.transform = transformation     
        
    def __getitem__(self,index):
        """return a role assignment in a dictionary form"""
        sample = self.data[index]
        return sample
    
    def __len__(self):
        return len(self.data)
    
def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train_sample.csv"))
    print(train_data.head())
    
    return DataLoader(KKbox_set(train_data), batch_size=batch_size)

def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")
        
def train(model, train_loader, epochs, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    
    # TODO: Paste the train() method developed in the notebook here.

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch, batch['target'].float()
            
            batch_X = move_to(batch_X,device)
            batch_y = move_to(batch_y,device)
                        
            # TODO: Complete this train method to train the model provided.
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(batch_X)
            
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()

        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    
    parser.add_argument('--embedding_dim', type=int, default=10, metavar='N',
                        help='size of the embeddings (default: 32)')
    
    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    #parser.add_argument('--data-dir', type=str, default="../data/")
    #parser.add_argument('--model-dir', type=str, default="../data/models/")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)
    
    token_info_path = os.path.join(args.data_dir, 'token_to_idx.pickle')
    with open(token_info_path, 'rb') as handle:
        token_to_idx = pickle.load(handle)

    # Build the model.
    model = NCF(n_dim = args.embedding_dim).to(device)

    print("Model loaded with embedding_dim {}.".format(
        args.embedding_dim
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    train(model, train_loader, args.epochs, optimizer, loss_fn, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
        }
        torch.save(model_info, f)

    # Save the token dictionary
    token_info_path = os.path.join(args.model_dir, 'token_to_idx.pickle')
    with open(token_info_path, 'wb') as handle:
        pickle.dump(token_to_idx, handle)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
