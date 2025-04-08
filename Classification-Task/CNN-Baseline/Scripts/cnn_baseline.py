######################################################
# IMPORTS
######################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.utils.data import DataLoader
import time

torch.cuda.empty_cache()


######################################################
# CNN CLASS
######################################################

class CNN(nn.Module):
    def __init__(self, input_channels=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 8, kernel_size=3, stride=1, padding=1)  # 8 filters, kernel size 3x3x3
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)  # Pooling layer with stride 2 to downsample
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 23 * 32 * 32, 128)  # Adjust the size based on the final spatial size after pooling
        self.fc2 = nn.Linear(128, 1)  # Output layer for binary classification
    def forward(self, x):
        x = self.conv1(x)  # Apply the first convolution
        x = nn.ReLU()(x)  # ReLU activation
        x = self.pool(x)  # Apply max pooling
        x = self.conv2(x)  # Apply the second convolution
        x = nn.ReLU()(x)  # ReLU activation
        x = self.pool(x)  # Apply max pooling
        x = x.view(x.size(0), -1)
        x = self.fc1(x)  # Fully connected layer 1
        x = self.fc2(x)  # Output layer
        return x


######################################################
# HELPER FUNCTIONS
######################################################


def count_zeros_ones(vector):
    num_zeros = (vector == 0).sum().item()  # Count elements that are 0
    num_ones = (vector == 1).sum().item()   # Count elements that are 1
    return num_zeros, num_ones


def count_matches(predictions, targets):
    correct_matches = (predictions == targets).sum().item()  # Sum the number of correct matches
    total_comparisons = predictions.numel()  # Total number of elements in the tensor
    return total_comparisons, correct_matches


######################################################
# DATA FUNCTIONS
######################################################

def unload_data_from_npz(split_data_path):
    loaded_data = np.load(split_data_path, allow_pickle=True)
    voxels = loaded_data['voxels']
    labels = loaded_data['labels']
    tvoxels = loaded_data['tvoxels']
    tlabels = loaded_data['tlabels']
    num_folds = (len(loaded_data.files)-4)//2
    train_inds = [loaded_data[f'train_inds_{i}'] for i in range(num_folds)]
    val_inds = [loaded_data[f'val_inds_{i}'] for i in range(num_folds)]
    # print(f"TRAIN/VAL DATA: {voxels.shape} {labels.shape}")
    # print(f"TEST DATA:      {tvoxels.shape} {tlabels.shape}")
    # for i in range(len(train_inds)):
    #     print(f"fold #{i}: train/val [{len(train_inds[i])}/{len(val_inds[i])}]")
    return voxels, labels, tvoxels, tlabels, num_folds, train_inds, val_inds


def process_data(train_data, train_labels):
    zero_indices = (train_labels.squeeze() == 0).nonzero(as_tuple=True)[0]
    one_indices = (train_labels.squeeze() == 1).nonzero(as_tuple=True)[0]
    num_zeros, num_ones = zero_indices.size(0), one_indices.size(0)
    oversample_factor = num_ones // num_zeros 
    oversampled_zero_indices = zero_indices.repeat(oversample_factor + 1)[:num_ones]
    balanced_indices = torch.cat([oversampled_zero_indices, one_indices])
    balanced_indices = balanced_indices[torch.randperm(balanced_indices.size(0))]
    return train_data[balanced_indices], train_labels[balanced_indices]


######################################################
# TRAINING FUNCTIONS
######################################################

def train_and_evaluate(device, voxels, labels, train_indices, val_indices, model, criterion, optimizer, num_epochs, batch_size, save_path):
    torch.manual_seed(42)

    # reshape the data to have C_in be the medicals photos channel (=4)
    train_data = torch.Tensor(voxels[train_indices]).reshape(int(len(train_indices)/4), 4, 95, 128, 128).to(device)
    val_data = torch.Tensor(voxels[val_indices]).reshape(int(len(val_indices)/4), 4, 95, 128, 128).to(device)
    train_labels = torch.Tensor(labels[train_indices]).reshape(int(len(train_indices)/4), 4).to(device)
    val_labels = torch.Tensor(labels[val_indices]).reshape(int(len(val_indices)/4), 4).to(device)
    train_labels = torch.unique(train_labels, dim=1)
    val_labels = torch.unique(val_labels, dim=1)
    train_data, train_labels = process_data(train_data, train_labels) # balances data
    print(f"Training Data: {train_data.shape} {train_labels.shape} #0s/#1s: {count_zeros_ones(train_labels)}")
    print(f"Validation Data: {val_data.shape} {val_labels.shape} #0s/#1s: {count_zeros_ones(val_labels)}") 

    # create the dataset and dataloader
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # store some values to later use for plotting
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    # put the model in training mode
    start_time = time.time()
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # and append to our storage for printing
        avg_train_accuracy, avg_train_loss = evaluate(model, train_loader, device, criterion)
        avg_valid_accuracy, avg_valid_loss = evaluate(model, val_loader, device, criterion)
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        train_accuracies.append(avg_train_accuracy)
        valid_accuracies.append(avg_valid_accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, "
              f"Valid Loss: {avg_valid_loss:.4f}, Valid Accuracy: {avg_valid_accuracy:.4f}")
        
    torch.save(model, save_path)
    print(f"Model saved to {save_path}")

    print('Finished Training!')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    print("Final Training Accuracy: {}".format(train_accuracies[-1]))
    print("Final Validation Accuracy: {}".format(valid_accuracies[-1]))

    return
    

def evaluate(model, loader, device, criterion):
    correct_preds, total_preds, tot_loss = 0, 0, 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        tot_loss += loss.item()
        preds = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold to get binary predictions
        total_comparisons, correct_matches = count_matches(preds, labels)
        correct_preds += correct_matches
        total_preds += total_comparisons
    model.train()
    return correct_preds/total_preds, tot_loss/len(loader)


def TRAIN_MODEL(num_folds=0, learning_rate=0.001, num_epochs=5, batch_size=16, save_path=""):

    # no need to test all folds if we don't want to
    num_folds = len(train_inds) if num_folds==0 else num_folds

    # for each fold, we can train a model
    for fold in range(num_folds):
        print(f"\n***********************\nTRAINING FOLD #{fold+1}/{num_folds}\n")
        
        # get data indices
        train_indices = train_inds[fold]
        val_indices = val_inds[fold]
        
        # initialize model, loss function, and optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNN(input_channels=4).to(device)
        # model = SmallResNet3D(input_channels=4).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print(f"Using Device: {device}!")
        
        # train the model
        train_and_evaluate(device, voxels, labels, train_indices, val_indices, model, criterion, optimizer, num_epochs, batch_size, save_path)


######################################################
# MAIN
######################################################

# reload from the files of split data
split_data_path = "/scratch/k/khalvati/liufeli2/LLMs/data/SPLIT_DATA_FULL.npz"
voxels, labels, tvoxels, tlabels, num_folds, train_inds, val_inds = unload_data_from_npz(split_data_path)

try:
    loc = "/scratch/k/khalvati/liufeli2/LLMs/cnn_baseline/saved_model_80epochs.pth"
    TRAIN_MODEL(num_folds=1, learning_rate=0.000001, num_epochs=80, batch_size=16, save_path=loc)

except KeyboardInterrupt:
    print("Process interrupted. Clearing memory...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("Memory cleared successfully.")



# python /scratch/k/khalvati/liufeli2/LLMs/cnn_baseline/cnn_baseline.py