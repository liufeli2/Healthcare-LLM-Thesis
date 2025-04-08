######################################################
# IMPORTS
######################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import time

torch.cuda.empty_cache()

######################################################
# CNN CLASS
######################################################

class CNN(nn.Module):
    def __init__(self, input_channels=4):
        super(CNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # Downsample by 2x
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # Downsample by 2x
        # Bottleneck
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        # Decoder (Upsampling)
        self.deconv1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0, output_padding=(1, 0, 0))
        self.deconv2 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2, padding=0, output_padding=(1, 0, 0))
        self.final_conv = nn.Conv3d(16, 1, kernel_size=3, padding=1)  # Output mask (1 channel)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x1p = self.pool1(x1)  # Downsample
        x2 = F.relu(self.conv2(x1p))
        x2p = self.pool2(x2)  # Downsample
        x3 = F.relu(self.conv3(x2p))
        x4 = F.relu(self.deconv1(x3))  # Upsample
        x5 = F.relu(self.deconv2(x4))  # Upsample
        out = torch.sigmoid(self.final_conv(x5))  # Sigmoid activation for binary segmentation
        return out  # The shape should now match the input size
    


######################################################
# DATA FUNCTIONS
######################################################

def unload_data_from_npz(split_data_path, processed_segs_path):
    loaded_data = np.load(split_data_path, allow_pickle=True)
    voxels = loaded_data['voxels']
    labels = loaded_data['labels']
    tvoxels = loaded_data['tvoxels']
    tlabels = loaded_data['tlabels']
    num_folds = (len(loaded_data.files)-4)//2
    train_inds = [loaded_data[f'train_inds_{i}'] for i in range(num_folds)]
    val_inds = [loaded_data[f'val_inds_{i}'] for i in range(num_folds)]
    loaded_seg_data = np.load(processed_segs_path)
    trainval_seg_labels, test_seg_labels = loaded_seg_data['trainval_seg_labels'], loaded_seg_data['test_seg_labels']
    # print(f"TRAIN/VAL DATA: {voxels.shape} {labels.shape}")
    # print(f"TEST DATA:      {tvoxels.shape} {tlabels.shape}")
    # for i in range(len(train_inds)):
    #     print(f"fold #{i}: train/val [{len(train_inds[i])}/{len(val_inds[i])}]")
    return voxels, labels, tvoxels, tlabels, num_folds, train_inds, val_inds, trainval_seg_labels, test_seg_labels

def get_indices_from_labels(labels, mini_train_inds, mini_val_inds):
    train_segs = []
    val_segs = []  
    for i in range(0, len(mini_train_inds), 4):
        index = mini_train_inds[i]//4
        train_segs.append(labels[index])
    for i in range(0, len(mini_val_inds), 4):
        index = mini_val_inds[i]//4
        val_segs.append(labels[index])
    train_segs = np.array(train_segs)
    val_segs = np.array(val_segs)
    return train_segs, val_segs


# ######################################################
# # TRAINING FUNCTIONS
# ######################################################


def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous().unsqueeze(1)  # Add an extra dimension to match pred
    intersection = (pred * target).sum(dim=(2, 3, 4))
    pred_sum = pred.sum(dim=(2, 3, 4))
    target_sum = target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    loss = 1 - dice.mean()
    return loss

######################################################
# TRAINING FUNCTIONS
######################################################
    

def train_and_evaluate(device, voxels, labels, train_indices, val_indices, model, criterion, optimizer, num_epochs, batch_size, save_path):
    torch.manual_seed(42)

    # reshape the data to have C_in be the medicals photos channel (=4)
    train_data = torch.Tensor(voxels[train_indices]).reshape(int(len(train_indices)/4), 4, 95, 128, 128).to(device)
    val_data = torch.Tensor(voxels[val_indices]).reshape(int(len(val_indices)/4), 4, 95, 128, 128).to(device)
    # segmentation stuff
    train_segs, val_segs = get_indices_from_labels(seg_labels, train_indices, val_indices) 
    train_segs = torch.Tensor(np.array(train_segs)).reshape(int(len(train_indices)/4), 95, 128, 128).to(device)
    val_segs = torch.Tensor(np.array(val_segs)).reshape(int(len(val_indices)/4), 95, 128, 128).to(device)

    # print sizes just to check
    print(f"Training Data: {train_data.shape} Training Segs: {train_segs.shape}")
    print(f"Validation Data: {val_data.shape} Validation Segs: {val_segs.shape}")

    # create the dataset and dataloader
    train_dataset = TensorDataset(train_data, train_segs)
    val_dataset = TensorDataset(val_data, val_segs)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # store some values to later use for plotting
    train_losses, valid_losses = [], []

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
        avg_train_loss = train_loss / len(train_loader)
        avg_valid_loss = evaluate(model, val_loader, device, criterion)
        train_losses.append(avg_train_loss)
        valid_losses.append(avg_valid_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Valid Loss: {avg_valid_loss:.4f}")
        
    torch.save(model, save_path)
    print(f"Model saved to {save_path}")

    print('Finished Training!')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    return

def evaluate(model, loader, device, criterion):
    model.eval()
    tot_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            tot_loss += loss.item()
    model.train()
    return tot_loss / len(loader)



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
        criterion = dice_loss  # Use the dice_loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print(f"Using Device: {device}!")
        
        # train the model
        train_and_evaluate(device, voxels, labels, train_indices, val_indices, model, criterion, optimizer, num_epochs, batch_size, save_path)


######################################################
# MAIN
######################################################

# reload from the files of split data
split_data_path = "/scratch/k/khalvati/liufeli2/LLMs/data/SPLIT_DATA_FULL.npz"
processed_segs_path = "/scratch/k/khalvati/liufeli2/LLMs/data/SPLIT_SEG_FULL.npz"
voxels, labels, tvoxels, tlabels, num_folds, train_inds, val_inds, seg_labels, tseg_labels = unload_data_from_npz(split_data_path, processed_segs_path)

try:
    loc = "/scratch/k/khalvati/liufeli2/LLMs/seg_baseline/saved_model_70_32epochs.pth"
    TRAIN_MODEL(num_folds=1, learning_rate=0.0001, num_epochs=70, batch_size=24, save_path=loc)

except KeyboardInterrupt:
    print("Process interrupted. Clearing memory...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print("Memory cleared successfully.")
