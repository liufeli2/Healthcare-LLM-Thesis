######################################################
# IMPORTS
######################################################

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

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
    return voxels, labels, tvoxels, tlabels, num_folds, train_inds, val_inds



######################################################
# HELPER FUNCTIONS
######################################################


def compute_metrics(y_true, y_pred, y_scores=None):
    # convert tensors to numpy arrays
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    y_scores = y_scores.cpu().numpy() if y_scores is not None else None
    # generate confusion matrix components
    TP = ((y_true == 1) & (y_pred == 1)).sum()  # True Positives
    TN = ((y_true == 0) & (y_pred == 0)).sum()  # True Negatives
    FP = ((y_true == 0) & (y_pred == 1)).sum()  # False Positives
    FN = ((y_true == 1) & (y_pred == 0)).sum()  # False Negatives
    # compute metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    # compute AUC if y_scores is available
    auc = roc_auc_score(y_true, y_scores) if y_scores is not None else None
    return accuracy, precision, recall, f1, auc, (TP, TN, FP, FN)


def classify_and_evaluate(word, saved_model_path, data, labels, device):
    # load the model in evaluation mode
    model = torch.load(saved_model_path)
    model.to(device)
    model.eval()
    # prep all tensor data
    data = torch.Tensor(data).to(device)
    labels = torch.unique(torch.Tensor(labels), dim=1).to(device)
    # get predictions using the model
    with torch.no_grad():
        outputs = model(data)
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities > 0.5).long()
    # compute and print metrics
    accuracy, precision, recall, f1, auc, (TP, TN, FP, FN) = compute_metrics(labels, predictions, probabilities)
    print(f"{word} Classification Metrics:")
    print(f"Accuracy: {accuracy:.4f} F1 Score: {f1:.4f} Precision: {precision:.4f} Recall: {recall:.4f}")
    if auc is not None:
        print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    return accuracy, precision, recall, f1, auc, (TP, TN, FP, FN)


def MODEL_RESULTS(saved_model_path, num_folds=1):

    # no need to test all folds if we don't want to
    num_folds = len(train_inds) if num_folds==0 else num_folds

    # for each fold, we can train a model
    for fold in range(num_folds):
        print(f"\n***********************\nRESULTS ON FOLD #{fold+1}/{num_folds}\n")
        
        # get data indices
        train_indices = train_inds[fold]
        val_indices = val_inds[fold]

        # data processed like how we did for training
        train_voxels = voxels[train_indices].reshape(int(len(train_indices)/4), 4, 95, 128, 128)
        train_labels = labels[train_indices].reshape(int(len(train_indices)/4), 4) 
        val_voxels = voxels[val_indices].reshape(int(len(val_indices)/4), 4, 95, 128, 128)
        val_labels = labels[val_indices].reshape(int(len(val_indices)/4), 4) 
        test_voxels = tvoxels.reshape(int(len(tlabels)/4), 4, 95, 128, 128)
        test_labels = tlabels.reshape(int(len(tlabels)/4), 4) 
    
        print(f"Training {train_voxels.shape} {train_labels.shape}")
        print(f"Validation {val_voxels.shape} {val_labels.shape}")
        print(f"Testing {test_voxels.shape} {test_labels.shape}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using Device: {device}!")
        classify_and_evaluate("\nTRAINING", saved_model_path, train_voxels, train_labels, device)
        classify_and_evaluate("\nVALIDATION", saved_model_path, val_voxels, val_labels, device)
        classify_and_evaluate("\nTESTING", saved_model_path, test_voxels, test_labels, device)



######################################################
# MAIN
######################################################

# reload from the files of split data
split_data_path = "/scratch/k/khalvati/liufeli2/LLMs/data/SPLIT_DATA_FULL.npz"
voxels, labels, tvoxels, tlabels, num_folds, train_inds, val_inds = unload_data_from_npz(split_data_path)

# saved_model_path = "/scratch/k/khalvati/liufeli2/LLMs/cnn_baseline/models/saved_model_50epochs.pth"
# saved_model_path = "/scratch/k/khalvati/liufeli2/LLMs/cnn_baseline/models/saved_model_80epochs.pth"
# saved_model_path = "/scratch/k/khalvati/liufeli2/LLMs/cnn_baseline/models/saved_model_100epochs.pth"
# saved_model_path = "/scratch/k/khalvati/liufeli2/LLMs/cnn_baseline/models/saved_model_200epochs.pth"
save_model_numbers = [50, 80, 100, 200]

for num in save_model_numbers:
    print(f"\n**********************\n{num} EPOCHS")
    saved_model_path = f"/scratch/k/khalvati/liufeli2/LLMs/cnn_baseline/models/saved_model_{num}epochs.pth"
    MODEL_RESULTS(saved_model_path)


# python /scratch/k/khalvati/liufeli2/LLMs/cnn_baseline/evaluate_cnn_results.py