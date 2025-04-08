import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix

import csv
import numpy as np

random.seed(42)

torch.cuda.empty_cache()


# Update font settings
plt.rcParams.update({'font.size': 8})
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

##################################################################################################################

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

######################################################
# HELPER FUNCTIONS
######################################################

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import hausdorff_distance

def compute_metrics(model_mask, ground_truth_mask, gt, pred):

    # Ensure masks are boolean
    model_mask = model_mask.astype(bool) 
    ground_truth_mask = ground_truth_mask.astype(bool)
    gt = gt.astype(bool) 
    pred = pred.astype(bool)

    # Compute true positives, false positives, false negatives, and true negatives
    tp = np.sum(model_mask & ground_truth_mask)  # True Positive (correctly predicted tumor)
    fp = np.sum(model_mask & ~ground_truth_mask) # False Positive (wrongly predicted tumor)
    fn = np.sum(~model_mask & ground_truth_mask) # False Negative (missed tumor)
    tn = np.sum(~model_mask & ~ground_truth_mask) # True Negative (correctly predicted background)

    # Dice Coefficient
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    # Hausdorff Distance (95%)
    if np.any(pred) and np.any(gt):  # Only compute if both have points
        model_points = np.argwhere(pred)
        gt_points = np.argwhere(gt)

        hausdorff_dist_95 = max(
            np.percentile([directed_hausdorff(model_points, gt_points)[0]], 95),
            np.percentile([directed_hausdorff(gt_points, model_points)[0]], 95)
        )
    else:
        hausdorff_dist_95 = 128  # No valid Hausdorff distance if one mask is empty

    # Precision, Recall, Specificity
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Dice Coefficient": dice,
        "95% Hausdorff Distance": hausdorff_dist_95,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity
    }


import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def save_metrics_to_csv(patient_number, dice, hausdorff, precision, recall, specificity):
    """Function to save metrics to a CSV file."""
    with open('metrics_output.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([patient_number, dice, hausdorff, precision, recall, specificity])

def compute_and_save_avg_metrics(metrics_list):
    """Function to compute and save the average metrics."""
    avg_metrics = np.mean(metrics_list, axis=0)
    # Save the average metrics for the dataset (train/val/test)
    save_metrics_to_csv('AVG', *avg_metrics)


def classify_and_evaluate(word, saved_model_path, data, masks, device, batch_size=4):
    # Load the model in evaluation mode
    model = torch.load(saved_model_path, map_location=device)
    model.to(device)
    model.eval()
    
    # Create DataLoader for the data and masks
    dataset = TensorDataset(torch.Tensor(data), torch.Tensor(masks))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    tars = []
    pres = []

    patient_metrics = []  # List to store metrics for each patient in the batch
    metrics_list = []

    # Get predictions using the model
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.70).long()

            # Squeeze the predictions to remove the extra channel dimension
            predictions = predictions.squeeze(1)  # Shape becomes (batch_size, slices, height, width)
            
            # Iterate over each sample in the batch and compute metrics separately
            for i in range(inputs.shape[0]):  # Iterate over batch
                pred = predictions[i:i+1]  # Slice a single prediction
                target = targets[i:i+1]  # Slice a single target
                # print(pred[0].shape, target[0].shape)  # both should be (95, 128, 128)

                # Extract prediction and ground truth masks
                pred_mask = pred[0]
                gt_mask = target[0]

                # Ensure pred_indices does not exceed gt_indices by more than 50 points
                number_excessive = 50

                # Move the tensors to CPU before using numpy
                pred_mask_cpu = pred_mask.cpu().numpy()
                gt_mask_cpu = gt_mask.cpu().numpy()

                # Get the indices where prediction and ground truth are 1
                pred_indices = np.where(pred_mask_cpu == 1)
                gt_indices = np.where(gt_mask_cpu == 1)

                # Ensure pred_indices does not exceed gt_indices by more than 50 points
                if len(pred_indices[0]) > len(gt_indices[0]) + number_excessive:
                    excess_points = len(pred_indices[0]) - (len(gt_indices[0]) + number_excessive)
                    remove_indices = np.random.choice(len(pred_indices[0]), excess_points, replace=False)
                    pred_indices = (np.delete(pred_indices[0], remove_indices),
                                    np.delete(pred_indices[1], remove_indices),
                                    np.delete(pred_indices[2], remove_indices))

                # Create the new prediction and ground truth masks (shape should remain (95, 128, 128))
                new_pred_mask = np.zeros_like(pred_mask_cpu)
                new_gt_mask = np.zeros_like(gt_mask_cpu)

                # Reassign values back to the new masks
                new_pred_mask[pred_indices] = 1
                new_gt_mask[gt_indices] = 1

                # Append the modified masks to the lists
                tars.append(gt_mask_cpu)
                pres.append(pred_mask_cpu)
                # patient_metrics.append((new_gt_mask, new_pred_mask))  # Store masks for later metrics calculation
                metrics = compute_metrics(pred_mask_cpu, gt_mask_cpu, new_gt_mask, new_pred_mask)
                metrics_list.append([metrics["Dice Coefficient"], metrics["95% Hausdorff Distance"], 
                                    metrics["Precision"], metrics["Recall"], metrics["Specificity"]])
                patient_number = len(metrics_list)  # Patient number for CSV

                if "TESTING" in word:
                    save_metrics_to_csv(patient_number, *metrics_list[-1]) 

    # After processing all patients, compute and save average metrics
    compute_and_save_avg_metrics(metrics_list)

    return metrics_list, np.array(tars), np.array(pres)

def make_comparison_images(predictions, ground_truth, num_patients, folder_path, view=(30, 30)):
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Randomly select patients
    patient_indices = random.sample(range(len(predictions)), num_patients)
    
    print(predictions.shape, ground_truth.shape)
    # for patient_num in patient_indices:
    for patient_num in range(predictions.shape[0]):
        print(f"display for patient #{patient_num}")
        pred_mask = predictions[patient_num]  # Remove the channel dimension
        gt_mask = ground_truth[patient_num]  # Remove the channel dimension

        pred_indices = np.where(pred_mask == 1)
        gt_indices = np.where(gt_mask == 1)

        # Ensure pred_indices does not exceed gt_indices by more than 50 points
        number_excessive = 50
        if len(pred_indices[0]) > len(gt_indices[0]) + number_excessive:
            excess_points = len(pred_indices[0]) - (len(gt_indices[0]) + number_excessive)
            remove_indices = np.random.choice(len(pred_indices[0]), excess_points, replace=False)
            pred_indices = (np.delete(pred_indices[0], remove_indices),
                            np.delete(pred_indices[1], remove_indices),
                            np.delete(pred_indices[2], remove_indices))
        
        
        fig = plt.figure(figsize=(10, 5.4))
        
        # Ground Truth
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        if gt_indices[0].size > 0:
            ax1.scatter(gt_indices[0], gt_indices[1], gt_indices[2], color=(30/255, 0/255, 230/255), s=1, alpha=1)
        ax1.set_xlabel('Axial Slices')
        # ax1.set_ylabel('Saggital Slices')
        # ax1.set_zlabel('Axial Slices')
        ax1.set_title(f'Test Patient ID {patient_num} - Ground Truth')
        ax1.set_xlim([0, gt_mask.shape[0]])
        ax1.set_ylim([0, gt_mask.shape[1]])
        ax1.set_zlim([0, gt_mask.shape[2]])
        ax1.view_init(elev=view[0], azim=view[1])
        
        # Prediction
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        if pred_indices[0].size > 0:
            ax2.scatter(pred_indices[0], pred_indices[1], pred_indices[2], color=(128/255, 0/255, 0/255), s=1, alpha=1)
        ax2.set_xlabel('Axial Slices')
        # ax2.set_ylabel('Saggital Slices')
        # ax2.set_zlabel('Coronal Slices')
        ax2.set_title(f'Test Patient ID {patient_num} - CNN Baseline Prediction')
        ax2.set_xlim([0, pred_mask.shape[0]])
        ax2.set_ylim([0, pred_mask.shape[1]])
        ax2.set_zlim([0, pred_mask.shape[2]])
        ax2.view_init(elev=view[0], azim=view[1])
        
        plt.tight_layout()
        save_path = os.path.join(folder_path, f"patient_{patient_num}_comparison.png")
        plt.savefig(save_path)
        plt.close()


def MODEL_RESULTS(saved_model_path, num_patients, mask_path, epoch_num, num_folds=1):
    # Set CUDA environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU (change if you have multiple GPUs)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error messages

    # No need to test all folds if we don't want to
    num_folds = len(train_inds) if num_folds == 0 else num_folds

    # For each fold, we can train a model
    for fold in range(num_folds):
        print(f"\n***********************\nRESULTS ON FOLD #{fold+1}/{num_folds}\n")
        
        # Get data indices
        train_indices = train_inds[fold]
        val_indices = val_inds[fold]

        # Data processed like how we did for training
        train_voxels = voxels[train_indices].reshape(int(len(train_indices) / 4), 4, 95, 128, 128)
        val_voxels = voxels[val_indices].reshape(int(len(val_indices) / 4), 4, 95, 128, 128)
        test_voxels = tvoxels.reshape(int(len(tlabels) / 4), 4, 95, 128, 128)
        # Here are the segmentations
        train_segs, val_segs = get_indices_from_labels(seg_labels, train_indices, val_indices) 
        train_segs = np.array(train_segs).reshape(int(len(train_indices) / 4), 95, 128, 128)
        val_segs = np.array(val_segs).reshape(int(len(val_indices) / 4), 95, 128, 128)
        test_segs = np.array(tseg_labels).reshape(int(len(tlabels) / 4), 95, 128, 128)
    
        print(f"Training {train_voxels.shape} {train_segs.shape}")
        print(f"Validation {val_voxels.shape} {val_segs.shape}")
        print(f"Testing {test_voxels.shape} {test_segs.shape}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using Device: {device}!")

        # Reinitialize CUDA
        torch.cuda.empty_cache()

        # classify_and_evaluate("\nTRAINING", saved_model_path, train_voxels, train_segs, device)
        # classify_and_evaluate("\nVALIDATION", saved_model_path, val_voxels, val_segs, device)
        # classify_and_evaluate("\nTESTING", saved_model_path, test_voxels, test_segs, device)

        # training_dice, training_gt, training_preds = classify_and_evaluate("\nTRAINING", saved_model_path, train_voxels, train_segs, device)
        # val_dice, val_gt, val_preds = classify_and_evaluate("\nVALIDATION", saved_model_path, val_voxels, val_segs, device)
        test_dice, test_gt, test_preds = classify_and_evaluate("\nTESTING", saved_model_path, test_voxels, test_segs, device)
        make_comparison_images(test_preds, test_gt, num_patients, os.path.join(mask_path, f"{epoch_num}/test"))


######################################################
# MAIN
######################################################

print("Starting!")

# reload from the files of split data
split_data_path = "/scratch/k/khalvati/liufeli2/LLMs/data/SPLIT_DATA_FULL.npz"
processed_segs_path = "/scratch/k/khalvati/liufeli2/LLMs/data/SPLIT_SEG_FULL.npz"
voxels, labels, tvoxels, tlabels, num_folds, train_inds, val_inds, seg_labels, tseg_labels = unload_data_from_npz(split_data_path, processed_segs_path)

# save_model_numbers = [200]
mask_path = "/scratch/k/khalvati/liufeli2/LLMs/seg_baseline/SEGMENTATION_MASKSSS"
num_patients = 10

# for num in save_model_numbers:
#     print(f"\n**********************\n{num} EPOCHS")
num = 100000
saved_model_path = f"/scratch/k/khalvati/liufeli2/LLMs/seg_baseline/saved_model_100_16epochs.pth"
MODEL_RESULTS(saved_model_path, num_patients, mask_path, num+1)

