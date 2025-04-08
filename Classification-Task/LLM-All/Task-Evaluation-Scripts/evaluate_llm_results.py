################################################################################# 
# IMPORTS
################################################################################# 

import os
import re
import numpy as np


################################################################################# 
# EXTRACT FUNCTIONS
################################################################################# 

def extract_number(file_path):
    match = re.search(r'scan_results_(\d+)\.txt', file_path)
    if match:
        return int(match.group(1))
    return float('inf')  # Return a large number if no match found

def get_file_paths_in_folder(folder_path):
    file_paths = []
    # Walk through the folder and its subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt'):
                file_paths.append(os.path.join(root, file))
    return sorted(file_paths, key=extract_number)

def extract_tally_and_labels(file_path):
    tally = None
    true_label = None
    prediction_label = None
    with open(file_path, 'r') as file:
        for line in file:
            # Search for the "Tally" line and extract the array
            if line.startswith("Tally:"):
                tally_match = re.search(r'Tally:\s*(\[[^\]]*\])', line)
                if tally_match:
                    tally = eval(tally_match.group(1))  # Convert string list to actual list
            # Search for the "True Label" and "Prediction Label" on the same line
            if "True Label:" in line and "Prediction Label:" in line:
                true_label_match = re.search(r'True Label:\s*(\S+)', line)
                prediction_label_match = re.search(r'Prediction Label:\s*(\S+)', line)
                if true_label_match and prediction_label_match:
                    true_label = true_label_match.group(1)
                    prediction_label = prediction_label_match.group(1)
    return tally, true_label, prediction_label


def evaluate_prediction(tally):
    actual_guess = np.argmax(tally)
    return actual_guess


def evaluate_prediction_single(tally):
    lgg_hgg_guess, actual_guess = np.argmax(tally[0:2]), np.argmax(tally)
    # print(tally[0], tally[1], tally[0]/tally[1])
    return actual_guess, sum(tally[2:])


def scores(confusion_matrix):
    # True positives (TP), false positives (FP), false negatives (FN) for each class
    # For Class 0 (True class 0), we consider Predicted 0 (Class 0), Predicted 1 (Class 1), Predicted 2 (Don't know), Predicted 3 (Don't know)
    TP_0 = confusion_matrix[0, 0]  # Correctly predicted as Class 0
    FP_0 = confusion_matrix[1, 0] + confusion_matrix[0, 2] + confusion_matrix[0, 3]  # Misclassified as Class 0
    FN_0 = confusion_matrix[1, 1] + confusion_matrix[1, 2] + confusion_matrix[1, 3]  # Missed Class 0 predictions

    # For Class 1 (True class 1), we consider Predicted 0 (Class 0), Predicted 1 (Class 1), Predicted 2 (Don't know), Predicted 3 (Don't know)
    TP_1 = confusion_matrix[1, 1]  # Correctly predicted as Class 1
    FP_1 = confusion_matrix[0, 1] + confusion_matrix[1, 2] + confusion_matrix[1, 3]  # Misclassified as Class 1
    FN_1 = confusion_matrix[0, 0] + confusion_matrix[0, 2] + confusion_matrix[0, 3]  # Missed Class 1 predictions

    # Accuracy: (True Positives for Class 0 + True Positives for Class 1) / Total Samples
    accuracy = (TP_0 + TP_1) / np.sum(confusion_matrix)

    # Precision and Recall for Class 0
    precision_0 = TP_0 / (TP_0 + FP_0) if (TP_0 + FP_0) != 0 else 0
    recall_0 = TP_0 / (TP_0 + FN_0) if (TP_0 + FN_0) != 0 else 0

    # Precision and Recall for Class 1
    precision_1 = TP_1 / (TP_1 + FP_1) if (TP_1 + FP_1) != 0 else 0
    recall_1 = TP_1 / (TP_1 + FN_1) if (TP_1 + FN_1) != 0 else 0

    # Output the results
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision for Class 0: {precision_0:.2f}")
    print(f"Recall for Class 0: {recall_0:.2f}")
    print(f"Precision for Class 1: {precision_1:.2f}")
    print(f"Recall for Class 1: {recall_1:.2f}")

    col_sum = np.sum(confusion_matrix, axis=0)
    row_sum = np.sum(confusion_matrix, axis=1)
    tot_sum = np.sum(confusion_matrix)

    print(col_sum, row_sum, tot_sum)


# this will group into 2x4 and catagorise each scan seperately (no evaluation metrics)
def RUN_FOR_SCANS():

    labels = []
    predictions = []
    all_preds = [[[], []], [[], []]]
    confusion = np.zeros((2, 4))

    for file_path in file_paths:
        tally, true_label, prediction_label = extract_tally_and_labels(file_path)
        if true_label == None:
            continue
        ground_truth = class_map[true_label]
        pred, ratio = evaluate_prediction_single(tally)

        # all_preds[ground_truth][pred].append(ratio)
        confusion[ground_truth, pred] += 1
        labels.append(ground_truth)
        predictions.append(pred)

        print(f"Scan:{extract_number(file_path)} T:{ground_truth} P:{pred}", tally)
    # print(all_preds)

    print()
    accuracy, precision, recall, f1, (TP, TN, FP, FN) = compute_metrics(labels, predictions)
    print(f"TESTING Classification Metrics:")
    print(f"Accuracy: {accuracy:.4f} F1 Score: {f1:.4f} Precision: {precision:.4f} Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

    return confusion


# this will only create a 2x2 confusion matrix and group 4 patient scans into 1
def RUN_FOR_PATIENTS():

    labels = []
    predictions = []

    confusion = np.zeros((2, 4))
    total_tally = None
    files_processed = 0
    ground_truths = []
    scan_nums = []

    for file_path in file_paths:
        tally, true_label, prediction_label = extract_tally_and_labels(file_path)
        
        if true_label is None:
            continue
        
        ground_truth = class_map[true_label]
        if total_tally is None:
            total_tally = tally
        else:
            total_tally = np.add(total_tally, tally) 
        files_processed += 1
        ground_truths.append(ground_truth)
        scan_nums.append(extract_number(file_path))

        if files_processed == 4:
            pred = evaluate_prediction(total_tally)
            confusion[ground_truth, pred] += 1
            print(f"Scans {scan_nums} T:{ground_truth} P:{pred}", total_tally)
            labels.append(ground_truth)
            predictions.append(pred)
            total_tally = None
            files_processed = 0
            ground_truths = []
            scan_nums = []

    if files_processed > 0:
        pred = evaluate_prediction(total_tally)
        confusion[ground_truth, pred] += 1
        print(f"Scans {scan_nums} T:{ground_truth} P:{pred}", total_tally)
        labels.append(ground_truth)
        predictions.append(pred)

    accuracy, precision, recall, f1, (TP, TN, FP, FN) = compute_metrics(labels, predictions)
    print(f"TESTING Classification Metrics:")
    print(f"Accuracy: {accuracy:.4f} F1 Score: {f1:.4f} Precision: {precision:.4f} Recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    # return accuracy, precision, recall, f1, (TP, TN, FP, FN)

    return confusion


def compute_metrics(y_true, y_pred):
    # convert tensors
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
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
    return accuracy, precision, recall, f1, (TP, TN, FP, FN)



################################################################################# 
# MAIN
################################################################################# 


folder_path = "/scratch/k/khalvati/liufeli2/LLMs/data/single_slice_glioma_tests"
class_map = {"LGG":0, "HGG":1, "NG":2, "Unknown":3, 0:"LGG", 1:"HGG", 2:"NG", 3:"Unknown"}
file_paths = get_file_paths_in_folder(folder_path)

confusion = RUN_FOR_SCANS()
# confusion = RUN_FOR_PATIENTS()

print()
print(confusion)