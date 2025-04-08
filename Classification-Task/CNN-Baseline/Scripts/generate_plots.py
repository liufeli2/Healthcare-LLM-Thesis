import matplotlib.pyplot as plt

# Read and parse the data from the text file
def parse_log_file(filepath):
    epochs = []
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith("Epoch"):
                parts = line.strip().split(", ")
                epoch = int(parts[0].split("[")[1].split("/")[0])
                train_loss = float(parts[1].split(": ")[1])
                train_accuracy = float(parts[2].split(": ")[1])
                val_loss = float(parts[3].split(": ")[1])
                val_accuracy = float(parts[4].split(": ")[1])
                
                epochs.append(epoch)
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
    
    return epochs, train_losses, train_accuracies, val_losses, val_accuracies

# Plot the data
def plot_metrics(epochs, train_losses, train_accuracies, val_losses, val_accuracies):
    plt.figure(figsize=(12, 5))
    
    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Main
file_path = "/Users/felicialiu/Desktop/ESC499/Code/photo_processing/plot_data.txt"
epochs, train_losses, train_accuracies, val_losses, val_accuracies = parse_log_file(file_path)
plot_metrics(epochs, train_losses, train_accuracies, val_losses, val_accuracies)
