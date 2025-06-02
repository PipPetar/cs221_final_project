import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score

# ----------------------------
# Classes defined at top-level
# ----------------------------

class GameDataset(Dataset):
    """
    A PyTorch Dataset for loading features and labels from in-memory tensors.
    """
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.X = features
        self.Y = labels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


class NeuralNet(nn.Module):
    """
    A feed-forward neural network with multiple hidden layers and a chosen activation.
    """
    def __init__(self, input_size, hidden_units, activation, dropout):
        super(NeuralNet, self).__init__()

        # Choose the activation
        if activation.lower() == 'mish':
            self.activation = nn.Mish()
        elif activation.lower() == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation.lower() == 'silu':
            self.activation = nn.SiLU()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()

        layers = []
        current_size = input_size

        for hidden_size in hidden_units:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(self.activation)
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.Dropout(dropout))
            current_size = hidden_size

        # Final output layer
        layers.append(nn.Linear(current_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LiveAccuracyPlot:
    """
    A class that tracks and plots training and validation metrics over epochs.
    """
    def __init__(self, validation_loader_plot, model_dir, device, criterion, num_epochs):
        self.val_loader = validation_loader_plot
        self.epoch_data = []
        self.train_loss_data = []
        self.val_loss_data = []
        self.val_accuracy_data = []
        self.best_val_accuracy = 0
        self.best_val_accuracy_epoch = 0

        self.model_dir = model_dir
        self.device = device
        self.criterion = criterion
        self.num_epochs = num_epochs

        # Initialize the figure for plotting
        self.fig, self.axs = plt.subplots(1, 2, figsize=(15, 5))
        self.fig.suptitle("Training Progress")

    def _plot_current_progress(self):
        """
        Internal helper to clear and redraw the plot.
        """
        self.axs[0].clear()
        self.axs[1].clear()

        # Plot training and validation loss
        self.axs[0].plot(self.epoch_data, self.train_loss_data, label='Training Loss', color='green')
        self.axs[0].plot(self.epoch_data, self.val_loss_data, label='Validation Loss', color='orange')
        self.axs[0].set_xlabel('Epoch')
        self.axs[0].set_ylabel('Loss')
        self.axs[0].legend()
        self.axs[0].set_title('Loss Over Epochs')

        # Plot validation accuracy
        self.axs[1].plot(self.epoch_data, self.val_accuracy_data, label='Validation Accuracy', color='blue')
        if self.epoch_data:
            self.axs[1].scatter(
                self.best_val_accuracy_epoch,
                self.best_val_accuracy,
                color='red', s=50, zorder=5,
                label=f'Best Val Acc: {self.best_val_accuracy:.2%}'
            )
        self.axs[1].set_xlabel('Epoch')
        self.axs[1].set_ylabel('Accuracy')
        self.axs[1].legend()
        self.axs[1].set_title('Validation Accuracy Over Epochs')

        self.fig.tight_layout()
        # Brief pause to update the plot, especially if running interactively
        plt.pause(0.001)

    def update(self, epoch, current_epoch_train_loss, model_to_eval):
        """
        Updates the tracking data for the current epoch, evaluates on validation set,
        saves best model, and prints to console.
        """
        model_to_eval.eval()
        current_val_epoch_loss = 0
        correct_val_epoch = 0
        total_val_epoch = 0

        with torch.no_grad():
            for batch_X_val, batch_Y_val in self.val_loader:
                batch_X_val, batch_Y_val = batch_X_val.to(self.device), batch_Y_val.to(self.device)
                outputs_val = model_to_eval(batch_X_val)
                loss_val = self.criterion(outputs_val, batch_Y_val)
                current_val_epoch_loss += loss_val.item() * batch_X_val.size(0)

                preds_val = (outputs_val > 0.5).float()
                correct_val_epoch += (preds_val == batch_Y_val).sum().item()
                total_val_epoch += batch_Y_val.size(0)

        avg_epoch_val_loss = (
            current_val_epoch_loss / len(self.val_loader.dataset)
            if len(self.val_loader.dataset) > 0 else 0
        )
        current_epoch_val_accuracy = (
            correct_val_epoch / total_val_epoch
            if total_val_epoch > 0 else 0
        )

        self.epoch_data.append(epoch)
        self.train_loss_data.append(current_epoch_train_loss)
        self.val_loss_data.append(avg_epoch_val_loss)
        self.val_accuracy_data.append(current_epoch_val_accuracy)

        # Save best model if improved
        if current_epoch_val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = current_epoch_val_accuracy
            self.best_val_accuracy_epoch = epoch
            print(f"Epoch {epoch+1}: New best validation accuracy: {self.best_val_accuracy:.4f}. Saving model...")
            torch.save(model_to_eval.state_dict(), os.path.join(self.model_dir, 'best_validation_model.pth'))

        self._plot_current_progress()

        print(
            f'Epoch {epoch+1}/{self.num_epochs} -> '
            f'Train Loss: {current_epoch_train_loss:.4f}, '
            f'Val Loss: {avg_epoch_val_loss:.4f}, '
            f'Val Acc: {current_epoch_val_accuracy:.4f}'
        )

    def finalize(self):
        """
        Generates the final plot, shows it, and prints final summary.
        """
        self._plot_current_progress()
        plt.show()
        print(
            f"Training complete. Best Validation Accuracy: {self.best_val_accuracy:.4f} "
            f"at epoch {self.best_val_accuracy_epoch+1}."
        )


def main():
    # ============================
    # Settings and Hyperparameters
    # ============================

    # Paths and directories
    MODEL_DIR = 'Grokfast_pytorch_models_colab'
    TRAIN_DATA_PATH = 'final_assembled_game_data2.1.csv'
    TEST_DATA_PATH = 'final_assembled_test_data2.1.csv'

    # Model hyperparameters
    INPUT_SIZE = 2000
    HIDDEN_UNITS = [512, 256, 128, 64]
    ACTIVATION_FUNCTION = 'gelu'
    DROPOUT_RATE = 0.25

    # Training hyperparameters
    BATCH_SIZE = 256
    NUM_EPOCHS = 100
    LEARNING_RATE = 5e-6
    WEIGHT_DECAY = 0.01
    VALIDATION_SPLIT = 0.1

    # Grokfast (EMA) hyperparameters
    LAMBDA_EMA = 2.0
    ALPHA_EMA = 0.88

    # Device config
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {DEVICE}')

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(42)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # ============================
    # Data Loading
    # ============================
    try:
        train_dataset_np = np.loadtxt(TRAIN_DATA_PATH, delimiter=',')
        test_dataset_np = np.loadtxt(TEST_DATA_PATH, delimiter=',')
    except FileNotFoundError:
        print("ERROR: CSV files not found. Check paths.")
        return

    # Split into features and labels
    x_train_full = train_dataset_np[:, :INPUT_SIZE].astype(np.float32)
    y_train_full = train_dataset_np[:, INPUT_SIZE].astype(np.float32)

    x_test = test_dataset_np[:, :INPUT_SIZE].astype(np.float32)
    y_test = test_dataset_np[:, INPUT_SIZE].astype(np.float32)

    # Convert to tensors
    x_train_full_tensor = torch.from_numpy(x_train_full)
    y_train_full_tensor = torch.from_numpy(y_train_full).unsqueeze(1)
    x_test_tensor = torch.from_numpy(x_test)
    y_test_tensor = torch.from_numpy(y_test).unsqueeze(1)

    # Create the PyTorch Dataset
    full_dataset = GameDataset(x_train_full_tensor, y_train_full_tensor)
    train_size = int((1 - VALIDATION_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create DataLoaders
    # If multiprocessing issues persist on Windows, set num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_dataset = GameDataset(x_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    # ============================
    # Initialize Model
    # ============================
    model = NeuralNet(INPUT_SIZE, HIDDEN_UNITS, ACTIVATION_FUNCTION, DROPOUT_RATE).to(DEVICE)
    print(model)

    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    # LiveAccuracyPlot instance
    live_plot = LiveAccuracyPlot(val_loader, MODEL_DIR, DEVICE, criterion, NUM_EPOCHS)

    # Prepare EMA gradient arrays
    ema_gradients = [torch.zeros_like(p.data) for p in model.parameters() if p.requires_grad]
    use_ema_grad = len(ema_gradients) > 0

    if use_ema_grad:
        print("Using EMA gradient modification.")
    else:
        print("Not using EMA gradient modification.")

    # ============================
    # Training Loop
    # ============================
    print("Starting training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_epoch_train_loss = 0.0

        for batch_idx, (batch_X, batch_Y) in enumerate(train_loader):
            batch_X, batch_Y = batch_X.to(DEVICE), batch_Y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()

            # Apply EMA-based gradient modification
            if use_ema_grad:
                with torch.no_grad():
                    param_idx = 0
                    for param_group in optimizer.param_groups:
                        for param in param_group['params']:
                            if param.requires_grad and param.grad is not None:
                                ema_gradients[param_idx] = (
                                    ALPHA_EMA * ema_gradients[param_idx]
                                    + (1 - ALPHA_EMA) * param.grad
                                )
                                modified_grad = param.grad + LAMBDA_EMA * ema_gradients[param_idx]
                                param.grad.copy_(modified_grad)
                                param_idx += 1

            optimizer.step()
            running_epoch_train_loss += loss.item() * batch_X.size(0)

        epoch_train_loss = (
            running_epoch_train_loss / len(train_loader.dataset)
            if len(train_loader.dataset) > 0 else 0
        )
        live_plot.update(epoch, epoch_train_loss, model)
        scheduler.step()

    live_plot.finalize()
    print("Training finished.")

    # Save the final epoch model
    final_epoch_model_path = os.path.join(MODEL_DIR, 'final_epoch_model.pth')
    torch.save(model.state_dict(), final_epoch_model_path)
    print(f"Model from final epoch saved to {final_epoch_model_path}")

    # ============================
    # Evaluate on Test Set
    # ============================
    print("\nEvaluating on the test set using the best validation model...")
    best_validation_model_path = os.path.join(MODEL_DIR, 'best_validation_model.pth')

    eval_model = NeuralNet(INPUT_SIZE, HIDDEN_UNITS, ACTIVATION_FUNCTION, DROPOUT_RATE).to(DEVICE)
    if os.path.exists(best_validation_model_path):
        eval_model.load_state_dict(torch.load(best_validation_model_path, map_location=DEVICE))
        print(f"Loaded best validation model from {best_validation_model_path}")
    else:
        print(f"ERROR: Best validation model not found at {best_validation_model_path}.")
        # Fallback option:
        # eval_model.load_state_dict(torch.load(final_epoch_model_path, map_location=DEVICE))

    eval_model.eval()
    y_pred_probs_list = []
    y_true_list = []

    with torch.no_grad():
        for batch_X_test, batch_Y_test in test_loader:
            batch_X_test, batch_Y_test = batch_X_test.to(DEVICE), batch_Y_test.to(DEVICE)
            outputs_test = eval_model(batch_X_test)
            y_pred_probs_list.extend(outputs_test.cpu().numpy().flatten())
            y_true_list.extend(batch_Y_test.cpu().numpy().flatten())

    if y_true_list:
        y_pred_test = (np.array(y_pred_probs_list) > 0.5).astype(int)
        y_true_test = np.array(y_true_list).astype(int)
        test_accuracy = accuracy_score(y_true_test, y_pred_test)
        print(f"Test Set Accuracy (using best validation model): {test_accuracy:.4f}")

        from sklearn.metrics import confusion_matrix, classification_report
        print("\nConfusion Matrix (Test Set):")
        print(confusion_matrix(y_true_test, y_pred_test))

        print("\nClassification Report (Test Set):")
        if len(np.unique(y_pred_test)) > 1 and len(np.unique(y_true_test)) > 1:
            print(classification_report(y_true_test, y_pred_test, target_names=['Loss (0)', 'Win (1)'], zero_division=0))
        else:
            print("Classification report cannot be generated due to only one class present.")
    else:
        print("No data processed from the test_loader. Cannot evaluate.")


if __name__ == '__main__':
    # Required on Windows to avoid freezing issues with multiprocessing
    multiprocessing.freeze_support()

    # Run main
    main()
