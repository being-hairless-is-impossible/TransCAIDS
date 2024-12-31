from tqdm import tqdm
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm  # Import tqdm for progress bars

class Trainer:
    def __init__(self, model, criterion, optimizer, warmup_scheduler, plateau_scheduler,
                 decay_scheduler, early_stopping_patience, model_save_path, device,
                 save_dir):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.plateau_scheduler = plateau_scheduler
        self.decay_scheduler = decay_scheduler
        self.early_stopping_patience = early_stopping_patience
        self.model_save_path = model_save_path  # This will be updated to save in models directory
        self.device = device
        self.save_dir = save_dir

        # Initialize lists to store metrics per epoch
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.val_f1s = []
        self.val_accuracies = []
        self.val_precisions = []
        self.val_recalls = []

        # Create subdirectories in save_dir
        if self.save_dir:
            self.logs_dir = os.path.join(self.save_dir, 'logs')
            self.models_dir = os.path.join(self.save_dir, 'models')
            self.plots_dir = os.path.join(self.save_dir, 'plots')
            os.makedirs(self.logs_dir, exist_ok=True)
            os.makedirs(self.models_dir, exist_ok=True)
            os.makedirs(self.plots_dir, exist_ok=True)
            # Update model_save_path to save in models directory
            self.model_save_path = os.path.join(self.models_dir, self.model_save_path)
        else:
            self.logs_dir = None
            self.models_dir = None
            self.plots_dir = None

    def train(self, train_loader, val_loader, num_epochs):
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            self.epochs.append(epoch + 1)

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []

            # Use tqdm for training progress bar
            train_loop = tqdm(train_loader, desc=f'\n Epoch [{epoch + 1}/{num_epochs}] Training')

            for data in train_loop:
                inputs_cyber, inputs_physical, labels = data
                inputs_cyber = inputs_cyber.to(self.device)
                inputs_physical = inputs_physical.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs_cyber, inputs_physical)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs_cyber.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

                # Update tqdm description
                train_loop.set_postfix(loss=loss.item())

            # Calculate average training loss
            train_loss = train_loss / len(train_loader.dataset)
            # Calculate training metrics (if needed)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []

            # Use tqdm for validation progress bar
            val_loop = tqdm(val_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}] Validation')

            with torch.no_grad():
                for data in val_loop:
                    inputs_cyber, inputs_physical, labels = data
                    inputs_cyber = inputs_cyber.to(self.device)
                    inputs_physical = inputs_physical.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs_cyber, inputs_physical)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item() * inputs_cyber.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

                    # Update tqdm description
                    val_loop.set_postfix(loss=loss.item())

            # Calculate average validation loss and metrics
            val_loss = val_loss / len(val_loader.dataset)
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_precision = precision_score(val_labels, val_preds, average='macro', zero_division=0)
            val_recall = recall_score(val_labels, val_preds, average='macro', zero_division=0)
            val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            self.val_precisions.append(val_precision)
            self.val_recalls.append(val_recall)
            self.val_f1s.append(val_f1)

            # Adjust learning rate
            if self.warmup_scheduler and epoch < self.warmup_scheduler.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.plateau_scheduler.step(val_loss)
                self.decay_scheduler.step()

            # Early stopping and model checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save the best model
                torch.save(self.model.state_dict(), self.model_save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stopping_patience:
                    print("Early stopping...")
                    break

            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Val F1: {val_f1:.4f}, Val Acc: {val_accuracy:.4f}, "
                  f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")

        print("Training complete.")

    def test(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data in test_loader:
                inputs_cyber, inputs_physical, labels = data
                inputs_cyber = inputs_cyber.to(self.device)
                inputs_physical = inputs_physical.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs_cyber, inputs_physical)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        return accuracy, precision, recall, f1

    def save_training_log(self):
        if self.logs_dir:
            log_path = os.path.join(self.logs_dir, 'training_log.csv')
        else:
            log_path = 'training_log.csv'

        # Create DataFrame with the required metrics
        df = pd.DataFrame({
            'epoch': self.epochs,
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'val_f1': self.val_f1s,
            'val_accuracy': self.val_accuracies,
            'val_precision': self.val_precisions,
            'val_recall': self.val_recalls
        })
        df.to_csv(log_path, index=False)
        print(f"Training log saved to {log_path}")

    def plot_metrics(self):
        if self.plots_dir:
            loss_plot_path = os.path.join(self.plots_dir, 'loss_plot.png')
            f1_plot_path = os.path.join(self.plots_dir, 'f1_plot.png')
            acc_plot_path = os.path.join(self.plots_dir, 'accuracy_plot.png')
        else:
            loss_plot_path = 'loss_plot.png'
            f1_plot_path = 'f1_plot.png'
            acc_plot_path = 'accuracy_plot.png'

        # Plot Loss
        plt.figure()
        plt.plot(self.epochs, self.train_losses, label='Train Loss')
        plt.plot(self.epochs, self.val_losses, label='Validation Loss')
        plt.legend()
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Loss plot saved to {loss_plot_path}")

        # Plot F1 Score
        plt.figure()
        plt.plot(self.epochs, self.val_f1s, label='Validation F1 Score')
        plt.legend()
        plt.title('Validation F1 Score over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.savefig(f1_plot_path)
        plt.close()
        print(f"F1 Score plot saved to {f1_plot_path}")

        # Plot Accuracy
        plt.figure()
        plt.plot(self.epochs, self.val_accuracies, label='Validation Accuracy')
        plt.legend()
        plt.title('Validation Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig(acc_plot_path)
        plt.close()
        print(f"Accuracy plot saved to {acc_plot_path}")
