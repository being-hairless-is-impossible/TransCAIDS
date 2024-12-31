import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import copy
import os  # Import os for directory operations
from tqdm import tqdm
import torch.nn.init as init

class Trainer:
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 warmup_scheduler=None,
                 early_stopping_patience=50,
                 model_save_path="best_model.pth",
                 save_dir="results",  # New parameter
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        self.model = model.to(self.device)  # Move model to GPU/CPU

        if warmup_scheduler is not None:
            use_warmup = True

        # Apply Kaiming initialization
        self.apply_kaiming_init()

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience  # Default set to 50
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        self.model_save_path = model_save_path

        # Create save_dir and subdirectories
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.models_dir = os.path.join(self.save_dir, 'models')
        self.plots_dir = os.path.join(self.save_dir, 'plots')
        self.logs_dir = os.path.join(self.save_dir, 'logs')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Update model_save_path to models_dir
        self.model_save_path = os.path.join(self.models_dir, os.path.basename(self.model_save_path))

        # Track losses and metrics
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        self.val_accuracy_scores = []
        self.val_precision_scores = []
        self.val_recall_scores = []

        # Initialize CSV log file
        self.csv_log_path = os.path.join(self.logs_dir, 'training_logs.csv')
        with open(self.csv_log_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,val_f1,val_accuracy,val_precision,val_recall\n')

        # Optional: Initialize a list to store metrics for Pandas DataFrame
        self.metrics = []

    def apply_kaiming_init(self):
        """
        Apply Kaiming initialization (He initialization) to the layers in the model.
        This ensures proper weight initialization for ReLU activations.
        """

        def kaiming_init(layer):
            if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

        self.model.apply(kaiming_init)

    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            # Add tqdm for training progress
            train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # Move data to GPU/CPU
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            val_loss, val_f1, val_accuracy, val_precision, val_recall = self.evaluate(val_loader)
            self.scheduler.step(val_loss)

            train_loss = running_loss / len(train_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            self.val_accuracy_scores.append(val_accuracy)
            self.val_precision_scores.append(val_precision)
            self.val_recall_scores.append(val_recall)

            # Append metrics to the list
            self.metrics.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_f1': val_f1,
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall
            })

            # Log metrics to CSV
            with open(self.csv_log_path, 'a') as f:
                f.write(f"{epoch + 1},{train_loss:.4f},{val_loss:.4f},{val_f1:.4f},{val_accuracy:.4f},{val_precision:.4f},{val_recall:.4f}\n")

            print(
                f"\n Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val F1: {val_f1:.4f}, "
                f"Val Acc: {val_accuracy:.4f}, "
                f"Val Prec: {val_precision:.4f}, "
                f"Val Rec: {val_recall:.4f}"
            )

            # Early stopping and model saving
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), self.model_save_path)  # Save the model
                print(f"Model saved at epoch {epoch + 1}")
                self.early_stop_counter = 0  # Reset the counter if improvement occurs
            else:
                self.early_stop_counter += 1
                print(
                    f"No improvement in validation loss. Early stop counter: {self.early_stop_counter}/{self.early_stopping_patience}"
                )

                if self.early_stop_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {self.early_stopping_patience} epochs with no improvement!")
                    break

        self.model.load_state_dict(self.best_model_wts)
        self.plot_metrics()

        # Save the metrics DataFrame
        df_metrics = pd.DataFrame(self.metrics)
        df_metrics_path = os.path.join(self.logs_dir, 'training_metrics.csv')
        df_metrics.to_csv(df_metrics_path, index=False)
        print(f"Training metrics saved to {df_metrics_path}")

    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []

        # Add tqdm for validation progress
        loader = tqdm(loader, desc="Validation", unit="batch")

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # Move data to GPU/CPU
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        avg_loss = running_loss / len(loader)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        return avg_loss, f1, accuracy, precision, recall

    def test(self, test_loader):
        self.model.eval()
        all_labels = []
        all_preds = []

        # Add tqdm for testing progress
        test_loader = tqdm(test_loader, desc="Testing", unit="batch")

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # Move data to GPU/CPU
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return accuracy, precision, recall, f1

    def plot_metrics(self):
        epochs = range(1, len(self.train_losses) + 1)

        # Plot Training and Validation Loss
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        loss_plot_path = os.path.join(self.plots_dir, 'loss_plot.png')
        plt.savefig(loss_plot_path)
        plt.close()  # Close the figure to free memory
        print(f"Loss plot saved to {loss_plot_path}")

        # Plot Validation F1 Score
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.val_f1_scores, label='Validation F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.title('Validation F1 Score')
        plt.legend()
        f1_plot_path = os.path.join(self.plots_dir, 'f1_score_plot.png')
        plt.savefig(f1_plot_path)
        plt.close()
        print(f"F1 Score plot saved to {f1_plot_path}")

        # Plot Validation Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.val_accuracy_scores, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        acc_plot_path = os.path.join(self.plots_dir, 'accuracy_plot.png')
        plt.savefig(acc_plot_path)
        plt.close()
        print(f"Accuracy plot saved to {acc_plot_path}")

        # Plot Validation Precision
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.val_precision_scores, label='Validation Precision')
        plt.xlabel('Epochs')
        plt.ylabel('Precision')
        plt.title('Validation Precision')
        plt.legend()
        prec_plot_path = os.path.join(self.plots_dir, 'precision_plot.png')
        plt.savefig(prec_plot_path)
        plt.close()
        print(f"Precision plot saved to {prec_plot_path}")

        # Plot Validation Recall
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.val_recall_scores, label='Validation Recall')
        plt.xlabel('Epochs')
        plt.ylabel('Recall')
        plt.title('Validation Recall')
        plt.legend()
        rec_plot_path = os.path.join(self.plots_dir, 'recall_plot.png')
        plt.savefig(rec_plot_path)
        plt.close()
        print(f"Recall plot saved to {rec_plot_path}")
