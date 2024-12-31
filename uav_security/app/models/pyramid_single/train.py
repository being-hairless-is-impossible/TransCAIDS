from typing import List
import os
import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader

from app.model.pyramid_single.pyramid_conv_transformer_single import PyramidTransformer
from app.model.pyramid_single.pyramid_conv_lstm_single import PyramidConvLSTM
from app.trainer.trainer import Trainer
from app.dataloader.dataloader import UAVDataset
from app.trainer.warmup import WarmupScheduler
from typing import List
import os
import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader

from app.model.pyramid_single.pyramid_conv_transformer_single import PyramidTransformer
from app.model.pyramid_single.pyramid_conv_lstm_single import PyramidConvLSTM
from app.model.pyramid_single.pyramid_conv_lstm_kan_lstm_single import PyramidConvLSTMKAN
from app.trainer.trainer import Trainer
from app.dataloader.dataloader import UAVDataset
from app.trainer.warmup import WarmupScheduler


def train_pyramid_model(file_path,
                        variant: str = 'lstm',
                        num_epochs: int = 100,
                        batch_size: int = 128,
                        learning_rate: float = 0.001,
                        use_warmup: bool = False,
                        split: List[float] = [0.7, 0.2, 0.1],
                        hidden_dimension: int = 512,
                        patience: int = 30,
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                        model_save_path: str = 'best_pyramid_model.pth',
                        save_dir: str = None):
    '''
    :param file_path: Path to the dataset CSV file
    :param variant: The variant of the model to use, either 'lstm' or 'transformer'
    :param num_epochs: Number of epochs to train
    :param batch_size: Batch size
    :param learning_rate: Learning rate
    :param split: The split ratio for training, validation, and test sets
    :param hidden_dimension: The hidden dimension size for LSTM or Transformer
    :param patience: The number of epochs to wait before breaking training, based on validation result
    :param device: The device to use for training (e.g., 'cuda' or 'cpu')
    :param model_save_path: Path to save the best model
    :param save_dir: Directory to save logs and plots
    :return:
    '''
    # Create save directory if it doesn't exist
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Load the dataset
    device = torch.device(device)
    dataset = UAVDataset(file_path)

    # Get input dimensions and number of classes
    input_dim = dataset.input_dim
    num_classes = dataset.num_classes

    # Split dataset into training, validation, and test sets
    train_size = int(split[0] * len(dataset))
    val_size = int(split[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Create the model
    if variant == 'lstm':
        model = PyramidConvLSTM(
            input_dim=input_dim,
            num_classes=num_classes,
            num_layers=5,
        )
    elif variant == 'transformer':
        model = PyramidTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
            num_layers=3,
            attention_heads=4,
            dropout=0.5
        )
    elif variant == 'lstm_kan':
        model = PyramidConvLSTMKAN(
            input_dim=input_dim,
            num_classes=num_classes,
            num_layers=5,
        )
    else:
        raise ValueError(f"Invalid variant: {variant}")

    # Print model parameter count
    print('Model Parameter Size:', sum(p.numel() for p in model.parameters()))

    # Loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Composite learning rate scheduler: warmup + plateau + exponential decay
    if use_warmup:
        warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=5, base_lr=learning_rate / 5, final_lr=learning_rate)
    else:
        warmup_scheduler = None
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )
    decay_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # Initialize trainer with model, criterion, optimizers, and schedulers
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=plateau_scheduler,  # Assuming you're using ReduceLROnPlateau
        warmup_scheduler=warmup_scheduler,
        early_stopping_patience=patience,
        model_save_path=model_save_path,
        save_dir=save_dir,
        device=device
    )

    # Train and evaluate
    trainer.train(train_loader, val_loader, num_epochs)

    # Test the model on the test set
    accuracy, precision, recall, f1 = trainer.test(test_loader)
    print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Save test metrics
    if save_dir:
        test_metrics_path = os.path.join(save_dir, 'test_metrics.txt')
    else:
        test_metrics_path = 'test_metrics.txt'
    with open(test_metrics_path, 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test Precision: {precision:.4f}\n")
        f.write(f"Test Recall: {recall:.4f}\n")
        f.write(f"Test F1 Score: {f1:.4f}\n")
    print(f"Test metrics saved to {test_metrics_path}")


if __name__ == '__main__':
    if __name__ == '__main__':
        path = '/data/cyber_ready.csv'
        datatype = 'cyber'


        variant = 'lstm_kan'
        # Training with PyramidConv (LSTM variant)
        train_pyramid_model(
            file_path=path,
            variant=variant,
            num_epochs=100,
            patience=10,
            hidden_dimension=128,
            model_save_path='best_pyramid_model_lstm.pth',
            save_dir=f'/home/shengguang/PycharmProjects/uav_security/app/outputs/{datatype}_pyramid_conv_{variant}'
        )
        #
        # variant = 'transformer'
        # train_pyramid_model(
        #     file_path=path,
        #     variant=variant,
        #     num_epochs=100,
        #     patience=10,
        #     hidden_dimension=128,
        #     model_save_path='best_pyramid_model_transformer.pth',
        #     save_dir=f'/home/shengguang/PycharmProjects/uav_security/app/outputs/{datatype}_pyramid_conv_{variant}'
        # )