import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader

from app.dataloader.dataloader import UAVDataset
from app.model.basic.models.cnn import CNN
from app.model.basic.models.cnn_lstm import CNNLSTM
from app.model.basic.models.cnn_lstm_residual import CNNLSTMResidual
from app.model.basic.models.lstm import LSTMModel
from app.model.basic.models.transformer import TransformerModel
from app.trainer.trainer import Trainer


def train_model(file_path,
                variant='none',
                mode='cyber',
                num_epochs=10,
                batch_size=256,
                learning_rate=0.001,
                split=[0.7, 0.2, 0.1],
                num_layers=8,
                hidden_dimension=512,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                save_dir='results',  # New parameter
                model_save_path='best_model.pth'):
    import os  # Import here or at the top of the script

    # Create save_dir if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Update model_save_path to be inside save_dir
    model_save_path = os.path.join(save_dir, model_save_path)

    # Load dataset
    device = torch.device(device)
    dataset = UAVDataset(file_path, mode=mode)

    # Split dataset
    train_size = int(split[0] * len(dataset))
    val_size = int(split[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Create model
    input_dim = dataset.input_dim
    num_classes = dataset.num_classes

    if variant == 'cnn_lstm_residual':
        model = CNNLSTMResidual(input_dim=input_dim, hidden_dim=hidden_dimension, num_layers=num_layers, num_classes=num_classes)
    elif variant == 'cnn':
        model = CNN(input_dim=input_dim, hidden_dim=hidden_dimension, num_layers=num_layers, num_classes=num_classes)
    elif variant == 'cnn_lstm':
        model = CNNLSTM(input_dim=input_dim, hidden_dim=hidden_dimension, num_layers=num_layers, num_classes=num_classes)
    elif variant == 'lstm':
        model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dimension, num_layers=num_layers, num_classes=num_classes)
    elif variant == 'transformer':
        model = TransformerModel(input_dim=input_dim, num_classes=num_classes)
    else:
        raise NotImplementedError(f"Variant '{variant}' is not implemented.")

    # Loss function, optimizer, and lr scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    # Trainer
    trainer = Trainer(model,
                      criterion,
                      optimizer,
                      scheduler,
                      model_save_path=model_save_path,
                      save_dir=save_dir,  # Pass save_dir to Trainer
                      device=device)

    # Train and evaluate
    trainer.train(train_loader, val_loader, num_epochs)

    # Test
    accuracy, precision, recall, f1 = trainer.test(test_loader)
    print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")



if __name__ == '__main__':
    pathList = ['/home/shengguang/PycharmProjects/uav_security/app/data/cyber_ready.csv',
            '/home/shengguang/PycharmProjects/uav_security/app/data/physical_ready.csv',
            '/home/shengguang/PycharmProjects/uav_security/app/data/fuse.csv'
            ]
    dataType = ['cyber', 'physical', 'fuse']

    for i in range(len(pathList)):
        datatype = dataType[i]
        path = pathList[i]
        variant = 'lstm'
        train_model(path,
                    variant=variant,
                    num_epochs=100,
                    hidden_dimension=512,
                    num_layers=8,
                    model_save_path='best_model_cyber.pth',
                    save_dir=f'/home/shengguang/PycharmProjects/uav_security/app/outputs/{datatype}_{variant}'
                    )
        variant = 'cnn_lstm_residual'
        train_model(path,
                    variant=variant,
                    num_epochs=100,
                    hidden_dimension=512,
                    num_layers=8,
                    model_save_path='best_model_cyber.pth',
                    save_dir=f'/home/shengguang/PycharmProjects/uav_security/app/outputs/{datatype}_{variant}'
                    )

        variant = 'cnn'
        train_model(path,
                    variant=variant,
                    num_epochs=100,
                    hidden_dimension=512,
                    num_layers=8,
                    model_save_path='best_model_cyber.pth',
                    save_dir=f'/home/shengguang/PycharmProjects/uav_security/app/outputs/{datatype}_{variant}'
                    )
        variant = 'cnn_lstm'
        train_model(path,
                    variant=variant,
                    num_epochs=100,
                    hidden_dimension=512,
                    num_layers=8,
                    model_save_path='best_model_cyber.pth',
                    save_dir=f'/home/shengguang/PycharmProjects/uav_security/app/outputs/{datatype}_{variant}'
                    )

        variant = 'transformer'
        train_model(path,
                    variant=variant,
                    num_epochs=100,
                    hidden_dimension=512,
                    num_layers=8,
                    model_save_path='best_model_cyber.pth',
                    save_dir=f'/home/shengguang/PycharmProjects/uav_security/app/outputs/{datatype}_{variant}'
                    )


