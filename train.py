import time
t0 = time.time()
accumulated_training_time = 0
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from resnet18 import ResNet, BasicBlock
from resnet18_torchvision import build_model
from training_utils import train, validate
from utils import save_plots, get_data

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', default='scratch',
    help='choose model built from scratch or the Torchvision model',
    choices=['scratch', 'torchvision']
)
parser.add_argument(
    '--profiling', action="store_true", 
    default=False, help="profile one batch"
)
parser.add_argument(
    '--epoch', default=10
)
args = vars(parser.parse_args())

# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Learning and training parameters.
epochs = int(args['epoch'])
batch_size = 64
learning_rate = 0.01
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_loader, valid_loader = get_data(batch_size=batch_size)

# Define model based on the argument parser string.
if args['model'] == 'scratch':
    # print('[INFO]: Training ResNet18 built from scratch...')
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=10).to(device)
    plot_name = 'resnet_scratch'
if args['model'] == 'torchvision':
    # print('[INFO]: Training the Torchvision ResNet18 model...')
    model = build_model(pretrained=False, fine_tune=True, num_classes=10).to(device) 
    plot_name = 'resnet_torchvision'
# # print(model)

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
# print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
# print(f"{total_trainable_params:,} training parameters.")

# Optimizer.
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# Loss function.
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    t1 = time.time()
    print("[profiling] init time: {}s".format(t1-t0))
    for epoch in range(epochs):
        # print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc, accumulated_training_time = train(
            model, 
            train_loader, 
            optimizer, 
            criterion,
            device,
            epoch,
            accumulated_training_time
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, 
            valid_loader, 
            criterion,
            device
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        # print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        # print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        # print('-'*50)
        
    # Save the loss and accuracy plots.
    save_plots(
        train_acc, 
        valid_acc, 
        train_loss, 
        valid_loss, 
        name=plot_name
    )
    # print('TRAINING COMPLETE')