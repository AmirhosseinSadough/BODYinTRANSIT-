import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import *
from classification_model import mahcross_n
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

def train(args):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Leg Movement
    feature_idx_uppleg_right_velo   = np.arange(75,78)
    feature_idx_uppleg_left_velo    = np.arange(21,24)
    feature_idx_lowleg_right_velo   = np.arange(93,96)
    feature_idx_lowleg_left_velo    = np.arange(39,42)

    # hip fexion
    feature_idx_hip_right_flexion   = np.arange(81,90)
    feature_idx_hip_left_flexion    = np.arange(27,36)

    # shoulders
    feature_idx_shoulder_right_flexion = np.arange(261,270)
    feature_idx_shoulder_right_pos  = np.arange(234,243)
    feature_idx_shoulder_left_flexion  = np.arange(144,153)
    feature_idx_shoulder_left_pos = np.arange(171,180)

    # 
    feature_idx_LeftKnee_pos = np.arange(48,51)

    # EMG muscles
    feature_idx_emg_muscle= np.arange(0,2)
    # ECG
    feature_idx_ecg = np.arange(2,3)
    
    # all features
    featidx = np.concatenate((feature_idx_uppleg_right_velo+6, feature_idx_uppleg_left_velo+6, feature_idx_lowleg_right_velo+6, feature_idx_lowleg_left_velo+6, 
                                      feature_idx_hip_right_flexion+6, feature_idx_hip_left_flexion+6, feature_idx_shoulder_right_flexion+6, feature_idx_shoulder_right_pos+6,
                                      feature_idx_shoulder_left_flexion+6, feature_idx_shoulder_left_pos+6,feature_idx_LeftKnee_pos+6))



    print("length of features: ", len(featidx))


    train_dataset = torch.load("train_dataset_class.pth", weights_only=False)
    val_dataset = torch.load("val_dataset_class.pth", weights_only=False)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = mahcross_n(num_sensors=len(featidx), d_model=256, num_heads=8, num_layers=2, num_classes=3, rnn_hidden_size=128, dropout_prob=0.2)
    # checkpoint = torch.load('checkpoint_f1_0.4300_epoch_13_dec.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['model_state_dict'])  # or checkpoint['state_dict'] depending on how it's saved
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # Load checkpoint if resuming training
    start_epoch = 0

    # Training loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1_scores, val_f1_scores = [], []

    f1_best = 0.40

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:

            X, Xc, y = batch
            X, Xc, y = X.to(device), Xc.to(device), y.to(device)

            X = X[:, :, :, featidx]
            Xc = Xc[:, :, :, featidx]

            optimizer.zero_grad()
            outputs = model(X, Xc)
            loss = criterion(outputs, y + 1)  # Shift labels by +1 for CrossEntropyLoss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += (predicted == (y + 1)).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{train_loss/train_total:.4f}",
                    "acc": f"{train_correct/train_total:.4f}",
                }
            )

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total
        train_f1 = f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average="macro")
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_f1_scores.append(train_f1)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                X, Xc, y = batch
                X, Xc, y = X.to(device), Xc.to(device), y.to(device)
                
                X = X[:, :, :, featidx]
                Xc = Xc[:, :, :, featidx]

                outputs = model(X, Xc)
                loss = criterion(outputs, y + 1)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += (predicted == (y + 1)).sum().item()

                # Accumulate predictions and targets for F1 score calculation
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend((y + 1).cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_f1 = f1_score(all_targets, all_preds, average="macro")

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)

        print(
            f"Epoch [{epoch+1}/{args.epochs}], "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}"
        )

        if f1_best < val_f1:
            f1_best = val_f1
            print(f"Best F1 score: {f1_best:.4f}")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"checkpoint_f1_{val_f1:.4f}_epoch_{epoch+1}_dec.pth",
            )
    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.plot(val_f1_scores, label="Train F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training and validation accuracy")

    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the baseline model")

    parser.add_argument(
        "--db_folder",
        type=str,
        default="./dataset/MLSP_bit_dataset/train",
        help="the path to the dataset",
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=100,
        help="the size of the windows to extract as inputs for the models",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "--save_every", type=int, default=10, help="Save checkpoint every n epochs"
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    args = parser.parse_args()
    train(args)
