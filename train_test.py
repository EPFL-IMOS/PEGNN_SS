import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import NAdam

def train_gnn_model(model, Train_DATA, Validation_DATA, window_size, device, EPOCHS, lr, patience = 200):
    
    # Initialize model and optimizer
    optimizer = NAdam(model.parameters(), lr=lr)
    
    # Training parameters
    best_loss = float('inf')
    best_model = None
    counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        # Training loop
        for idx, data in enumerate(Train_DATA):
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out.squeeze(), data.y.float().view(-1, 37, window_size).permute(0, 2, 1).squeeze())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_losses = np.array(train_losses)
        recon_epoch_loss = np.sqrt(train_losses.mean())
        total_epoch_loss = recon_epoch_loss

        # Print progress at every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch: {epoch+1}")
            print('Training Loss:', recon_epoch_loss)

        val_losses = []

        # Validation loop
        val_loss = 0
        with torch.no_grad():
            for idx, data in enumerate(Validation_DATA):
                out = model(data)
                val_loss = F.mse_loss(out.squeeze(), data.y.float().view(-1, 37, window_size).permute(0, 2, 1).squeeze())
                val_losses.append(val_loss.item())

        val_losses = np.array(val_losses)
        recon_epoch_loss_eval = np.sqrt(val_losses.mean())
        total_epoch_loss_eval = recon_epoch_loss_eval

        # Print progress at every 10 epochs
        if (epoch + 1) % 10 == 0:
            print('Validation Loss:', recon_epoch_loss_eval)

        # Early stopping
        if total_epoch_loss_eval < best_loss:
            best_loss = total_epoch_loss_eval
            best_model = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping')
                break

    # Load the best model
    model.load_state_dict(best_model)
    
    return model


def test_gnn_model(model, Test_DATA, window_size, device):
    model.to(device)
    model.eval()
    preds_list = []
    targets_list = []
    cost = 0
    
    with torch.inference_mode():
        for idx, data in enumerate(Test_DATA):
            data = data.to(device)
            y_pred = model(data).squeeze()
            preds_list.append(y_pred.cpu().numpy())
            targets_list.append(data.y.float().view(-1, 37, window_size).permute(0, 2, 1).squeeze().cpu().numpy())
            cost += torch.mean((y_pred - data.y.float().view(-1, 37, window_size).permute(0, 2, 1).squeeze())**2).item()
    
    cost = cost / (idx + 1)
    mse = np.sqrt(cost)
    print(f"MSE: {mse:.6f}")

    return preds_list, targets_list, mse