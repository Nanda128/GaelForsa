import torch
import numpy as np
from scada_tcn_model import SCADATCNModel, SCADATCNTrainer
from data_loader import create_data_loaders
import argparse
import os

def train_model(num_epochs: int = 50, batch_size: int = 32, learning_rate: float = 1e-3,
               hidden_channels: int = 64, num_tcn_layers: int = 4, save_path: str = 'scada_tcn_model.pth'):
    """
    Train the SCADA TCN model.

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        hidden_channels: TCN hidden width D
        num_tcn_layers: Number of TCN layers
        save_path: Path to save trained model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader = create_data_loaders(
        batch_size=batch_size,
        window_length=100,  # L
        forecast_horizon=6   # K
    )


    sample_batch = next(iter(train_loader))
    num_features = sample_batch['x'].shape[1]  # F
    num_regime_flags = sample_batch['flags'].shape[1]  # R
    num_fault_classes = 5  # C

    print(f"Features: {num_features}, Regime flags: {num_regime_flags}, Fault classes: {num_fault_classes}")


    model = SCADATCNModel(
        num_features=num_features,
        num_regime_flags=num_regime_flags,
        hidden_channels=hidden_channels,
        num_tcn_layers=num_tcn_layers,
        forecast_horizon=6,
        num_fault_classes=num_fault_classes
    )


    trainer = SCADATCNTrainer(model, device)
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            losses = trainer.train_step(batch)
            train_losses.append(losses)
        avg_train_losses = {
            k: np.mean([loss[k] for loss in train_losses])
            for k in train_losses[0].keys()
        }

        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
    
                x = batch['x'].to(device)
                m_miss = batch['m_miss'].to(device)
                flags = batch['flags'].to(device)
                y_true = batch['y_true'].to(device)
                y_fault = batch['y_fault'].squeeze().to(device)

                m_mask = torch.ones_like(m_miss)
                x_corrupt = x

                x_hat, y_hat, p_fault = model(x_corrupt, m_miss, m_mask, flags)

                l_rec = torch.tensor(0.0)  
                l_pred = trainer.forecast_loss(y_hat, y_true)
                l_fault = trainer.fault_loss(p_fault, y_fault)

                val_losses.append({
                    'l_rec': 0.0,
                    'l_pred': l_pred.item(),
                    'l_fault': l_fault.item(),
                    'total': (0.3 * l_pred + 0.2 * l_fault).item()
                })
        avg_val_losses = {
            k: np.mean([loss[k] for loss in val_losses])
            for k in val_losses[0].keys()
        }

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train - Total: {avg_train_losses['total']:.4f}, "
              f"Rec: {avg_train_losses['l_rec']:.4f}, "
              f"Pred: {avg_train_losses.get('l_pred', 0):.4f}, "
              f"Fault: {avg_train_losses.get('l_fault', 0):.4f}")
        print(f"Val   - Total: {avg_val_losses['total']:.4f}, "
              f"Pred: {avg_val_losses['l_pred']:.4f}, "
              f"Fault: {avg_val_losses['l_fault']:.4f}")

        if avg_val_losses['total'] < best_val_loss:
            best_val_loss = avg_val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'train_losses': avg_train_losses,
                'val_losses': avg_val_losses,
                'config': {
                    'num_features': num_features,
                    'num_regime_flags': num_regime_flags,
                    'hidden_channels': hidden_channels,
                    'num_tcn_layers': num_tcn_layers
                }
            }, save_path)
            print(f"Saved best model to {save_path}")

    print("Training completed")
def load_and_test_model(model_path: str = 'scada_tcn_model.pth'):
    """Load trained model and run inference test."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint['config']
    model = SCADATCNModel(
        num_features=config['num_features'],
        num_regime_flags=config['num_regime_flags'],
        hidden_channels=config['hidden_channels'],
        num_tcn_layers=config['num_tcn_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    trainer = SCADATCNTrainer(model, device)
    _, test_loader = create_data_loaders(batch_size=1)
    test_batch = next(iter(test_loader))
    results = trainer.predict(
        test_batch['x'], test_batch['m_miss'], test_batch['flags']
    )

    print("Inference test results:")
    print(f"Reconstruction shape: {results['x_hat'].shape}")
    print(f"Forecast shape: {results['y_hat'].shape}")
    print(f"Fault probs shape: {results['p_fault'].shape}")
    print(f"Top fault class: {torch.argmax(results['p_fault'], dim=1).item()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SCADA TCN Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden channels')
    parser.add_argument('--layers', type=int, default=4, help='TCN layers')
    parser.add_argument('--save_path', type=str, default='scada_tcn_model.pth', help='Model save path')
    parser.add_argument('--test', action='store_true', help='Run inference test instead of training')

    args = parser.parse_args()

    if args.test:
        load_and_test_model(args.save_path)
    else:
        train_model(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            hidden_channels=args.hidden,
            num_tcn_layers=args.layers,
            save_path=args.save_path
        )

