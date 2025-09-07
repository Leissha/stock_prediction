"""
PyTorch Trainer - train, fit, predict, evaluate, save, load models
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from loguru import logger


def get_device(device=None):
    """
    Get the appropriate device for PyTorch operations.
    
    Args:
        device: Device to use (auto-detected if None)
    
    Returns:
        device: PyTorch device object
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def fit(model, x_train, y_train, x_val=None, y_val=None,
        epochs=25, batch_size=32, learning_rate=1e-3,
        patience=5, device=None):
    """
    Train a PyTorch model with early stopping and gradient clipping.
    
    This function implements the complete training loop for time series prediction:
    1. Setup optimizer and loss function
    2. Create data loaders for batch processing
    3. Training loop with forward/backward passes
    4. Validation and early stopping
    5. Save best model weights
    
    Args:
        model: PyTorch model to train
        x_train, y_train: Training data
        x_val, y_val: Validation data (optional)
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for AdamW optimizer
        patience: Early stopping patience
        device: Device to use (auto-detected if None)
    
    Returns:
        Trained model
    """
    # ============================================================================
    # 1. SETUP: Device, Optimizer, and Loss Function
    # ============================================================================
    # Use provided device
    logger.info(f"Training on device: {device}")
    
    # Move model to device (GPU/CPU)
    model.to(device)
    
    # Setup AdamW optimizer with weight decay for regularization
    # AdamW = Adam with decoupled weight decay (better than Adam for most cases)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Mean Squared Error loss for regression (predicting continuous values)
    criterion = torch.nn.MSELoss()
    
    # ============================================================================
    # 2. DATA PREPARATION: Convert to PyTorch tensors and create data loaders
    # ============================================================================
    
    # Convert numpy arrays to PyTorch tensors
    # TensorDataset pairs input (x) and target (y) data together
    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(x_train, dtype=torch.float32),  # Input features
            torch.as_tensor(y_train, dtype=torch.float32).view(-1, 1)  # Target values (reshape to column vector)
        ),
        batch_size=batch_size,      # Number of samples per batch
        shuffle=True,               # Randomize order for better training
        pin_memory=(device == "cuda")  # Faster GPU transfer if using CUDA
    )
    
    # Prepare validation data (if provided)
    if x_val is not None and y_val is not None:
        # Convert validation data to tensors and move to device
        x_val_tensor = torch.as_tensor(x_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.as_tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    else:
        # No validation data provided
        x_val_tensor = y_val_tensor = None
    
    # ============================================================================
    # 3. TRAINING LOOP: Main training with early stopping
    # ============================================================================
    
    # Initialize variables for early stopping and best model tracking
    best_val_loss = float("inf")  # Track best validation loss
    bad_epochs = 0                # Count epochs without improvement
    best_state = None             # Store best model weights
    
    # Main training loop: iterate through epochs
    for epoch in range(1, epochs + 1):
        # Set model to training mode (enables dropout, batch norm updates, etc.)
        model.train()
        train_loss = 0.0
        
        # ========================================================================
        # 3.1 BATCH TRAINING: Process data in batches
        # ========================================================================
        
        # Each parameter p has: p.data (the weight value) and p.grad (its gradient)
        # Training steps per batch:
        #   1) zero_grad() → clears p.grad only (weights p.data stay as last-updated)
        #   2) forward/loss/backward → computes new grads and writes them into p.grad
        #   3) clip_grad_norm_ → rescales p.grad to cap exploding gradients
        #   4) optimizer.step() → updates weights in-place (p.data) and optimizer state

        for xb, yb in train_loader:
            # Move batch data to device (GPU/CPU)
            xb, yb = xb.to(device), yb.to(device)
            
            # Clear gradients from previous batch (important!)
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass: compute predictions using current model weights
            pred = model(xb)
            
            # Calculate loss between predictions and true values
            loss = criterion(pred, yb)
            
            # Backward pass: compute gradients (how much to change each weight)
            loss.backward()
            
            # Gradient clipping: prevent exploding gradients (common in RNNs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update model weights using computed gradients
            optimizer.step()
            
            # Accumulate loss for this epoch
            train_loss += loss.item()
        
        # Calculate average training loss for this epoch
        train_loss /= len(train_loader)
        
        # ========================================================================
        # 3.2 VALIDATION: Check model performance and early stopping
        # ========================================================================
        
        val_loss = None
        if x_val_tensor is not None:
            # Set model to evaluation mode (disables dropout, batch norm uses running stats)
            model.eval()
            
            # Validation forward pass (no gradients needed)
            with torch.no_grad():
                val_pred = model(x_val_tensor)
                val_loss = criterion(val_pred, y_val_tensor).item()
            
            # Early stopping logic: save best model and check for improvement
            if val_loss < best_val_loss - 1e-6:  # Small tolerance to avoid noise
                # New best model found!
                best_val_loss = val_loss
                bad_epochs = 0
                # Save current model state (weights) as the best so far
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                # No improvement this epoch
                bad_epochs += 1
                if bad_epochs >= patience:
                    logger.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break
        
        # ========================================================================
        # 3.3 LOGGING: Print training progress
        # ========================================================================
        
        # Log progress every 5 epochs, first epoch, and last epoch
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            log_msg = f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}"
            if val_loss is not None:
                log_msg += f", val_loss: {val_loss:.4f}"
            logger.info(log_msg)
    
    # ============================================================================
    # 4. FINALIZATION: Load best model weights
    # ============================================================================
    
    # Restore the best model weights (from early stopping)
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Loaded best model (val_loss={best_val_loss:.6f})")
    
    return model


@torch.no_grad()  # Disable gradient computation for inference (faster, less memory)
def predict(model, x_test, device):
    """Make predictions with the model."""
    # Set model to evaluation mode (disables dropout, batch norm uses running stats)
    model.eval()
    
    # Convert input to tensor and move to device
    x_tensor = torch.as_tensor(x_test, dtype=torch.float32).to(device)
    
    # Forward pass: get predictions
    predictions = model(x_tensor).cpu().numpy()  # Move back to CPU and convert to numpy
    
    return predictions


@torch.no_grad()  # Disable gradient computation for evaluation
def evaluate(model, x_test, y_test, device):
    """Evaluate model performance."""
    # Set model to evaluation mode
    model.eval()
    
    # Setup loss function
    criterion = torch.nn.MSELoss()
    
    # Convert data to tensors and move to device
    x_tensor = torch.as_tensor(x_test, dtype=torch.float32).to(device)
    y_tensor = torch.as_tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    
    # Get model predictions
    predictions = model(x_tensor)
    
    # Calculate loss between predictions and true values
    loss = criterion(predictions, y_tensor).item()  # .item() converts tensor to Python float
    
    return loss


def save_model(model, filepath):
    """Save model state and configuration."""
    try:
        # Create checkpoint dictionary with model weights and configuration
        save_dict = {
            'model_state_dict': model.state_dict(),  # Model weights (PyTorch built-in)
            'model_config': {                        # Model configuration (custom)
                'input_size': model.input_size,
                'model_name': model.model_name,
                'layers': model.layers,
                'dropout_rate': model.dropout_rate
            }
        }
        
        # Convert .h5 extension to .pth (PyTorch format)
        if filepath.endswith('.h5'):
            filepath = filepath.replace('.h5', '.pth')
        
        # Save checkpoint to file
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False


def load_model(model_class, filepath, device):
    """Load model from saved state."""
    try:
        # Load checkpoint from file
        checkpoint = torch.load(filepath, map_location=device)
        
        # Extract model configuration from checkpoint
        config = checkpoint['model_config']
        
        # Create new model instance with saved configuration
        model = model_class(
            input_size=config['input_size'],
            model_name=config['model_name'],
            layers=config['layers'],
            dropout_rate=config['dropout_rate']
        )
        
        # Load saved weights into the model (PyTorch built-in method)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to device
        model.to(device)
        
        logger.info(f"Model loaded from {filepath}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None
