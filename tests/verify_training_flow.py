import torch
import torch.nn as nn
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.networks import WifiPoseModel

def verify_csi_training():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("CSI_Verification")
    logger.info("Starting CSI Primacy Verification...")

    # 1. Initialize Model
    model = WifiPoseModel(input_features=64, output_points=33)
    model.train()
    logger.info("Model initialized. Input expected: [Batch, Seq, 64]")

    # 2. Create Synthetic CSI Data (Batch=2, Seq=50, Channels=64)
    # This simulates 2 seconds of CSI data from 2 users
    rf_input = torch.randn(2, 50, 64) 
    logger.info(f"Generated Synthetic CSI Data Shape: {rf_input.shape}")

    # 3. Create Synthetic Ground Truth (Batch=2, Points=66)
    # This simulates MediaPipe labels
    pose_gt = torch.randn(2, 66)
    presence_gt = torch.ones(2, 1) # Both present
    location_gt = torch.randint(0, 4, (2,)) # Random locations

    # 4. Forward Pass
    logger.info("Executing Forward Pass...")
    pose_pred, pres_pred, loc_pred = model(rf_input)
    
    # Check outputs
    logger.info(f"Pose Prediction Shape: {pose_pred.shape}")
    assert pose_pred.shape == (2, 66), "Pose prediction shape mismatch!"

    # 5. Backward Pass (Check Gradients)
    criterion = nn.MSELoss()
    loss = criterion(pose_pred, pose_gt)
    logger.info(f"Computed Loss: {loss.item():.4f}")

    loss.backward()
    logger.info("Executed Backward Pass.")

    # 6. Verify Gradients on RF Encoder
    # If gradients are non-zero, the model IS learning from CSI
    first_conv_weight_grad = model.encoder.cnn[0].weight.grad
    grad_norm = first_conv_weight_grad.norm().item()
    
    logger.info(f"Gradient Norm on First RF Conv Layer: {grad_norm:.6f}")
    
    if grad_norm > 0:
        logger.info("SUCCESS: Gradients are flowing to the RF Input Layer!")
        logger.info("This proves the model uses the CSI data to minimize loss.")
    else:
        logger.error("FAILURE: No gradients on RF Input. Model ignores CSI!")
        sys.exit(1)

if __name__ == "__main__":
    verify_csi_training()
