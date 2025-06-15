#!/usr/bin/env python3
"""
Test script to verify CrossTimeScaleFusion works with hierarchical_attention
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from src.models.custom_layers import CrossTimeScaleFusion

def test_cross_time_scale_fusion():
    """Test CrossTimeScaleFusion with hierarchical_attention"""
    print("Testing CrossTimeScaleFusion with hierarchical_attention...")
    
    try:
        # Create the fusion module
        d_model = 256
        time_scales = [1, 3, 5]
        fusion_type = "hierarchical_attention"
        
        cts_fusion = CrossTimeScaleFusion(
            d_model=d_model,
            time_scales=time_scales,
            fusion_type=fusion_type,
            dropout_rate=0.1,
            num_heads_hierarchical=4
        )
        
        print(f"✓ CrossTimeScaleFusion created successfully with fusion_type='{fusion_type}'")
        print(f"  - d_model: {d_model}")
        print(f"  - time_scales: {time_scales}")
        print(f"  - Module type: {type(cts_fusion)}")
        
        # Test forward pass with dummy data
        batch_size = 2
        num_symbols = 5
        seq_len = 128
        
        # Create dummy input tensor [B*N, T, C]
        x = torch.randn(batch_size * num_symbols, seq_len, d_model)
        print(f"  - Input shape: {x.shape}")
        
        # Forward pass
        with torch.no_grad():
            output = cts_fusion(x)
            print(f"  - Output shape: {output.shape}")
        
        print("✓ Forward pass completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cross_time_scale_fusion()
    if success:
        print("\n✓ CrossTimeScaleFusion test passed!")
    else:
        print("\n✗ CrossTimeScaleFusion test failed!")
        sys.exit(1)
