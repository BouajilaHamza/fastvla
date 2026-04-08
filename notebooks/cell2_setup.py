# FastVLA Kaggle T4 Notebook - Setup Cell (Cell 2)
# Copy this entire cell and replace Cell 2 in your notebook

"""
⚠️ CRITICAL: Install dependencies in correct order to avoid version conflicts
"""

# 1. Upgrade huggingface_hub FIRST (required by Unsloth and datasets)
!pip install -q --upgrade 'huggingface_hub>=0.30.0' --no-cache-dir

# 2. Install Unsloth and dependencies  
!pip install -q unsloth_zoo --no-cache-dir
!pip install -q git+https://github.com/unslothai/unsloth.git --no-cache-dir

# 3. Install FastVLA from GitHub (now has latest Unsloth API + dtype fixes)
!pip install -q git+https://github.com/BouajilaHamza/FastVLA.git --no-cache-dir

# 4. Install/upgrade remaining dependencies
!pip install -q --upgrade datasets triton bitsandbytes accelerate peft transformers timm --no-cache-dir

# ⚠️ CRITICAL: Import Unsloth FIRST to apply patches
import unsloth
print('✓ Unsloth imported first (patches applied)')

# Now verify FastVLA installation
print('\n' + '='*60)
print('FastVLA Environment Diagnostic')
print('='*60)

try:
    import fastvla
    print(f'✓ FastVLA imported')
    
    # Check if Unsloth detection works
    from fastvla import model as fastvla_model
    if fastvla_model.UNSLOTH_AVAILABLE:
        print('  ✓ Unsloth integration detected')
    else:
        print('  ⚠ Unsloth not detected (4-bit loading may fail)')
        
    import torch
    print(f'  ✓ PyTorch: {torch.__version__}')
    print(f'  ✓ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  ✓ Device: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'✗ Import failed: {e}')

print('='*60)
print('✅ Environment ready! Proceed to Cell 3 to load the model.')
