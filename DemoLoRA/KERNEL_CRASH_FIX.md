# Jupyter Kernel Crash - Root Cause & Fix

## Problem Summary
Your Jupyter notebook kernel was crashing repeatedly with exit code `undefined` because VS Code was using the **wrong Python interpreter**.

## Root Cause Analysis

### From the Logs
```
12:26:18.233 [info] Starting Kernel (Python Path: /opt/miniforge3/bin/python, Conda, 3.11.13)
12:26:42.247 [error] Disposing session as kernel process died ExitCode: undefined
```

**The Issue:**
- VS Code was launching the kernel with `/opt/miniforge3/bin/python` (base conda environment)
- Your dependencies were installed in `/home/kimj24/416Final/envfin/bin/python` (your venv)
- The base conda environment doesn't have `transformers`, `peft`, `trl`, etc.
- When the notebook tried to import missing packages, the kernel crashed

### Why It Kept Failing
The logs show VS Code trying to use old/invalid kernels:
```
[warn] The following kernels use interpreters that are no longer valid or not recognized
Kernels: /work/envs/venv/DL_gpu/bin/python (invalid path)
Valid interpreter ids include: ~/416Final/envfin/bin/python
```

VS Code had cached kernel configurations pointing to non-existent or wrong Python environments.

## The Fix

### What I Did

1. **Installed Jupyter/IPykernel in your venv**
   ```bash
   source /home/kimj24/416Final/envfin/bin/activate
   pip install ipykernel jupyter
   ```
   - This adds the necessary Jupyter kernel infrastructure to your virtual environment

2. **Created a Jupyter Kernel Spec**
   ```bash
   python -m ipykernel install --user --name=envfin --display-name="Python (envfin-416Final)"
   ```
   - This registers your virtual environment as a selectable kernel in Jupyter/VS Code
   - Kernel spec location: `/home/kimj24/.local/share/jupyter/kernels/envfin/`

3. **Added Instructions to Notebook**
   - Added a prominent markdown cell at the top explaining how to select the correct kernel

### Kernel Spec Details
The created kernel spec (`/home/kimj24/.local/share/jupyter/kernels/envfin/kernel.json`):
```json
{
 "argv": [
  "/home/kimj24/416Final/envfin/bin/python",
  "-Xfrozen_modules=off",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Python (envfin-416Final)",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
```

This ensures the kernel uses the correct Python interpreter with all your installed dependencies.

## How to Use the Fix

### Step 1: Select the Correct Kernel
1. Open `LoRADemo.ipynb` in VS Code
2. Look for the **"Select Kernel"** button in the top-right corner
3. Click it and choose **"Python (envfin-416Final)"** from the dropdown
4. The kernel should start without crashing

### Step 2: Verify It Works
Run the first cell (imports). It should execute successfully without errors:
```python
import torch
import json
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (...)
```

If you see output like:
```
✅ All imports successful
```
Then the kernel is working correctly!

### Step 3: If It Still Crashes
If the kernel still crashes after selecting the correct kernel:

1. **Restart VS Code completely**
   ```bash
   # Close VS Code and reopen
   ```

2. **Clear VS Code's kernel cache**
   - Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
   - Run: `Jupyter: Clear All Cached Information`
   - Restart VS Code

3. **Verify kernel availability**
   ```bash
   jupyter kernelspec list
   ```
   You should see:
   ```
   envfin      /home/kimj24/.local/share/jupyter/kernels/envfin
   ```

4. **Check kernel manually**
   ```bash
   source /home/kimj24/416Final/envfin/bin/activate
   python -m ipykernel --version
   # Should print: 7.1.0
   ```

## Understanding the Different Python Environments

You have multiple Python installations on your system:

| Environment | Path | Has Dependencies? | Use For |
|-------------|------|-------------------|---------|
| **Your venv** ✅ | `~/416Final/envfin/bin/python` | YES (all packages) | **LoRA notebook** |
| Base conda | `/opt/miniforge3/bin/python` | NO | System tasks |
| Anaconda | `/opt/anaconda3/bin/python` | NO | Other projects |
| Huggingface env | `/opt/miniforge3/envs/huggingface/bin/python` | Unknown | HuggingFace projects |
| System Python | `/usr/bin/python3` | NO | System scripts |

**Always use `Python (envfin-416Final)` for this LoRA project.**

## Technical Background: Why Kernels Crash

### Common Reasons for Kernel Crashes

1. **Missing Packages** (your issue)
   - Kernel uses Python environment without required packages
   - Import fails → kernel crashes

2. **Memory Exhaustion**
   - Model too large for available RAM/VRAM
   - Python process killed by OS (OOM killer)
   - Exit code: 137 or 9

3. **CUDA Issues**
   - Wrong CUDA version
   - GPU driver incompatibility
   - Usually shows CUDA-related error before crash

4. **Segmentation Faults**
   - Low-level C/C++ library errors
   - Often from PyTorch, NumPy, or CUDA libraries
   - Exit code: 139 or 11

5. **Incompatible Dependencies**
   - Version conflicts between packages
   - Usually shows ImportError or AttributeError

### Your Specific Case
Your crash was Type 1 (Missing Packages):
- No error message before crash (just `ExitCode: undefined`)
- Happens immediately on first import
- Wrong Python environment selected

## Prevention for Future Projects

To avoid this issue in future notebooks:

### 1. Always Create Kernel Spec for New Environments
```bash
# Activate your venv
source /path/to/your/venv/bin/activate

# Install ipykernel
pip install ipykernel

# Register the kernel
python -m ipykernel install --user --name=myproject --display-name="Python (MyProject)"
```

### 2. Verify Kernel Before Running Notebooks
- Check the kernel name in VS Code's top-right corner
- Ensure it matches your project's environment
- Don't use "Python 3" or generic kernels for projects with specific dependencies

### 3. Add Kernel Selection Instructions to Notebooks
Add this to the top of your notebooks:
```markdown
## 🔧 Setup: Select Correct Kernel
**Before running**: Click "Select Kernel" → Choose "Python (YourEnvName)"
```

### 4. Use Requirements Files
Create `requirements.txt` for reproducibility:
```bash
pip freeze > requirements.txt
```

Others can then set up the same environment:
```bash
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=myenv
```

## Troubleshooting Commands

### List Available Kernels
```bash
jupyter kernelspec list
```

### Remove Old/Broken Kernels
```bash
jupyter kernelspec remove <kernel-name>
```

### Test Kernel Manually
```bash
source /home/kimj24/416Final/envfin/bin/activate
python -m ipykernel_launcher
# Should start without errors
# Press Ctrl+C to exit
```

### Check Package Installation
```bash
source /home/kimj24/416Final/envfin/bin/activate
python -c "import transformers; print(transformers.__version__)"
python -c "import torch; print(torch.__version__)"
```

## Summary

✅ **Fixed**: Installed ipykernel and created proper kernel spec  
✅ **Root cause**: Wrong Python interpreter (base conda instead of your venv)  
✅ **Solution**: Select "Python (envfin-416Final)" kernel in VS Code  
✅ **Prevention**: Always create kernel specs for virtual environments  

Your notebook should now run without kernel crashes! 🎉
