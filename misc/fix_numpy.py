"""
Fix NumPy DLL issues on Windows
Run this script to fix NumPy installation problems
"""

import subprocess
import sys

print("Fixing NumPy installation...")
print("=" * 60)

# Step 1: Uninstall NumPy
print("\n1. Uninstalling NumPy...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "numpy", "-y"], check=False)

# Step 2: Install NumPy with all dependencies
print("\n2. Installing NumPy 1.24.3 (compatible version)...")
subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.24.3", "--no-cache-dir", "--force-reinstall"], check=True)

# Step 3: Test import
print("\n3. Testing NumPy import...")
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} imported successfully!")
except Exception as e:
    print(f"✗ NumPy import failed: {e}")
    print("\nIf this fails, you may need to:")
    print("1. Install Visual C++ Redistributables:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("2. Or use a virtual environment:")
    print("   python -m venv venv")
    print("   venv\\Scripts\\activate")
    print("   pip install -r requirements.txt")

print("\n" + "=" * 60)
print("Done!")

