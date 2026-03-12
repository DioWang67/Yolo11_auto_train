import os
import site

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    packages = site.getsitepackages()
    for p in packages:
        torch_lib = os.path.join(p, "torch", "lib")
        if os.path.exists(torch_lib):
            print(f"Adding DLL directory: {torch_lib}")
            os.add_dll_directory(torch_lib)
            break
except Exception as e:
    print(e)
    pass

import torch
print("Torch imported successfully:", torch.__version__)
