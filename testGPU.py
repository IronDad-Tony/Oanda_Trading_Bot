import torch

# 1. 檢查 PyTorch 版本
print(f"PyTorch Version: {torch.__version__}")

# 2. 檢查 PyTorch 是用哪個 CUDA 版本編譯的
#    對於 cu121，這裡應該顯示 12.1
#    對於 cu128，這裡應該顯示 12.8 (如果這個包存在且被正確安裝)
print(f"PyTorch compiled with CUDA Version: {torch.version.cuda}")

# 3. 檢查 CUDA 是否可用 (這是最重要的)
is_cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {is_cuda_available}")

if is_cuda_available:
    # 4. 獲取可用的 GPU 數量
    print(f"Number of GPUs Available: {torch.cuda.device_count()}")

    # 5. 獲取當前 GPU 的索引 (預設是 0)
    current_gpu_idx = torch.cuda.current_device()
    print(f"Current GPU Index: {current_gpu_idx}")

    # 6. 獲取當前 GPU 的名稱
    print(f"Current GPU Name: {torch.cuda.get_device_name(current_gpu_idx)}")

    # 7. 檢查 PyTorch 使用的 cuDNN 版本
    #    這會是 PyTorch 捆綁的 cuDNN 版本，可能與您系統安裝的不同
    print(f"cuDNN Version (used by PyTorch): {torch.backends.cudnn.version()}")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print(f"CUDA version detected by PyTorch: {torch.version.cuda}")
    print(f"cuDNN version detected by PyTorch: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
  

    # 8. 嘗試一個簡單的 GPU 運算
    try:
        x = torch.tensor([1.0, 2.0, 3.0])
        print(f"Original tensor: {x}")
        x_cuda = x.cuda() # 或者 x.to('cuda')
        print(f"Tensor moved to GPU: {x_cuda}")
        print(f"Device of x_cuda: {x_cuda.device}")
        print("GPU is working correctly!")
    except Exception as e:
        print(f"Error during GPU tensor operation: {e}")
else:
    print("CUDA is NOT available. Please check your installation and drivers.")
    print("Ensure your NVIDIA drivers are up to date and support CUDA 12.x.")
    print("Also verify that you installed the CUDA-enabled PyTorch package (e.g., with 'cu121').")

# 退出 Python 解釋器
# exit()