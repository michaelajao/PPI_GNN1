import torch
import sys
import platform

def check_gpu():
    print("=" * 40)
    print("SYSTEM INFORMATION")
    print("=" * 40)
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print("-" * 40)
    
    print("\nCUDA INFORMATION")
    print("=" * 40)
    
    if torch.cuda.is_available():
        print(f"CUDA is available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\nDevice {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  CUDA cores: {props.multi_processor_count}")
            
        # Set current device
        torch.cuda.set_device(0)
        print(f"\nCurrent CUDA device: {torch.cuda.current_device()}")
        
        # Test tensor operations on GPU
        print("\nGPU Tensor Test:")
        x = torch.rand(5, 5).cuda()
        y = torch.rand(5, 5).cuda()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z = x @ y  # Matrix multiplication
        end_time.record()
        
        # Wait for operations to complete
        torch.cuda.synchronize()
        print(f"  Time for 5x5 matrix multiplication: {start_time.elapsed_time(end_time):.3f} ms")
        print(f"  Result is on GPU: {z.is_cuda}")
        
        # Test with larger tensor
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        
        start_time.record()
        z = x @ y
        end_time.record()
        
        torch.cuda.synchronize()
        print(f"  Time for 1000x1000 matrix multiplication: {start_time.elapsed_time(end_time):.3f} ms")
        
        # Memory information
        print(f"\nCUDA Memory Summary:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
    else:
        print("CUDA is NOT available. This code will run on CPU only.")
        print("To use GPU acceleration, check your PyTorch installation and CUDA setup.")
    
    print("=" * 40)

if __name__ == "__main__":
    check_gpu()
    
    if not torch.cuda.is_available() and len(sys.argv) > 1 and sys.argv[1] == "--fail-if-no-gpu":
        sys.exit(1)
