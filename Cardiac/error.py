# diagnostic.py
import subprocess
import sys
import platform

def diagnose_torch_issue():
    print("=" * 50)
    print("PyTorch导入问题诊断工具")
    print("=" * 50)
    
    # 1. 系统信息
    print(f"\n1. 系统信息:")
    print(f"   操作系统: {platform.system()} {platform.release()}")
    print(f"   Python版本: {sys.version}")
    
    # 2. GPU信息
    print(f"\n2. GPU信息:")
    try:
        nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if nvidia_smi.returncode == 0:
            # 提取驱动版本和GPU型号
            for line in nvidia_smi.stdout.split('\n'):
                if 'Driver Version' in line:
                    print(f"   {line.strip()}")
                if 'NVIDIA-SMI' in line:
                    print(f"   {line.strip()}")
                if 'GeForce' in line or 'Tesla' in line or 'RTX' in line:
                    if '|   0 ' in line or '|====' not in line:
                        print(f"   GPU型号: {line.strip()}")
        else:
            print("   nvidia-smi未找到，可能未安装NVIDIA驱动")
    except FileNotFoundError:
        print("   nvidia-smi命令不存在，请确认是否安装NVIDIA驱动")
    
    # 3. CUDA版本
    print(f"\n3. CUDA信息:")
    try:
        nvcc = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if nvcc.returncode == 0:
            for line in nvcc.stdout.split('\n'):
                if 'release' in line:
                    print(f"   CUDA Toolkit: {line.strip()}")
        else:
            print("   nvcc未找到，可能未安装CUDA Toolkit")
    except FileNotFoundError:
        print("   nvcc命令不存在")
    
    # 4. 尝试导入torch
    print(f"\n4. PyTorch检查:")
    try:
        import torch
        print(f"   ✓ PyTorch导入成功")
        print(f"   PyTorch版本: {torch.__version__}")
        print(f"   CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU型号: {torch.cuda.get_device_name(0)}")
            print(f"   PyTorch编译时的CUDA版本: {torch.version.cuda}")
        else:
            # 尝试判断是否为CPU版本
            if hasattr(torch, 'version') and hasattr(torch.version, 'cuda'):
                if torch.version.cuda is None:
                    print("   ⚠️ 当前安装的是CPU-only版本的PyTorch")
    except ImportError as e:
        print(f"   ✗ PyTorch导入失败: {e}")
    except Exception as e:
        print(f"   ✗ 其他错误: {e}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    diagnose_torch_issue()