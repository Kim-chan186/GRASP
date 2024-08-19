import wandb
import psutil
import GPUtil

def log_system_metrics():
    # CPU 사용량
    cpu_usage = psutil.cpu_percent(interval=None)
    
    # 메모리 사용량
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    
    # GPU 사용량 및 온도
    gpus = GPUtil.getGPUs()
    gpu_usage = gpus[0].load * 100
    gpu_temp = gpus[0].temperature
    
    # 메트릭 로깅
    wandb.log({
        "cpu_usage": cpu_usage,
        "memory_usage": memory_usage,
        "gpu_usage": gpu_usage,
        "gpu_temp": gpu_temp
    })

    