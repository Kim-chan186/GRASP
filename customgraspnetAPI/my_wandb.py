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
        "res/CPU_usage": cpu_usage,
        "res/GPU_usage": gpu_usage,
        "res/GPU_temp": gpu_temp,
        "res/memory_usage": memory_usage
    })

# ead4e17f10e71dced63d75d8841e02621ac32330
# 프로젝트 초기화
def my_wandb_init(key, project_name, config):
    if key is not None:
        wandb.login(key=key)
    wandb.init(project=project_name, config=config)
    