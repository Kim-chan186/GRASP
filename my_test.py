import wandb
import psutil
import GPUtil
# ead4e17f10e71dced63d75d8841e02621ac32330
# 프로젝트 초기화
wandb.init(project="my-project", config={"learning_rate": 0.001, "epochs": 10})

epochs = 20
for epoch in range(epochs):
    # Training logic...
    lr = epoch / 100.
    # wandb.log({"learning_rate": lr})

    # CPU 사용량
    cpu_usage = psutil.cpu_percent(interval=None)
    
    # 메모리 사용량
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    
    # GPU 사용량 및 온도
    gpus = GPUtil.getGPUs()
    gpu_usage = gpus[0].load * 100
    gpu_temp = gpus[0].temperature

    loss = epoch/10.
    # wandb.log({"train/loss": loss})
    if epoch > 5:
        wandb.log({"epoch/epoch": epoch})
    # 메트릭 로깅
    wandb.log({
        "resource/cpu_usage": cpu_usage,
        "resource/memory_usage": memory_usage,
        "resource/gpu_usage": gpu_usage,
        "resource/gpu_temp": gpu_temp
    })
    
wandb.finish()