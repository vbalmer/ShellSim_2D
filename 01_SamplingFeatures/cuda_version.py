import torch

print(torch.cuda.get_device_name(0))
print(torch.version.cuda)

import psutil

print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")

disk_usage = psutil.disk_usage('/')

print(f"Total Disk Size: {disk_usage.total / (1024**3):.2f} GB")
print(f"Used Disk Size: {disk_usage.used / (1024**3):.2f} GB")
print(f"Free Disk Size: {disk_usage.free / (1024**3):.2f} GB")
print(f"Usage Percent: {disk_usage.percent}%")