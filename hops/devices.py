"""
Utility functions to retrieve information about available services and setting up security for the Hops platform.

These utils facilitates development by hiding complexity for programs interacting with Hops services.
"""
import subprocess
import time
import threading
import random


def get_gpu_info():
    # Get the gpu information
    gpu_str = ''
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "--format=csv,noheader,nounits", "--query-gpu=name,memory.total,memory.used,utilization.gpu"]).decode()
        gpu_info = gpu_info.split('\n')
    except:
        gpu_str = '\nCould not find any GPUs accessible for the container\n'
        return gpu_str

    # Check each gpu
    gpu_str = ''
    for line in gpu_info:
        if len(line) > 0:
            name, total_memory, memory_used, gpu_util = line.split(',')
            gpu_str += '\nName: ' + name + '\n'
            gpu_str += 'Total memory: ' + total_memory + '\n'
            gpu_str += 'Currently allocated memory: ' + memory_used + '\n'
            gpu_str += 'Current utilization: ' + gpu_util + '\n'
            gpu_str += '\n'

    return gpu_str


def get_gpu_util():
    gpu_str = ''
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "--format=csv,noheader,nounits", "--query-gpu=name,memory.total,memory.used,utilization.gpu"]).decode()
        gpu_info = gpu_info.split('\n')
    except:
        return gpu_str

    gpu_str = '\n------------------------------ GPU usage information ------------------------------\n'
    for line in gpu_info:
        if len(line) > 0:
            name, total_memory, memory_used, gpu_util = line.split(',')
            gpu_str += '[Type: ' + name + ', Memory Usage: ' + memory_used + ' /' + total_memory + ' (MB), Current utilization: ' + gpu_util + '%]\n'
    gpu_str += '-----------------------------------------------------------------------------------\n'
    return gpu_str


def print_periodic_gpu_utilization():
    t = threading.currentThread()
    while getattr(t, "do_run", True):
        print(get_gpu_util())
        time.sleep(10)


def get_num_gpus():
    """ Get the number of GPUs available in the environment

    Returns:
      Number of GPUs available in the environment
    """
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "--format=csv,noheader,nounits", "--query-gpu=name"]).decode()
        gpu_info = gpu_info.split('\n')
    except:
        return 0

    count = 0
    for line in gpu_info:
        if len(line) > 0:
            count += 1
    return count


def get_minor_gpu_device_numbers():

    gpu_info = []
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "--format=csv,noheader,nounits", "--query-gpu=pci.bus_id"]).decode()
    except:
        return gpu_info

    gpu_info = gpu_info.split('\n')
    device_id_list = []
    for line in gpu_info:
        if len(line) > 0:
            pci_bus_id = line.split(',')
            device_id_list.append(pci_bus_id)


def get_gpu_uuid():
    gpu_uuid = []
    try:
        gpu_uuid = subprocess.check_output(["nvidia-smi", "--format=csv,noheader", "--query-gpu=uuid"]).decode()
        gpu_uuid = gpu_uuid.split('\n')
    except:
        #print('Failed to get gpu uuid.')
        gpu_uuid = ['GPU-cb5c5595-4d01-0b68-7fd3-a0496e5ac4aa', 'GPU-47013580-4e02-8ad9-4319-b5690c3f3108',
                    'GPU-fb9c8e36-db9c-8c04-afa1-1922f2eeb0b2', 'GPU-393e9461-f2ec-0d0f-0023-2dd571c2807e',
                    'GPU-f83746ab-bdea-3591-0c09-f21b64fb3756', 'GPU-4c3da8f4-3afe-01fe-a537-0e767670cff0',
                    'GPU-ddcbad08-ea08-7196-cb1a-4be05e89b3a4', 'GPU-8880730b-d131-64ec-c980-91b6ee4f1231',
                    'GPU-a86fcab4-c817-da4e-fa51-a460e3ebe230', 'GPU-b87d68ae-2ab3-27a8-0a91-208ef278b958',
                    'GPU-7633d3f1-206f-d07d-0d5a-1ac13390130c', 'GPU-57466f09-4da5-3146-1e20-b3c76fd351ee',
                    'GPU-433f2fa7-1bc2-09d8-d144-4f6beed7f99e', 'GPU-b2079f6e-aedd-3a18-4127-a899ba8a70c3',
                    'GPU-d61cb7c6-15d3-0161-cd26-b3a4d1aea265', 'GPU-5c73eb31-18b2-56fb-d4c2-7e8b7fab6fc4',
                    'GPU-24ca66fc-6fb0-4713-b25a-19abeb02dff1', 'GPU-b884dd3d-1bec-1db2-3568-a6d0d8633c0c',
                    'GPU-29cce9fb-b95c-de2d-ba82-12189f273de3', 'GPU-e0753578-d97e-01d5-68d9-445255dba3cf']
    #gpu_uuid = [str(x) for x in gpu_uuid if x]
    gpu_uuid = random.choice(gpu_uuid)
    return [gpu_uuid]
