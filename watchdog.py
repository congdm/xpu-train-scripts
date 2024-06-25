import os
import subprocess
import pathlib
import time

def check_last_success_epoch(output_folder, LORA_NAME, epochs):
    epoch = 0
    resume = None
    if pathlib.Path(output_folder).exists():
        dir_list = os.listdir(output_folder)
        x = pathlib.Path(output_folder, f'{LORA_NAME}_{epoch+1:04d}')
        while epoch < epochs and x.exists():
            resume = f'{output_folder}/{LORA_NAME}_{epoch+1:04d}'
            epoch = epoch + 1
            x = pathlib.Path(output_folder, f'{LORA_NAME}_{epoch+1:04d}')
    return epoch, resume

def main():
    epochs = 128
    output_folder = 'output'
    LORA_NAME = 'Mayu_lora'
    epoch, resume = check_last_success_epoch(output_folder, LORA_NAME, epochs)
    while epoch < epochs:
        command = [
            'python', 'train_hunyuan.py',
            '--output-folder', output_folder,
        ]
        if resume is not None:
            command.append('--resume')
            command.append(resume)
        subprocess.run(command)
        epoch, resume = check_last_success_epoch(output_folder, LORA_NAME, epochs)

main()
    

