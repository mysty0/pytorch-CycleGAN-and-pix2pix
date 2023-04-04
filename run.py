import torch
from sys import platform

if not torch.cuda.is_available():
    print("GPU not detected")
#    quit()

print("Env loaded OK, platform: {}".format(platform))

import os
import sys

if platform == "win32":
    python = "venv\\Scripts\\python.exe"
    dataset = 'datasets/cel_face_lights'
else:
    python = "venv/bin/python"
    dataset = '/mnt/storage/scratch/$USER/datasets/cel_face_lights'


base_cmd = python + f" train.py --dataroot {dataset} --name |name| --model dcs --display_id -1"
base_test_cmd = python + f" test.py --dataroot {dataset} --name |name| --model dcs".format("dataset")
#base_cmd = python + " train.py --dataroot datasets/cel_face2 --name {} --model pix2pix  --batch_size 80"#--display_id -1
#base_test_cmd = python + " test.py --dataroot datasets/cel_face2 --name {} --model pix2pix"
base_name = "cel_face_lights_dcs"

experement = int(sys.argv[1])
print(f"experement num {experement}")

if experement == 0:
    args = " --vae --netG resnet_9blocks"
    name = base_name + "_vae_resnet"
    cmd = base_cmd.replace('|name|', name)
    os.system(cmd + args + " --n_epochs 400")
    os.system(base_test_cmd.replace('|name|', name) + args)
elif experement == 1:
    name = base_name + "_vae_resnet_lowres"
    args = " --vae --netG resnet_9blocks --load_size 156 --crop_size 128"
    cmd = base_cmd.replace('|name|', name)
    os.system(cmd + args + " --n_epochs 400")
    os.system(base_test_cmd.replace('|name|', name) + args)
elif experement == 2:
    name = base_name + "_vae_resnet_long"
    args = " --vae --netG resnet_9blocks"
    cmd = base_cmd.replace('|name|', name)
    os.system(cmd + args + " --n_epochs 800")
    os.system(base_test_cmd.replace('|name|', name) + args)
elif experement == 3:
    name = base_name + "_vae_resnet_hightres"
    args = " --vae --netG resnet_9blocks --load_size 600 --crop_size 512"
    cmd = base_cmd.replace('|name|', name)
    os.system(cmd + args + " --n_epochs 800")
    os.system(base_test_cmd.replace('|name|', name) + args)
elif experement == 4:
    name = base_name + "_more_epoch"
    args = ""
    cmd = base_cmd.replace('|name|', name)
    os.system(cmd + args + " --n_epochs 300 --n_epochs_decay 300")
    os.system(base_test_cmd.replace('|name|', name) + args)