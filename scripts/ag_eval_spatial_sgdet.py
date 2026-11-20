import os
import sys
import random

def run_eval(device='0', ckpt_path='exps/params/sgdet/spatial/checkpoint_22.pth'):
    port = random.randint(10000,59999)
    num_devices = len(device.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.system(f"python -m torch.distributed.launch \
                    --master_addr 127.0.0.1 \
                    --master_port {port} \
                    --nproc_per_node={num_devices} \
                    --use_env \
                    main.py \
                    --train_annotations_file agstar_train_partial.json \
                    --test_annotations_file agstar_val_partial.json \
                    --eval_star_vocab True \
                    --dataset_file ag_single \
                    --data_path data \
                    --backbone resnet50 \
                    --num_queries 100 \
                    --dec_layers_hopd 6 \
                    --dec_layers_interaction 6 \
                    --batch_size 8 \
                    --eval \
                    --pretrained {ckpt_path} \
                    --dsgg_task sgdet")


if __name__ == "__main__":
    device = '0'
    run_eval(device=device, ckpt_path='exps/spatial_sgdet_agstar_partial/checkpoint_14.pth')
