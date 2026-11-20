import os
import sys
import random

def run_train(device='0', tag='temporal_sgdet'):
    port = random.randint(10000,59999)
    num_devices = len(device.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.system(f"python -m torch.distributed.launch \
                    --master_addr 127.0.0.1 \
                    --master_port {port} \
                    --nproc_per_node={num_devices} \
                    --use_env \
                    main.py \
                    --pretrained exps/spatial_sgdet_agstar_partial/checkpoint_16.pth \
                    --output_dir exps/{tag}/ \
                    --train_annotations_file agstar_train_partial.json \
                    --dataset_file ag_multi \
                    --backbone resnet50 \
                    --num_queries 100 \
                    --dec_layers_hopd 6 \
                    --dec_layers_interaction 6 \
                    --epochs 6 \
                    --lr_drop 3 \
                    --batch_size 1 \
                    --num_workers 0 \
                    --freeze_mode 1 \
                    --query_temporal_interaction \
                    --dsgg_task sgdet")
    


if __name__ == "__main__":
    device = '0'
    run_train(device=device, tag='temporal_sgdet_agstar_partial')
