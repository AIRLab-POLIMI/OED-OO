import os
import sys
import random

def run_train(device='0', tag='spatial_sgdet'):
    port = random.randint(10000,59999)
    num_devices = len(device.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.system(f"python -m torch.distributed.launch \
                    --master_addr 127.0.0.1 \
                    --master_port {port} \
                    --nproc_per_node={num_devices} \
                    --use_env \
                    main.py \
                    --train_annotations_file star_train.json \
                    --test_annotations_file star_val.json \
                    --val_interval 2 \
                    --patience 5 \
                    --early_stop_set with \
                    --early_stop_metric R@50 \
                    --output_dir exps/{tag}/ \
                    --num_spatial_classes 9 \
                    --num_contacting_classes 18 \
                    --dataset_file star_single \
                    --data_path data \
                    --backbone resnet50 \
                    --num_queries 100 \
                    --dec_layers_hopd 6 \
                    --dec_layers_interaction 6 \
                    --batch_size 4 \
                    --epochs 30 \
                    --lr_drop 8 \
                    --pretrained exps/params/detr/detr-r50-pre-2stage_e29.pth \
                    --dsgg_task sgdet")
    


if __name__ == "__main__":
    device = '0'
    run_train(device=device, tag='spatial_sgdet_star')