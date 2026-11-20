import torch.utils.data
import torchvision

from .ag_single import build as build_ag_single
from .ag_multi import build as build_ag_multi
from .star_single import build as build_star_single

def build_dataset(image_set, args):
    if args.dataset_file == 'ag_single':
        return build_ag_single(image_set, args)
    if args.dataset_file == 'ag_multi':
        return build_ag_multi(image_set, args)
    if args.dataset_file == 'star_single':
        return build_star_single(image_set, args)
    if args.dataset_file == 'star_multi':
        pass # TODO
    raise ValueError(f'dataset {args.dataset_file} not supported')

