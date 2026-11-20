from pathlib import Path

from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_single as T
from torch.utils.data.dataset import ConcatDataset
import pdb

from .ag_single import DSGGDataset, ConvertCocoPolysToMask, make_coco_transforms

# Only IDs need to be unique across groups. Order within a list = column index.
REL_GROUPS = {
    "spatial":    [3, 4, 5, 6, 7, 26, 27, 28, 29],      # old spatial + new ones
    "contacting": list(range(8, 26)),               # 8..25
}

# global id -> column index within its group
ID_TO_COL = {
    name: {rid: i for i, rid in enumerate(ids)}
    for name, ids in REL_GROUPS.items()
}

def build_rel_labels(anno):
    ann_keys = {"spatial": "spatial_rel", "contacting": "contact_rel"}  # drop attention_rel
    labels = {name: [] for name in REL_GROUPS}
    for oid in range(1, len(anno)):
        obj = anno[oid]
        for name, ids in REL_GROUPS.items():
            vec = torch.zeros(len(ids))
            for rid in obj[ann_keys[name]]:
                vec[ID_TO_COL[name][int(rid)]] = 1
            labels[name].append(vec)
    return {name: torch.stack(v, dim=0) for name, v in labels.items()}

def _xywh_to_xyxy_clamped(boxes, w, h):
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)
    return boxes

def _valid(b):
    return (b[:, 3] > b[:, 1]) & (b[:, 2] > b[:, 0])

class DSGGDatasetOO(DSGGDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare = ConvertCocoPolysToMaskOO(self._transforms)

class ConvertCocoPolysToMaskOO(ConvertCocoPolysToMask):

    def __call__(self, image, target):
        w, h = image.size
        anno = [o for o in target["annotations"]
                if 'iscrowd' not in o or o['iscrowd'] in (0, -1)]
        person, rels = (anno[0] if anno else None), anno[1:]
        is_oo = [o.get('isobjobj', False) for o in rels]

        o_boxes = _xywh_to_xyxy_clamped([o["bbox"] for o in rels], w, h)
        s_boxes = _xywh_to_xyxy_clamped(
            [o['subj_bbox'] if oo else person['bbox'] for o, oo in zip(rels, is_oo)], w, h)
        o_classes = torch.tensor([o["category_id"] for o in rels], dtype=torch.int64)
        s_classes = torch.tensor(
            [o['subj_id'] if oo else person['category_id'] for o, oo in zip(rels, is_oo)],
            dtype=torch.int64)

        rels = build_rel_labels(anno)
        spatial = rels["spatial"]
        contact = rels["contacting"]

        keep = _valid(o_boxes) & _valid(s_boxes)
        o_boxes, s_boxes = o_boxes[keep], s_boxes[keep]
        o_classes, s_classes = o_classes[keep], s_classes[keep]
        spatial, contact = spatial[keep], contact[keep]
        n = len(o_boxes)

        boxes = torch.cat([o_boxes, s_boxes], dim=0)          # [objects..., subjects...]
        target = {
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
            "boxes": boxes,
            "labels": torch.cat([o_classes, s_classes], dim=0),
            "iscrowd": torch.zeros(2 * n, dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
        }
        image, target = self._transforms(image, target)

        out_boxes = target['boxes']
        target['obj_boxes'] = out_boxes[:n]
        target['sub_boxes'] = out_boxes[n:]
        target['obj_labels'] = o_classes
        target['subj_labels'] = s_classes
        target['spatial_labels'] = spatial
        target['contacting_labels'] = contact
        target['matching_labels'] = torch.ones(n, dtype=torch.int64)
        
        return image, target

def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided data path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "action-genome", root / "star" / "annotations" / args.train_annotations_file),
        "val": (root / "action-genome", root / "star" / "annotations" / args.test_annotations_file),
    }
    img_folder, anno_file = PATHS[image_set]
    dataset = DSGGDatasetOO(img_folder, anno_file, transforms=make_coco_transforms(image_set), cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), is_train=(not args.eval))

    return dataset