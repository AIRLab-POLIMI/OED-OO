import torch
from collections import defaultdict, Counter
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import numpy as np
from util.box_ops import box_cxcywh_to_xyxy, box_iou


REL_GROUPS = {
    "spatial":    [3, 4, 5, 6, 7, 26, 27, 28, 29],      # old spatial + new ones
    "contacting": list(range(8, 26)),               # 8..25
}
STAR_RELCAT_MAPPING = {'r000' : 26, 'r001' : 6, 'r002' : 5, 'r003' : 7, 'r004' : 3, 'r005' : 4, 'r006' : 11, 
                       'r007' : 13, 'r008' : 23, 'r009' : 14, 'r010' : 16, 'r011' : 10, 'r012' : 9, 'r013' : 12,
                       'r014' : 15, 'r015' : 19, 'r016' : 22, 'r017' : 25, 'r018' : 20, 'r019' : 21, 'r020' : 24, 'r021' : 27,
                       'r022' : 28, 'r023' : 29}
PERSON_ID = 1
D1_SPATIAL_IDS = list(range(3, 9))       # ds1: spatial_rel - 3  -> ids 3..8
D1_CONTACT_IDS = list(range(9, 26))      # ds1: contact_rel - 9  -> ids 9..25
STAR_IDS = set(STAR_RELCAT_MAPPING.values())

# ---------- fast target getters (no image reads, no transforms) ----------
def _identity(image, target):
    return image, target

def make_fast_getter(ds):
    prepare = type(ds.prepare)(_identity)
    coco = ds.coco

    def get(idx):
        img_id = ds.ids[idx]
        info = coco.loadImgs(img_id)[0]
        img = Image.new('RGB', (info['width'], info['height']))
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        return prepare(img, {'image_id': img_id, 'annotations': anns})[1]

    return get


# ---------- helpers ----------
def _row_ids(row, id_list):
    return {id_list[i] for i in torch.nonzero(row > 0.5).flatten().tolist()}

def _triplets(t, spatial_ids, contact_ids, subj_labels=None):
    n = len(t['obj_labels'])
    if n == 0:
        return torch.zeros(0, 8), torch.zeros(0, 2, dtype=torch.int64), [], []
    boxes = torch.cat([t['sub_boxes'], t['obj_boxes']], dim=1)
    subj = subj_labels if subj_labels is not None else t['subj_labels']
    labels = torch.stack([subj, t['obj_labels']], dim=1)
    sp = [_row_ids(r, spatial_ids) for r in t['spatial_labels']]
    co = [_row_ids(r, contact_ids) for r in t['contacting_labels']]
    return boxes, labels, sp, co

def _iou(a, b):
    """a: (n,4) xyxy, b: (m,4) xyxy -> (n,m)"""
    if len(a) == 0 or len(b) == 0:
        return torch.zeros(len(a), len(b))
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-9)


# ---------- main check ----------
def check_consistency(ds1, ds2, tol=1e-3, max_images=None, max_report=20, near_dup_iou=0.8):
    """
    ds1: original AG dataset (DSGGDataset), ds2: modified dataset (DSGGDatasetOO).
    Expected, after applying the STAR filtering to ds1:
      - every filtered ds1 triplet is found in ds2 (ds1_triplet_missing_in_ds2 == 0)
      - its relations are a subset of the ds2 ones (relation_mismatch == 0)
    ds2 may have extra relations / triplets coming from STAR.
    """
    get1, get2 = make_fast_getter(ds1), make_fast_getter(ds2)
    id2idx_2 = {img_id: i for i, img_id in enumerate(ds2.ids)}
    sp2_ids, co2_ids = REL_GROUPS['spatial'], REL_GROUPS['contacting']

    stats, problems = defaultdict(int), []
    mismatch_kinds = Counter()          # (ids only in ds1, ids only in ds2) -> count

    total = min(len(ds1.ids), max_images) if max_images else len(ds1.ids)
    for i, img_id in enumerate(tqdm(ds1.ids, total=total)):
        if max_images is not None and i >= max_images:
            break
        if img_id not in id2idx_2:
            stats['image_missing_in_ds2'] += 1
            continue
        stats['images_compared'] += 1

        try:
            t1 = get1(i)
        except Exception as e:
            stats['ds1_error'] += 1
            problems.append((img_id, f'ds1 raised {type(e).__name__}: {e}'))
            continue
        try:
            t2 = get2(id2idx_2[img_id])
        except Exception as e:
            stats['ds2_error'] += 1
            problems.append((img_id, f'ds2 raised {type(e).__name__}: {e}'))
            continue

        # ds1: subject is always the person
        n1_raw = len(t1['obj_labels'])
        subj1 = torch.full((n1_raw,), PERSON_ID, dtype=torch.int64)
        b1, l1, sp1, co1 = _triplets(t1, D1_SPATIAL_IDS, D1_CONTACT_IDS, subj_labels=subj1)

        # apply the same filtering as the conversion script:
        # drop relation ids not in the STAR vocabulary, then drop triplets left with no relations
        sp1 = [s & STAR_IDS for s in sp1]
        co1 = [c & STAR_IDS for c in co1]
        keep1 = [k for k in range(len(l1)) if sp1[k] or co1[k]]
        stats['ds1_triplets_removed_by_star_filter'] += n1_raw - len(keep1)
        b1, l1 = b1[keep1], l1[keep1]
        sp1, co1 = [sp1[k] for k in keep1], [co1[k] for k in keep1]
        n1 = len(l1)

        b2, l2, sp2, co2 = _triplets(t2, sp2_ids, co2_ids)
        n2 = len(l2)

        # match by content (labels + boxes), independent of order
        matched1, matched2 = set(), set()
        if n1 and n2:
            cost = (b1[:, None] - b2[None]).abs().amax(-1)
            cost = cost + 1e6 * (l1[:, None] != l2[None]).any(-1).float()
            r, c = linear_sum_assignment(cost.numpy())
            for a, b in zip(r, c):
                if cost[a, b] >= tol:
                    continue
                matched1.add(int(a)); matched2.add(int(b))
                stats['triplets_matched'] += 1
                if not (sp1[a] <= sp2[b] and co1[a] <= co2[b]):
                    stats['relation_mismatch'] += 1
                    only1 = tuple(sorted((sp1[a] | co1[a]) - (sp2[b] | co2[b])))
                    only2 = tuple(sorted((sp2[b] | co2[b]) - (sp1[a] | co1[a])))
                    mismatch_kinds[(only1, only2)] += 1
                    problems.append((img_id, f'rel mismatch ds1 sp={sorted(sp1[a])} co={sorted(co1[a])} '
                                             f'vs ds2 sp={sorted(sp2[b])} co={sorted(co2[b])}'))
                elif sp1[a] != sp2[b] or co1[a] != co2[b]:
                    stats['ds2_has_extra_rels'] += 1          # expected: STAR rels merged in

        for a in sorted(set(range(n1)) - matched1):
            stats['ds1_triplet_missing_in_ds2'] += 1
            problems.append((img_id, f'ds1 triplet {a} (obj={l1[a, 1].item()}) has no match in ds2'))

        for b in sorted(set(range(n2)) - matched2):
            if l2[b, 0].item() != PERSON_ID:
                stats['ds2_extra_obj_obj_triplets'] += 1      # expected
                continue
            stats['ds2_person_triplet_not_in_ds1'] += 1       # expected: created from STAR
            # is it a near-duplicate of an existing ds1 triplet (same obj class, overlapping boxes)?
            if n1:
                same_cls = (l1[:, 1] == l2[b, 1])
                if same_cls.any():
                    iou = _iou(b1[:, 4:], b2[b:b + 1, 4:])[:, 0]
                    if (iou * same_cls.float()).max() > near_dup_iou:
                        stats['ds2_person_triplet_near_duplicate_of_ds1'] += 1

    print('--- summary ---')
    for k, v in sorted(stats.items()):
        print(f'{k:45s} {v}')
    if mismatch_kinds:
        print('--- relation mismatch kinds: (only in ds1, only in ds2) -> count ---')
        for k, v in mismatch_kinds.most_common():
            print(k, v)
    print(f'--- first {max_report} problems ---')
    for p in problems[:max_report]:
        print(p)
    return stats, problems


# ---------- debugging helpers ----------
def show_raw(ds, img_id):
    keys = ('id', 'category_id', 'bbox', 'spatial_rel', 'contact_rel', 'attention_rel',
            'isobjobj', 'subj_id', 'subj_bbox')
    for a in ds.coco.loadAnns(ds.coco.getAnnIds(imgIds=img_id)):
        print({k: a[k] for k in keys if k in a})

def explain_unmatched(img_id, ds1, ds2):
    get1, get2 = make_fast_getter(ds1), make_fast_getter(ds2)
    t1, t2 = get1(ds1.ids.index(img_id)), get2(ds2.ids.index(img_id))
    for a in range(len(t1['obj_labels'])):
        for b in range(len(t2['obj_labels'])):
            if t1['obj_labels'][a] == t2['obj_labels'][b]:
                print(f'ds1[{a}] vs ds2[{b}] obj={t1["obj_labels"][a].item()}')
                print('  sub_box ds1', t1['sub_boxes'][a].tolist(), 'ds2', t2['sub_boxes'][b].tolist())
                print('  obj_box ds1', t1['obj_boxes'][a].tolist(), 'ds2', t2['obj_boxes'][b].tolist())


# ---------- objobj diagnostic ----------
@torch.no_grad()
def diagnose_objobj(gt_list, pred_list, person_id=1, iou_thr=0.5):
    """gt_list: list of target dicts (per frame); pred_list: list of PostProcessDSGGOO outputs (per frame)."""
    keys = ['total', 'sub_box_ok', 'obj_box_ok', 'both_box_ok',
            'subj_cls_ok', 'obj_cls_ok', 'both_cls_ok', 'pred_subj_is_person']
    stats = {k: 0 for k in keys}
    frames = 0

    for g, p in zip(gt_list, pred_list):
        n = len(g['obj_labels'])
        if n == 0:
            continue
        subj_gt = g['subj_labels'].cpu()
        oo = (subj_gt != person_id).nonzero().flatten().tolist()   # indices of obj-obj GT triplets
        if not oo:
            continue
        frames += 1

        pb = box_cxcywh_to_xyxy(torch.as_tensor(p['pred_boxes']).float())   # [subjects(0..Q-1), objects(Q..2Q-1)]
        Q = pb.shape[0] // 2
        ps, po = pb[:Q], pb[Q:]
        plab = np.asarray(p['pred_labels'])

        for i in oo:
            stats['total'] += 1
            gs = box_cxcywh_to_xyxy(g['sub_boxes'][i:i+1].cpu().float())
            go = box_cxcywh_to_xyxy(g['obj_boxes'][i:i+1].cpu().float())
            iou_s = box_iou(gs, ps)[0][0]          # box_iou returns (iou, union); iou is (1, Q)
            iou_o = box_iou(go, po)[0][0]
            q = int((iou_s + iou_o).argmax())      # query that best localizes both boxes

            sub_ok = bool(iou_s[q] >= iou_thr)
            obj_ok = bool(iou_o[q] >= iou_thr)
            subj_ok = int(plab[q]) == int(subj_gt[i])
            objc_ok = int(plab[Q + q]) == int(g['obj_labels'][i])

            stats['sub_box_ok'] += sub_ok
            stats['obj_box_ok'] += obj_ok
            stats['both_box_ok'] += sub_ok and obj_ok
            stats['subj_cls_ok'] += subj_ok
            stats['obj_cls_ok'] += objc_ok
            stats['both_cls_ok'] += subj_ok and objc_ok
            stats['pred_subj_is_person'] += int(plab[q]) == person_id

    print(f'--- obj-obj diagnosis: {stats["total"]} GT triplets in {frames} frames ---')
    t = max(stats['total'], 1)
    for k in keys[1:]:
        print(f'{k:22s} {stats[k]:5d}  ({100 * stats[k] / t:.1f}%)')
    return stats
