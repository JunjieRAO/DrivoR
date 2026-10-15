"""FP32, per-layer Hungarian matching and mask-aware current-geometry losses."""
import torch
import torch.nn.functional as F
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment


def focal(logits, labels):
    probability = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    pt = labels * probability + (1 - labels) * (1 - probability)
    alpha = labels * 0.25 + (1 - labels) * 0.75
    return alpha * (1 - pt).pow(2) * ce


def pair_cost(pred, gt, mask, kind, classes):
    if kind == "agent":
        weights = pred.new_tensor([2, 2, .5, .5, .5, .5])
        cost = ((pred[:, None] - gt[None]).abs() * mask[None] * weights).sum(-1)
        return cost, None
    p, g = pred.reshape(-1, gt.shape[-1] // 2, 2), gt.reshape(-1, gt.shape[-1] // 2, 2)
    direct = (p[:, None] - g[None]).abs().mean((-1, -2))
    reverse = (p[:, None] - g.flip(1)[None]).abs().mean((-1, -2))
    flip = (classes[None] == 0) & (reverse < direct)
    return 10 * torch.where(flip, reverse, direct), flip


def layer_loss(prediction, targets, kind):
    logits, geometry = prediction["logits"].float(), prediction["geometry"].float()
    zero = logits.sum() * 0 + geometry.sum() * 0
    cls_loss, reg_loss, direction_loss = zero, zero, zero
    positives = geometry.new_zeros(())
    xy_error, matched_count, hits, gt_count = [zero.detach().clone() for _ in range(4)]
    for b in range(logits.shape[0]):
        if not targets[f"sg_{kind}_available"][b]:
            continue
        valid = targets[f"sg_{kind}_valid"][b].bool()
        gt = targets[f"sg_{kind}_geometry"][b, valid].float()
        classes = targets[f"sg_{kind}_class"][b, valid].long()
        mask = targets[f"sg_{kind}_mask"][b, valid].float()
        labels = torch.zeros_like(logits[b])
        positives += len(gt)
        gt_count += len(gt)
        if len(gt):
            with torch.no_grad():
                cost, flips = pair_cost(geometry[b], gt, mask, kind, classes)
                prob = logits[b].sigmoid().clamp(1e-6, 1 - 1e-6)
                pos = -.25 * (1 - prob).pow(2) * prob.log()
                neg = -.75 * prob.pow(2) * (1 - prob).log()
                cost = cost + pos[:, classes] - neg[:, classes]
                row, col = linear_sum_assignment(cost.detach().cpu().numpy())
                row = torch.as_tensor(row, device=logits.device)
                col = torch.as_tensor(col, device=logits.device)
            labels[row, classes[col]] = 1
            matched = gt[col]
            if kind == "map":
                matched = matched.reshape(len(col), -1, 2)
                matched = torch.where(flips[row, col, None, None], matched.flip(1), matched)
                matched = matched.flatten(1)
            delta = (geometry[b, row] - matched).abs()
            if kind == "agent":
                weights = delta.new_tensor([2, 2, .5, .5, .5, .5])
                reg_loss = reg_loss + (delta * mask[col] * weights).sum()
                xy_error += delta[:, :2].detach().mean(-1).sum()
            else:
                reg_loss = reg_loss + 10 * delta.mean(-1).sum()
                p = geometry[b, row].reshape(len(row), -1, 2)
                g = matched.reshape(len(row), -1, 2)
                directed = classes[col] == 1
                direction_loss = direction_loss + ((1 - F.cosine_similarity(
                    p[:, 1:] - p[:, :-1], g[:, 1:] - g[:, :-1], dim=-1)).mean(-1) * directed).sum()
                xy_error += delta.detach().mean(-1).sum()
            matched_count += len(row)
            hits += (logits[b, row].sigmoid().argmax(-1) == classes[col]).sum().detach()
        cls_loss = cls_loss + focal(logits[b], labels).sum()
    # DDP averages gradients; normalize by the global mean positive count.
    normalizer = positives.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(normalizer)
        normalizer /= dist.get_world_size()
    normalizer = normalizer.clamp_min(1)
    return (cls_loss + reg_loss + direction_loss) / normalizer, {
        f"geometry/{kind}_matched_normalized_l1": xy_error / matched_count.clamp_min(1),
        f"geometry/{kind}_matched_class_accuracy": hits / matched_count.clamp_min(1),
        f"geometry/{kind}_matched_fraction": matched_count / gt_count.clamp_min(1),
    }


def geometry_loss(predictions, targets):
    result, total = {}, None
    # Explicit FP32 path also under Lightning mixed precision.
    device = predictions["agent"][0]["geometry"].device
    with torch.autocast(device_type=device.type, enabled=False):
        for kind in ("agent", "map"):
            layers = predictions[kind]
            weights = [0.5] * (len(layers) - 1) + [1.0]
            loss = 0
            for layer, weight in zip(layers, weights):
                value, metrics = layer_loss(layer, targets, kind)
                loss = loss + weight * value / sum(weights)
            result[f"geometry/{kind}_loss"] = loss
            result.update(metrics)
            total = loss if total is None else total + loss
    result["geometry/loss"] = total
    return result
