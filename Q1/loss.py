import torch
import torch.nn.functional as F

def calculate_loss(cls_logits, box_preds, targets, model_stride=32, cls_weight=1.0, reg_weight=1.0):

    batch_size, _, feature_h, feature_w = cls_logits.shape
    device = cls_logits.device

    # Prepare targets
    cls_target = torch.zeros_like(cls_logits, device=device)
    reg_target = torch.zeros_like(box_preds, device=device)
    reg_mask = torch.zeros_like(cls_logits, dtype=torch.bool, device=device) # Mask for cells with assigned boxes

    total_gt_boxes = 0

    for i in range(batch_size): # iterate through images in the batch
        target = targets[i]
        # makes sure target exists and has 'boxes'
        if target is None or 'boxes' not in target:
            print(f"Warning: Target {i} is None or missing 'boxes'. Skipping.")
            continue

        gt_boxes = target['boxes'].to(device)
        num_gt = gt_boxes.shape[0]
        total_gt_boxes += num_gt

        if num_gt == 0:
            continue # missing ground truth boxes for this image

        # centers of ground truth boxes
        gt_center_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
        gt_center_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0

        # Target Assignment, assigns GT box to cell containing its center
        # Calculate which grid cell each GT box center falls into
        gt_grid_x_idx = (gt_center_x / model_stride).long().clamp(0, feature_w - 1)
        gt_grid_y_idx = (gt_center_y / model_stride).long().clamp(0, feature_h - 1)

        # check if indices are valid before using them
        if gt_grid_x_idx.numel() == 0 or gt_grid_y_idx.numel() == 0:
             continue

        # positive label (1.0) to the cells containing GT centers
        cls_target[i, 0, gt_grid_y_idx, gt_grid_x_idx] = 1.0

        # Assign regression targets ONLY to these positive cells
        gt_boxes_transposed = gt_boxes.T
        reg_target[i, :, gt_grid_y_idx, gt_grid_x_idx] = gt_boxes_transposed

        # Set the regression mask for these cells to True
        reg_mask[i, 0, gt_grid_y_idx, gt_grid_x_idx] = True

    # Calculate Losses
    # Classification Loss
    cls_loss = F.binary_cross_entropy_with_logits(cls_logits, cls_target, reduction='mean')
    # Regression Loss (Smooth L1 Loss)
    reg_mask_expanded = reg_mask.expand_as(reg_target)
    box_preds_masked = box_preds[reg_mask_expanded]
    reg_target_masked = reg_target[reg_mask_expanded]

    num_positive_cells = reg_mask.sum().item()
    if num_positive_cells > 0:
        reg_loss = F.smooth_l1_loss(box_preds_masked.view(-1, 4),
                                    reg_target_masked.view(-1, 4),
                                    reduction='sum')
        reg_loss = reg_loss / num_positive_cells # Normalize by number of positive assignments
    else:
        reg_loss = torch.tensor(0.0, device=device)

    #Total Loss
    total_loss = (cls_weight * cls_loss) + (reg_weight * reg_loss)

    return total_loss, cls_weight * cls_loss, reg_weight * reg_loss