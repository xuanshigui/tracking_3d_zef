import torch


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    The bounding boxes are expected to be in the format (x1, y1, x2, y2).

    Parameters:
    - box1: tensor of shape (4,)
    - box2: tensor of shape (4,)

    Returns:
    - iou: float, IoU between box1 and box2
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union = box1_area + box2_area - intersection

    return intersection / union


def merge_bounding_boxes(boxes, scores, categories, features):
    """
    Merge bounding boxes if IoU is 1 and sum their confidence scores.

    Parameters:
    - boxes: tensor of shape (n, 4)
    - scores: tensor of shape (n,)
    - categories: tensor of shape (n,)
    - features: tensor of shape (n, f)

    Returns:
    - merged_boxes: tensor of shape (m, 4)
    - merged_scores: tensor of shape (m,)
    - merged_categories: tensor of shape (m,)
    - merged_features: tensor of shape (m, f)
    """
    n = boxes.shape[0]
    merged_boxes = []
    merged_scores = []
    merged_categories = []
    merged_features = []
    merged = [False] * n

    if categories[0] == 1:
        categories = categories
    else:
        categories = torch.full(categories.shape, 2)

    for i in range(n):
        if merged[i]:
            continue

        current_box = boxes[i]
        current_score = scores[i]
        current_category = categories[i]
        current_feature = features[i]

        for j in range(i + 1, n):
            if merged[j]:
                continue

            if calculate_iou(current_box, boxes[j]) >= 0.95:
                current_box[0] = min(current_box[0], boxes[j][0])
                current_box[1] = min(current_box[1], boxes[j][1])
                current_box[2] = max(current_box[2], boxes[j][2])
                current_box[3] = max(current_box[3], boxes[j][3])
                current_score += scores[j]

                # Choose the category and features with the higher score
                if scores[j] > current_score:
                    current_category = categories[j]
                    current_feature = features[j]

                merged[j] = True

        merged_boxes.append(current_box)
        merged_scores.append(current_score)
        merged_categories.append(current_category)
        merged_features.append(current_feature)
        merged[i] = True

    return torch.stack(merged_boxes), torch.tensor(merged_scores), torch.tensor(merged_categories), torch.stack(
        merged_features)

