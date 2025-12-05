import numpy as np

import argparse

def cal_f1score(pred_cm, target_cm):
    """
    Calculate the F1-score for predicted contact maps compared to a target contact map.

    Args:
        pred_cm (np.ndarray): Predicted contact map, (L, L).
        target_cm (np.ndarray): Target contact map, (L, L).

    Returns:
        float: F1-score.
    """
    pred_pair_num = np.sum(pred_cm)  # total predicted positive
    target_pair_num = np.sum(target_cm)  # total target positive

    tp = np.sum(pred_cm * target_cm)

    precision = tp / pred_pair_num if pred_pair_num > 0 else 0.0
    recall = tp / target_pair_num if target_pair_num > 0 else 0.0

    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    fp = pred_pair_num - tp
    fn = target_pair_num - tp
    tn = pred_cm.size - (tp + fp + fn)
    
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0.0

    return f1_score, precision, recall, mcc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluating the secondary structure F1-score')
    parser.add_argument('--pred-cm', type=str, help='Path to the predicted 2D structure as .npy contact map')
    parser.add_argument('--gt-cm', type=str, help='Path to the ground truth 2D structure as .npy contact map')

    args = parser.parse_args()

    pred_cm_path = args.pred_cm
    gt_cm_path = args.gt_cm
    
    print(f'Predicted secondary structure: {pred_cm_path}')
    print(f'Ground-truth secondary structure: {gt_cm_path}')

    pred_cm = np.load(pred_cm_path)
    gt_cm = np.load(gt_cm_path)
    
    f1_score, precision, recall, mcc = cal_f1score(pred_cm, gt_cm)

    print(f'F1-score = {f1_score:.4f}')
    print(f'Precision = {precision:.4f}')
    print(f'Recall = {recall:.4f}')
    print(f'Interaction network fidelity (INF) = {mcc:.4f}')
