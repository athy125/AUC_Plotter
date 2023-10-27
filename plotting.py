import matplotlib.pyplot as plt
from itertools import cycle

def plot_roc_curve(fpr, tpr, roc_auc, multi_class=False):
    """
    Plot ROC curve(s) with AUC scores for binary or multiclass classification tasks.

    Parameters:
    - fpr: Dictionary containing false positive rates.
    - tpr: Dictionary containing true positive rates.
    - roc_auc: Dictionary containing AUC scores for each class.
    - multi_class: Set to True for multiclass problems.

    This function can plot multiple ROC curves and their AUC scores if multi_class is True.
    """
    if not multi_class:
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    else:
        plt.figure(figsize=(8, 6))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(roc_auc.keys(), colors):
            plt.plot(fpr[i], tpr[i], lw=2, color=color, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')

