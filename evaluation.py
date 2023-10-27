from sklearn.metrics import roc_curve, auc, roc_auc_score, plot_roc_curve
import matplotlib.pyplot as plt

def calculate_roc_auc(clf, X_test, y_test, multi_class=False, average="macro", plot=False):
    """
    Calculate ROC curve and AUC for binary or multiclass classification tasks.

    Parameters:
    - clf: Classifier model.
    - X_test: Test data.
    - y_test: True labels.
    - multi_class: Set to True for multiclass problems.
    - average: For multiclass, specify averaging method (e.g., "macro", "micro").
    - plot: Set to True to plot the ROC curve.

    Returns:
    - Dictionary containing ROC curve data and AUC scores.
    """
    if not multi_class:
        y_probs = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
    else:
        # For multiclass, calculate AUC for each class
        n_classes = clf.classes_.shape[0]
        fpr, tpr, roc_auc = {}, {}, {}

        for i in range(n_classes):
            y_true = (y_test == clf.classes_[i]).astype(int)
            y_probs = clf.predict_proba(X_test)[:, i]
            fpr[i], tpr[i], _ = roc_curve(y_true, y_probs)
            roc_auc[i] = auc(fpr[i], tpr[i])

    # If specified, plot the ROC curve
    if plot:
        if not multi_class:
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        else:
            plt.figure(figsize=(8, 6))
            for i in range(n_classes):
                plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')

    return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}
:
