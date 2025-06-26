import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import mplhep as hep
hep.style.use("CMS")

def plot_roc_curve(labels, scores, save_path):
    """Calculates and plots the ROC curve, saving it to a file."""
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    
    plt.figure()
    plt.plot(fpr, tpr, color='cornflowerblue', lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: t -> bqq vs q/g jets')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def plot_score_histogram(labels, scores, save_path):
    """Plots and saves a histogram of classifier scores for signal vs background."""
    scores_signal = scores[labels == 1]
    scores_background = scores[labels == 0]
    
    plt.figure()
    plt.hist(scores_background, bins=40, range=(0, 2), color='cornflowerblue', alpha=0.7, label='q/g jets')
    plt.hist(scores_signal, bins=40, range=(0, 2), histtype='step', color='darkorange', lw=2, label='t -> bqq')
    plt.xlabel('Classifier Score')
    plt.ylabel('No. of events')
    plt.legend(loc='upper left')
    plt.savefig(save_path)
    plt.close()