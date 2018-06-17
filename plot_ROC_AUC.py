
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc

def plot_ROC_AUC(name_classes, y_test, y_score):
    n_classes = len(name_classes)
    lw = 2 # linewidth
    
    # Find false/true positive rate
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','darksalmon'])
    for i, color, cls in zip(range(n_classes), colors, name_classes):
        plt.plot(fpr[i], tpr[i], color = color, lw = lw,
         label='ROC curve of class {0} (area = {1:0.2f})'
         ''.format(cls, roc_auc[i]))
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc = "lower right")
    plt.show()