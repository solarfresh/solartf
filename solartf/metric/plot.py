import itertools
import matplotlib.pyplot as plt
import numpy as np
from .score import ClassificationMetrics


class ClassificationMetricPlot:
    def __init__(self, clf_metrics: ClassificationMetrics,
                 cm_title='Confusion matrix',
                 cm_norm=True,
                 roc_title='Receiver operating characteristic curve',
                 roc_lw=3,
                 cmap=None,
                 figsize=(15, 15)):
        self.clf_metrics = clf_metrics
        self.cm_title = cm_title
        self.cm_norm = cm_norm
        self.roc_title = roc_title
        self.roc_lw = roc_lw

        self.cmap = cmap
        self.figsize = figsize

        self.fig, self.axs = plt.subplots(1, 2, figsize=self.figsize)
        self.axs = self.axs.ravel()

    def plot_confusion_matrix(self):
        accuracy = np.trace(self.clf_metrics.cm) / float(np.sum(self.clf_metrics.cm))
        misclass = 1 - accuracy

        if self.cmap is None:
            cmap = plt.get_cmap('Blues')
        else:
            cmap = self.cmap

        self.axs[0].imshow(self.clf_metrics.cm, interpolation='nearest', cmap=cmap)
        self.axs[0].set_title(self.cm_title)
        # self.axs[0].colorbar()

        tick_marks = np.arange(len(self.clf_metrics.label_name))
        self.axs[0].set_xticks(tick_marks)
        self.axs[0].set_xticklabels(self.clf_metrics.label_name, rotation=45)
        self.axs[0].set_yticks(tick_marks)
        self.axs[0].set_yticklabels(self.clf_metrics.label_name)

        if self.cm_norm:
            cm = self.clf_metrics.cm.astype('float') / self.clf_metrics.cm.sum(axis=1)[:, np.newaxis]
        else:
            cm = self.clf_metrics.cm

        thresh = cm.max() / 1.5 if self.cm_norm else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if self.cm_norm:
                self.axs[0].text(j, i, "{:0.4f}".format(cm[i, j]),
                                 horizontalalignment="center",
                                 color="white" if cm[i, j] > thresh else "black")
            else:
                self.axs[0].text(j, i, "{:,}".format(cm[i, j]),
                                 horizontalalignment="center",
                                 color="white" if cm[i, j] > thresh else "black")

        self.axs[0].set_ylabel('True label')
        self.axs[0].set_xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        return self

    def plot_roc_curve(self):
        self.axs[1].plot(self.clf_metrics.fpr["micro"], self.clf_metrics.tpr["micro"],
                         label='micro-average ROC curve (area = {0:0.2f})'
                               ''.format(self.clf_metrics.roc_auc["micro"]),
                         color='deeppink', linestyle=':', linewidth=4)

        self.axs[1].plot(self.clf_metrics.fpr["macro"], self.clf_metrics.tpr["macro"],
                         label='macro-average ROC curve (area = {0:0.2f})'
                               ''.format(self.clf_metrics.roc_auc["macro"]),
                         color='navy', linestyle=':', linewidth=4)

        colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for label_name, color in zip(self.clf_metrics.label_name, colors):
            plt.plot(self.clf_metrics.fpr[label_name], self.clf_metrics.tpr[label_name], color=color, lw=self.roc_lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(label_name, self.clf_metrics.roc_auc[label_name]))

        self.axs[1].plot([0, 1], [0, 1], 'k--', lw=self.roc_lw)
        self.axs[1].set_xlim([0.0, 1.0])
        self.axs[1].set_ylim([0.0, 1.05])
        self.axs[1].set_xlabel('False Positive Rate')
        self.axs[1].set_ylabel('True Positive Rate')
        self.axs[1].set_title(self.roc_title)
        self.axs[1].legend(loc="lower right")
        return self

    def show(self):
        self.plot_confusion_matrix().plot_roc_curve()
        plt.show()
