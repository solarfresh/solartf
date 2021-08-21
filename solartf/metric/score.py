import numpy as np
from scipy import interp
from sklearn.metrics import (confusion_matrix, roc_curve, auc)
from time import perf_counter


class ClassificationMetrics:
    def __init__(self, y_truth, y_pred, label_name=None):
        """
        y_truth and y_pred are arrays in categorical representation
        """
        self.y_truth = y_truth
        self.y_pred = y_pred
        self.label_name = np.arange(self.y_truth.shape[-1]) if label_name is None else np.array(label_name)

        self.fpr = dict()
        self.tpr = dict()
        self.roc_auc = dict()

        self.cm = self._get_cm(self.y_truth, self.y_pred)
        self._get_roc(self.y_truth, self.y_pred)

    @property
    def confusion_matrix(self):
        return self.cm

    def _get_cm(self, y_truth, y_pred):
        y_truth = np.argmax(y_truth, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)
        return confusion_matrix(y_truth, y_pred)

    def _get_roc(self, y_truth, y_pred):
        for i, label_name in enumerate(self.label_name):
            self.fpr[label_name], self.tpr[label_name], _ = roc_curve(y_truth[:, i], y_pred[:, i])
            self.roc_auc[label_name] = auc(self.fpr[label_name], self.tpr[label_name])

        self.fpr["micro"], self.tpr["micro"], _ = roc_curve(y_truth.ravel(), y_pred.ravel())
        self.roc_auc["micro"] = auc(self.fpr["micro"], self.tpr["micro"])

        all_fpr = np.unique(np.concatenate([self.fpr[label_name] for label_name in self.label_name]))
        mean_tpr = np.zeros_like(all_fpr)
        for label_name in self.label_name:
            mean_tpr += interp(all_fpr, self.fpr[label_name], self.tpr[label_name])

        mean_tpr /= self.label_name.size

        self.fpr["macro"] = all_fpr
        self.tpr["macro"] = mean_tpr
        self.roc_auc["macro"] = auc(self.fpr["macro"], self.tpr["macro"])
