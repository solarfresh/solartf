import numpy as np
from scipy import interp
from sklearn.metrics import (confusion_matrix, roc_curve, auc)
from time import perf_counter


class ClassificationMetrics:
    def __init__(self, y_true, y_pred, label_name=None):
        """
        y_true and y_pred are arrays in categorical representation
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.label_name = np.arange(self.y_true.shape[-1]) if label_name is None else np.array(label_name)

        self.fpr = dict()
        self.tpr = dict()
        self.roc_auc = dict()

        self.cm = self._get_cm(self.y_true, self.y_pred)
        self._get_roc(self.y_true, self.y_pred)

    @property
    def confusion_matrix(self):
        return self.cm

    def _get_cm(self, y_true, y_pred):
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)
        return confusion_matrix(y_true, y_pred)

    def _get_roc(self, y_true, y_pred):
        for i, label_name in enumerate(self.label_name):
            self.fpr[label_name], self.tpr[label_name], _ = roc_curve(y_true[:, i], y_pred[:, i])
            self.roc_auc[label_name] = auc(self.fpr[label_name], self.tpr[label_name])

        self.fpr["micro"], self.tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        self.roc_auc["micro"] = auc(self.fpr["micro"], self.tpr["micro"])

        all_fpr = np.unique(np.concatenate([self.fpr[label_name] for label_name in self.label_name]))
        mean_tpr = np.zeros_like(all_fpr)
        for label_name in self.label_name:
            mean_tpr += interp(all_fpr, self.fpr[label_name], self.tpr[label_name])

        mean_tpr /= self.label_name.size

        self.fpr["macro"] = all_fpr
        self.tpr["macro"] = mean_tpr
        self.roc_auc["macro"] = auc(self.fpr["macro"], self.tpr["macro"])


class DetectionMetrics:
    def _compute_ap(self, precisions, recalls):
        """
        Compute average precision of a particular classes (cls_idx)
        :param cls:
        :return:
        """
        previous_recall = 0
        average_precision = 0
        for precision, recall in zip(precisions[::-1], recalls[::-1]):
            average_precision += precision * (recall - previous_recall)
            previous_recall = recall
        return average_precision

    def _compute_precisions(self, class_id, interpolated=True):
        class_accumulators = self.total_accumulators[class_id]
        precisions = [class_accumulators[pr_scale].precision for pr_scale in self.pr_scale]
        if interpolated:
            interpolated_precision = []
            for precision in precisions:
                last_max = 0
                if interpolated_precision:
                    last_max = max(interpolated_precision)
                interpolated_precision.append(max(precision, last_max))
            precisions = interpolated_precision
        return precisions

    def _compute_recalls(self, class_id):
        class_accumulators = self.total_accumulators[class_id]
        recalls = [class_accumulators[pr_scale].recall for pr_scale in self.pr_scale]
        return recalls

    def print_total(self):
        print(f'There is/are {self.n_class} class(es)...')
        print(f'The overlap threshold was set as {self.overlap_threshold}')
        print(f'Precision scale is labeled as {self.pr_scale}')
        for class_id in range(self.n_class):
            print(f'==== For class ID: {class_id} ====')
            precisions = self._compute_precisions(class_id)
            recalls = self._compute_recalls(class_id)
            print(f'Precisions are {precisions}')
            print(f'Recalls are {recalls}')
            print(f'Average precision is {self._compute_ap(precisions=precisions, recalls=recalls)}')

    @property
    def evaluation_results(self):
        results = {'n_class': self.n_class, 'metric': []}
        for class_id in range(self.n_class):
            # class_id comes from annotation rather than models
            precisions = self._compute_precisions(class_id)
            recalls = self._compute_recalls(class_id)
            results['metric'].append({
                'class_id': class_id,
                'overlap_threshold': self.overlap_threshold,
                'precisions': precisions,
                'recalls': recalls,
                'ap': self._compute_ap(precisions=precisions, recalls=recalls)
            })

        return results


class KeypointsMetrics:
    def __init__(self, y_true, y_pred, label_name=None, nme_factor=None):
        """
        y_true and y_pred are arrays in categorical representation
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.label_name = np.arange(self.y_true.shape[-1] // 2) if label_name is None else np.array(label_name)
        self.nme_factor = nme_factor

        self.nme = dict()
        self._compute_nme(self.y_true, self.y_pred)

    def _compute_nme(self, y_true, y_pred):
        nme_factor = 1. if self.nme_factor is None else self.nme_factor
        y_true = y_true.astype(np.float32) / nme_factor
        y_pred = y_pred.astype(np.float32) / nme_factor
        total_dist = 0.
        for i, label_name in enumerate(self.label_name):
            dist = np.linalg.norm(y_true[:, 2 * i:2 * (i + 1)] - y_pred[:, 2 * i:2 * (i + 1)])
            self.nme[label_name] = dist ** 2 / (y_true.shape[0] * 2)
            total_dist += self.nme[label_name]

        self.nme['total'] = total_dist / self.label_name.size


class PerformanceMetrics:
    def __init__(self, time_window=1.0):
        # 'time_window' defines the length of the timespan over which the 'current fps' value is calculated
        self.time_window_size = time_window
        self.last_moving_statistic = Statistic()
        self.current_moving_statistic = Statistic()
        self.total_statistic = Statistic()
        self.last_update_time = None

    def update(self, last_request_start_time):
        current_time = perf_counter()

        if self.last_update_time is None:
            self.last_update_time = current_time
            return

        self.current_moving_statistic.latency += current_time - last_request_start_time
        self.current_moving_statistic.period = current_time - self.last_update_time
        self.current_moving_statistic.frame_count += 1

        if current_time - self.last_update_time > self.time_window_size:
            self.last_moving_statistic = self.current_moving_statistic
            self.total_statistic.combine(self.last_moving_statistic)
            self.current_moving_statistic = Statistic()

            self.last_update_time = current_time

        # Draw performance stats over frame
        # current_latency, current_fps = self.get_last()

    def get_last(self):
        return (self.last_moving_statistic.latency / self.last_moving_statistic.frame_count
                if self.last_moving_statistic.frame_count != 0
                else None,
                self.last_moving_statistic.frame_count / self.last_moving_statistic.period
                if self.last_moving_statistic.period != 0.0
                else None)

    def get_total(self):
        frame_count = self.total_statistic.frame_count + self.current_moving_statistic.frame_count
        return (((self.total_statistic.latency + self.current_moving_statistic.latency) / frame_count)
                if frame_count != 0
                else None,
                (frame_count / (self.total_statistic.period + self.current_moving_statistic.period))
                if frame_count != 0
                else None)

    def print_total(self):
        total_latency, total_fps = self.get_total()
        print("Latency: {:.1f} ms".format(total_latency * 1e3) if total_latency is not None else "Latency: N/A")
        print("FPS: {:.1f}".format(total_fps) if total_fps is not None else "FPS: N/A")


class Statistic:
    def __init__(self):
        self.latency = 0.0
        self.period = 0.0
        self.frame_count = 0

    def combine(self, other):
        self.latency += other.latency
        self.period += other.period
        self.frame_count += other.frame_count
