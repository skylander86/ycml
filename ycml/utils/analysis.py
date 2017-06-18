import logging
import os

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import label_binarize

from tabulate import tabulate

from .uriutils import uri_open

__all__ = ['classification_report', 'find_best_thresholds', 'generate_pr_curves']

logger = logging.getLogger(__name__)


def classification_report(Y_true, Y_proba, *, labels=None, target_names=None, thresholds=None, precision_thresholds=None):
    Y_true, Y_proba = _make_label_indicator(Y_true, Y_proba)
    Y_true, Y_proba, target_names = _filter_labels(Y_true, Y_proba, labels, target_names)
    n_classes = Y_true.shape[1]

    if isinstance(thresholds, float): thresholds = np.full(n_classes, thresholds)
    if thresholds is not None and n_classes == 2: thresholds[1] = 1.0 - thresholds[0]
    assert thresholds is None or ((thresholds <= 1).all() and (thresholds >= 0.0).all())

    if isinstance(precision_thresholds, float): precision_thresholds = np.full(n_classes, precision_thresholds)
    assert precision_thresholds is None or ((precision_thresholds <= 1).all() and (precision_thresholds >= 0.0).all())

    assert Y_true.shape[0] == Y_proba.shape[0]
    assert Y_true.shape[1] == Y_proba.shape[1]
    assert len(target_names) == n_classes

    table = []
    support_total, ap_score_total = 0.0, 0.0
    thresholds_best = np.zeros(n_classes)
    thresholds_minprec = np.zeros(n_classes)
    for i, name in enumerate(target_names):
        # Results using 0.5 as threshold
        p, r, f1, _ = precision_recall_fscore_support(Y_true[:, i], (Y_proba[:, i] >= 0.5), average='binary')  # Using thresholds
        support = Y_true[:, i].sum()
        ap_score = average_precision_score(Y_true[:, i], Y_proba[:, i])
        row = [name, '{:d}'.format(int(support)), '{:.3f}'.format(ap_score), '{:.3f}/{:.3f}/{:.3f}'.format(p, r, f1)]
        support_total += support
        ap_score_total += ap_score

        # Results using given thresholds
        if thresholds is not None:
            p, r, f1, _ = precision_recall_fscore_support(Y_true[:, i], Y_proba[:, i] >= thresholds[i], average='binary')  # Using thresholds
            row.append('{:.3f}: {:.3f}/{:.3f}/{:.3f}'.format(thresholds[i], p, r, f1))
            # print(i, thresholds[i], (Y_proba[:, i] >= thresholds[i]).astype(int))
        #end if

        # Results using optimal threshold
        if n_classes == 2 and i == 1:
            thresholds_best[i] = 1.0 - thresholds_best[0]
            p, r, f1, _ = precision_recall_fscore_support(Y_true[:, i], Y_proba[:, i] >= thresholds_best[i], average='binary')
            row.append('{:.3f}: {:.3f}/{:.3f}/{:.3f}'.format(thresholds_best[i], p, r, f1))
        else:
            p, r, t = precision_recall_curve(Y_true[:, i], Y_proba[:, i])
            f1 = np.nan_to_num((2 * p * r) / (p + r + 1e-8))
            best_f1_i = np.argmax(f1)
            thresholds_best[i] = t[best_f1_i]
            # thresholds_best[i] = 0.5 if np.isclose(t[best_f1_i], 0.0) else t[best_f1_i]
            row.append('{:.3f}: {:.3f}/{:.3f}/{:.3f}'.format(thresholds_best[i], p[best_f1_i], r[best_f1_i], f1[best_f1_i]))
            # print(i, thresholds_best[i], (Y_proba[:, i] >= thresholds_best[i]).astype(int))
        #end if

        # Results using optimal threshold for precision > precision_threshold
        if precision_thresholds is not None:
            if n_classes == 2 and i == 1:
                thresholds_minprec[i] = 1.0 - thresholds_minprec[0]
                p, r, f1, _ = precision_recall_fscore_support(Y_true[:, i], Y_proba[:, i] >= thresholds_minprec[i], average='binary')
                row.append('{:.3f}: {:.3f}/{:.3f}/{:.3f}'.format(thresholds_minprec[i], p, r, f1))

            else:
                try:
                    best_f1_i = max(filter(lambda k: p[k] >= precision_thresholds[i], range(p.shape[0])), key=lambda k: f1[k])
                    if best_f1_i == p.shape[0] - 1 or f1[best_f1_i] == 0.0: raise ValueError()

                    thresholds_minprec[i] = t[best_f1_i]
                    # thresholds_minprec[i] = 0.5 if np.isclose(t[best_f1_i], 0.0) else t[best_f1_i]
                    row.append('{:.3f}: {:.3f}/{:.3f}/{:.3f}'.format(thresholds_minprec[i], p[best_f1_i], r[best_f1_i], f1[best_f1_i]))
                    # print(i, precision_thresholds[i], (Y_proba[:, i] >= precision_thresholds[i]).astype(int))

                except ValueError:
                    best_f1_i = np.argmax(f1)
                    logger.warn('Unable to find threshold for label "{}" where precision >= {}.'.format(target_names[i], precision_thresholds[i], t[best_f1_i]))
                    row.append('-')
                #end try
        #end if

        table.append(row)
    #end for

    headers = ['Label', 'Support', 'AP', 'T=0.5']
    if thresholds is not None: headers.append('File T')
    headers.append('Best T')
    if precision_thresholds is not None: headers.append('Min Prec T')

    if n_classes > 1:
        macro_averages = ['Macro average', '-', '{:.3f}'.format(average_precision_score(Y_true, Y_proba, average='macro')), '{:.3f}/{:.3f}/{:.3f}'.format(*precision_recall_fscore_support(Y_true, Y_proba >= 0.5, average='macro'))]
        if thresholds is not None: macro_averages.append('{:.3f}/{:.3f}/{:.3f}'.format(*precision_recall_fscore_support(Y_true, Y_proba >= thresholds, average='macro')))
        macro_averages.append('{:.3f}/{:.3f}/{:.3f}'.format(*precision_recall_fscore_support(Y_true, Y_proba >= thresholds_best, average='macro')))
        if precision_thresholds is not None: macro_averages.append('{:.3f}/{:.3f}/{:.3f}'.format(*precision_recall_fscore_support(Y_true, Y_proba >= thresholds_minprec, average='macro')))

        micro_averages = ['Micro average', '-', '{:.3f}'.format(average_precision_score(Y_true, Y_proba, average='micro')), '{:.3f}/{:.3f}/{:.3f}'.format(*precision_recall_fscore_support(Y_true, Y_proba >= 0.5, average='micro'))]
        if thresholds is not None: micro_averages.append('{:.3f}/{:.3f}/{:.3f}'.format(*precision_recall_fscore_support(Y_true, Y_proba >= thresholds, average='micro')))
        micro_averages.append('{:.3f}/{:.3f}/{:.3f}'.format(*precision_recall_fscore_support(Y_true, Y_proba >= thresholds_best, average='micro')))
        if precision_thresholds is not None: micro_averages.append('{:.3f}/{:.3f}/{:.3f}'.format(*precision_recall_fscore_support(Y_true, Y_proba >= thresholds_minprec, average='micro')))

        perfect_set = ['Perfect set', str(Y_true.shape[0]), '-', '{:.3f}'.format(accuracy_score(Y_true, Y_proba >= 0.5))]
        if thresholds is not None: perfect_set.append('{:.3f}'.format(accuracy_score(Y_true, Y_proba >= thresholds)))
        perfect_set.append('{:.3f}'.format(accuracy_score(Y_true, Y_proba >= thresholds_best)))
        if precision_thresholds is not None: perfect_set.append('{:.3f}'.format(accuracy_score(Y_true, Y_proba >= thresholds_minprec)))

        table += [perfect_set, macro_averages, micro_averages]
        table.insert(-3, ['-' * max(len(row[i]) for row in table + [headers]) for i in range(len(macro_averages))])
    #end if

    return tabulate(table, headers=headers, tablefmt='psql') + '\n{} labels, {} instances, {} instance-labels'.format(n_classes, Y_true.shape[0], int(support_total))
#end def


def find_best_thresholds(Y_true, Y_proba, *, labels=None, target_names=None, precision_thresholds=None):
    Y_true, Y_proba = _make_label_indicator(Y_true, Y_proba)
    Y_true, Y_proba, target_names = _filter_labels(Y_true, Y_proba, labels, target_names)
    n_classes = Y_true.shape[1]

    if precision_thresholds is not None and isinstance(precision_thresholds, float): precision_thresholds = np.full(n_classes, precision_thresholds)

    assert Y_true.shape[0] == Y_proba.shape[0]
    assert Y_true.shape[1] == Y_proba.shape[1]
    assert len(target_names) == n_classes

    thresholds = np.zeros(n_classes)
    for i in range(n_classes):
        if n_classes == 2 and i == 1:
            thresholds[i] = 1.0 - thresholds[0]
            break
        #end if

        p, r, t = precision_recall_curve(Y_true[:, i], Y_proba[:, i])
        f1 = np.nan_to_num((2 * p * r) / (p + r + 1e-8))

        if precision_thresholds is None:  # use optimal threshold
            best_f1_i = np.argmax(f1)

        else:  # use optimal threshold for precision > precision_threshold
            try:
                best_f1_i = max(filter(lambda k: p[k] >= precision_thresholds[i], range(p.shape[0])), key=lambda k: f1[k])
                if best_f1_i == p.shape[0] - 1 or f1[best_f1_i] == 0.0: raise ValueError()

            except ValueError:
                best_f1_i = np.argmax(f1)
                logger.warn('Unable to find threshold for label "{}" where precision >= {}. Defaulting to best threshold of {}.'.format(target_names[i], precision_thresholds[i], t[best_f1_i]))
            #end try

        #end if

        thresholds[i] = t[best_f1_i]
    #end for

    return thresholds
#end def


def generate_pr_curves(Y_true, Y_proba, output_prefix, *, labels=None, target_names=None, thresholds=None, precision_thresholds=None):
    Y_true, Y_proba = _make_label_indicator(Y_true, Y_proba)
    Y_true, Y_proba, target_names = _filter_labels(Y_true, Y_proba, labels, target_names)
    n_classes = Y_true.shape[1]

    if isinstance(thresholds, float): thresholds = np.full(n_classes, thresholds)
    if thresholds is not None and n_classes == 2: thresholds[1] = 1.0 - thresholds[0]
    assert thresholds is None or ((thresholds <= 1).all() and (thresholds >= 0.0).all())

    if isinstance(precision_thresholds, float): precision_thresholds = np.full(n_classes, precision_thresholds)
    assert precision_thresholds is None or ((precision_thresholds <= 1).all() and (precision_thresholds >= 0.0).all())

    thresholds_best = np.zeros(n_classes)
    thresholds_minprec = np.zeros(n_classes)
    for i, name in enumerate(target_names):
        precision, recall, thresholds_ = precision_recall_curve(Y_true[:, i], Y_proba[:, i])
        f1 = np.nan_to_num((2 * precision * recall) / (precision + recall + 1e-8))
        ap_score = average_precision_score(Y_true[:, i], Y_proba[:, i])

        fig, ax = plt.subplots()
        ax.plot(recall, precision, label='Precision-Recall (AP={:.3f})'.format(ap_score))
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.set_ylim([0.0, 1.0])
        ax.set_xlim([0.0, 1.0])
        ax.set_title('Precision-Recall Curve for "{}"'.format(name))

        if thresholds is not None:  # Results using given thresholds
            p, r, f1_score, _ = precision_recall_fscore_support(Y_true[:, i], Y_proba[:, i] >= thresholds[i], average='binary')  # Using thresholds
            ax.plot([r], [p], marker='x', label='File T={:.3f}; F1={:.3f}'.format(thresholds[i], f1_score))
        #end if

        if n_classes == 2 and i == 1:  # Results using optimal threshold
            thresholds_best[i] = 1.0 - thresholds_best[0]
            p, r, f1_score, _ = precision_recall_fscore_support(Y_true[:, i], Y_proba[:, i] >= thresholds_best[i], average='binary')
        else:
            best_f1_i = np.argmax(f1)
            p, r, f1_score, thresholds_best[i] = precision[best_f1_i], recall[best_f1_i], f1[best_f1_i], thresholds_[best_f1_i]
        #end if
        ax.plot([r], [p], marker='x', label='Best T={:.3f}; F1={:.3f}'.format(thresholds_best[i], f1_score))

        if precision_thresholds is not None:  # Results using optimal threshold for precision > precision_threshold
            if n_classes == 2 and i == 1:
                thresholds_minprec[i] = 1.0 - thresholds_minprec[0]
                p, r, f1_score, _ = precision_recall_fscore_support(Y_true[:, i], Y_proba[:, i] >= thresholds_minprec[i], average='binary')

            else:
                try:
                    best_f1_i = max(filter(lambda k: precision[k] >= precision_thresholds[i], range(precision.shape[0])), key=lambda k: f1[k])
                    if best_f1_i == precision.shape[0] - 1 or f1[best_f1_i] == 0.0: raise ValueError()
                    thresholds_minprec[i] = thresholds_[best_f1_i]

                except ValueError:
                    best_f1_i = np.argmax(f1)
                    logger.warn('Unable to find threshold for label "{}" where precision >= {}.'.format(target_names[i], precision_thresholds[i], thresholds_[best_f1_i]))
                #end try
                p, r, f1_score, thresholds_minprec[i] = precision[best_f1_i], recall[best_f1_i], f1[best_f1_i], thresholds_[best_f1_i]
            #end if
            ax.plot([r], [p], marker='x', label='Minprec T={:.3f}; F1={:.3f}'.format(thresholds_minprec[i], f1_score))
        #end if

        ax.legend()
        plt.tight_layout()
        with uri_open(os.path.join(output_prefix, name + '.pdf'), 'wb') as f:
            fig.savefig(f, format='pdf')
            logger.info('Precision-Recall curve for "{}" saved to <{}>.'.format(name, f.name))
        #end with
    #end for
#end def


def _make_label_indicator(Y_true, Y_proba):
    if Y_true.ndim == 1 or Y_true.shape[1] == 1:  # have to be multiclass or binary
        Y_true_max = Y_true.max()
        assert Y_true_max == int(Y_true_max)

        if Y_true_max == 1:  # binary!
            temp = np.zeros((Y_true.shape[0], 2))
            temp[:, 0] = Y_true
            temp[:, 1] = 1.0 - Y_true
            Y_true = temp
        else:  # multiclass
            Y_true = label_binarize(Y_true, classes=range(int(Y_true_max) + 1))
        #end if

        if Y_proba.ndim == 1 or Y_proba.shape[1] == 1:
            assert Y_true.shape[1] == 2
            temp = np.zeros((Y_proba.shape[0], 2))
            temp[:, 0] = Y_proba if Y_proba.ndim == 1 else Y_proba[:, 0]
            temp[:, 1] = 1.0 - (Y_proba if Y_proba.ndim == 1 else Y_proba[:, 0])
            Y_proba = temp
        #end if
    #end if

    return Y_true, Y_proba
#end def


def _filter_labels(Y_true, Y_proba, labels, target_names):
    if target_names is None: target_names = list(range(Y_true.shape[1]))

    if labels:
        Y_true, Y_proba = Y_true[:, labels], Y_proba[:, labels]
        target_names = [target_names[j] for j in labels]
    #end if

    return Y_true, Y_proba, target_names
#end def


def main():
    Y_true = np.zeros((10, 3))
    Y_proba = np.random.rand(10, 3)

    Y_true[:4] = 1
    print(classification_report(Y_true, Y_proba, target_names=['ham', 'spam', 'bam'], precision_thresholds=0.75))
#end def


if __name__ == '__main__': main()
