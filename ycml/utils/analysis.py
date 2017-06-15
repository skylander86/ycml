import logging

import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support

from tabulate import tabulate

__all__ = ['classification_report', 'find_best_thresholds']

logger = logging.getLogger(__name__)


def classification_report(Y_true, Y_proba, *, labels=None, target_names=None, thresholds=None, precision_thresholds=None):
    if Y_proba.ndim == 1 or Y_proba.shape[1] == 1:
        temp = np.zeros((Y_proba.shape[0], 2))
        temp[:, 0] = Y_proba
        temp[:, 1] = 1.0 - Y_proba
        Y_proba = temp
        assert not (target_names and len(target_names) != 2)
    #end if

    if Y_true.ndim == 1 or Y_true.shape[1] == 1:
        temp = np.zeros((Y_true.shape[0], 2))
        temp[:, 0] = Y_true
        temp[:, 1] = 1 - Y_true
        Y_true = temp
        assert not (target_names and len(target_names) != 2)
    #end if

    n_classes = Y_true.shape[1]
    if target_names is None: target_names = list(range(n_classes))

    if labels:  # delete columns that we don't need
        Y_true, Y_proba = Y_true[:, labels], Y_proba[:, labels]
        target_names = [target_names[j] for j in labels]
        n_classes = Y_true.shape[1]
    #end if

    if isinstance(thresholds, float): thresholds = np.full((1, n_classes), thresholds)
    if thresholds is not None and thresholds.ndim == 1: thresholds = thresholds.reshape(1, n_classes)

    if isinstance(precision_thresholds, float): precision_thresholds = np.full((1, n_classes), precision_thresholds)
    if precision_thresholds is not None and precision_thresholds.ndim == 1: precision_thresholds = precision_thresholds.reshape(1, n_classes)

    assert Y_true.shape[0] == Y_proba.shape[0]
    assert Y_true.shape[1] == Y_proba.shape[1]
    assert len(target_names) == n_classes

    table = []
    support_total, ap_score_total = 0.0, 0.0
    thresholds_best = np.zeros((1, n_classes))
    thresholds_minprec = np.zeros((1, n_classes))
    for i, name in enumerate(target_names):
        # Results using 0.5 as threshold
        p, r, f1, _ = precision_recall_fscore_support(Y_true[:, i], Y_proba[:, i] >= 0.5, average='binary')  # Using thresholds
        support = Y_true[:, i].sum()
        ap_score = average_precision_score(Y_true[:, i], Y_proba[:, i])
        row = [name, '{:d}'.format(int(support)), '{:.3f}'.format(ap_score), '{:.3f}/{:.3f}/{:.3f}'.format(p, r, f1)]
        support_total += support
        ap_score_total += ap_score

        # print(i, 0.5, (Y_proba[:, i] >= 0.5).astype(int))

        # Results using given thresholds
        if thresholds is not None:
            p, r, f1, _ = precision_recall_fscore_support(Y_true[:, i], Y_proba[:, i] >= thresholds[0, i], average='binary')  # Using thresholds
            row.append('{:.3f}: {:.3f}/{:.3f}/{:.3f}'.format(thresholds[0, i], p, r, f1))
            # print(i, thresholds[0, i], (Y_proba[:, i] >= thresholds[0, i]).astype(int))
        #end if

        # Results using optimal threshold
        p, r, t = precision_recall_curve(Y_true[:, i], Y_proba[:, i])
        f1 = np.nan_to_num((2 * p * r) / (p + r + 1e-8))
        best_f1_i = np.argmax(f1)
        thresholds_best[0, i] = t[best_f1_i]
        # thresholds_best[0, i] = 0.5 if np.isclose(t[best_f1_i], 0.0) else t[best_f1_i]
        row.append('{:.3f}: {:.3f}/{:.3f}/{:.3f}'.format(thresholds_best[0, i], p[best_f1_i], r[best_f1_i], f1[best_f1_i]))
        # print(i, thresholds_best[0, i], (Y_proba[:, i] >= thresholds_best[0, i]).astype(int))

        # Results using optimal threshold for precision > precision_threshold
        if precision_thresholds is not None:
            try:
                best_f1_i = max(filter(lambda k: p[k] >= precision_thresholds[0, i], range(p.shape[0])), key=lambda k: f1[k])
                if best_f1_i == p.shape[0] - 1 or f1[best_f1_i] == 0.0: raise ValueError()

                thresholds_minprec[0, i] = t[best_f1_i]
                # thresholds_minprec[0, i] = 0.5 if np.isclose(t[best_f1_i], 0.0) else t[best_f1_i]
                row.append('{:.3f}: {:.3f}/{:.3f}/{:.3f}'.format(thresholds_minprec[0, i], p[best_f1_i], r[best_f1_i], f1[best_f1_i]))
                # print(i, precision_thresholds[0, i], (Y_proba[:, i] >= precision_thresholds[0, i]).astype(int))

            except ValueError:
                best_f1_i = np.argmax(f1)
                logger.warn('Unable to find threshold for label "{}" where precision >= {}.'.format(target_names[i], precision_thresholds[0, i], t[best_f1_i]))
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

        table += [macro_averages, micro_averages]
        table.insert(-2, ['-' * max(len(row[i]) for row in table + [headers]) for i in range(len(macro_averages))])
    #end if

    return tabulate(table, headers=headers, tablefmt='psql') + '\n{} labels, {} instances, {} instance-labels'.format(n_classes, Y_true.shape[0], int(support_total))
#end def


def find_best_thresholds(Y_true, Y_proba, *, precision_thresholds=None, target_names=None):
    if Y_proba.ndim == 1 or Y_proba.shape[1] == 1:
        temp = np.zeros((Y_proba.shape[0], 2))
        temp[:, 0] = Y_proba
        temp[:, 1] = 1.0 - Y_proba
        Y_proba = temp
        assert not (target_names and len(target_names) != 2)
    #end if

    if Y_true.ndim == 1 or Y_true.shape[1] == 1:
        temp = np.zeros((Y_true.shape[0], 2))
        temp[:, 0] = Y_true
        temp[:, 1] = 1 - Y_true
        Y_true = temp
        assert not (target_names and len(target_names) != 2)
    #end if

    n_classes = Y_true.shape[1]
    if target_names is None: target_names = list(range(n_classes))

    if precision_thresholds is not None:
        if isinstance(precision_thresholds, float): precision_thresholds = np.full((1, n_classes), precision_thresholds)
        elif precision_thresholds.ndim == 1: precision_thresholds = precision_thresholds.T
    #end if

    assert Y_true.shape[0] == Y_proba.shape[0]
    assert Y_true.shape[1] == Y_proba.shape[1]

    thresholds = np.zeros(n_classes)
    for i in range(n_classes):
        # Results using optimal threshold
        p, r, t = precision_recall_curve(Y_true[:, i], Y_proba[:, i])
        f1 = np.nan_to_num((2 * p * r) / (p + r + 1e-8))
        best_f1_i = np.argmax(f1)
        thresholds[i] = t[best_f1_i]

        # Results using optimal threshold for precision > precision_threshold
        if precision_thresholds is not None:
            try:
                best_f1_i = max(filter(lambda k: p[k] >= precision_thresholds[0, i], range(p.shape[0])), key=lambda k: f1[k])
                if best_f1_i == p.shape[0] - 1 or f1[best_f1_i] == 0.0: raise ValueError()

            except ValueError:
                best_f1_i = np.argmax(f1)
                logger.warn('Unable to find threshold for label "{}" where precision >= {}. Defaulting to best threshold of {}.'.format(target_names[i], precision_thresholds[0, i], t[best_f1_i]))
            #end try

            thresholds[i] = t[best_f1_i]
        #end if
    #end for

    return thresholds
#end def


def main():
    Y_true = np.zeros((10, 3))
    Y_proba = np.random.rand(10, 3)

    Y_true[:4] = 1
    print(classification_report(Y_true, Y_proba, target_names=['ham', 'spam', 'bam'], precision_thresholds=0.75))
#end def


if __name__ == '__main__': main()
