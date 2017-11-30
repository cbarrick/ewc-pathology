def true_positives(y, h, target=1, **kwargs):
    return ((h == target) & (y == target)).sum()


def false_positives(y, h, target=1, **kwargs):
    return ((h == target) & (y != target)).sum()


def true_negatives(y, h, target=1, **kwargs):
    return ((h != target) & (y != target)).sum()


def false_negatives(y, h, target=1, **kwargs):
    return ((h != target) & (y == target)).sum()


def confusion(y, h, target=1, **kwargs):
    return (
        true_positives(y, h, target=target, **kwargs),
        false_positives(y, h, target=target, **kwargs),
        true_negatives(y, h, target=target, **kwargs),
        false_negatives(y, h, target=target, **kwargs),
    )


def precision(y, h, target=1, **kwargs):
    tp = true_positives(y, h, target=target, **kwargs)
    fp = false_positives(y, h, target=target, **kwargs)
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 0


def recall(y, h, target=1, **kwargs):
    tp = true_positives(y, h, target=target, **kwargs)
    fn = false_negatives(y, h, target=target, **kwargs)
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 0


def f_score(y, h, beta=1, target=1, **kwargs):
    tp = true_positives(y, h, target=target, **kwargs)
    fp = false_positives(y, h, target=target, **kwargs)
    fn = false_negatives(y, h, target=target, **kwargs)
    beta2 = beta ** 2
    tp2 = (1 + beta2) * tp
    fn2 = beta2 * fn
    try:
        return tp2 / (tp2 + fn2 + fp)
    except ZeroDivisionError:
        return 0