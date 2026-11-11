"""Metrics with explicit labels, score direction and failure accounting."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, auc, confusion_matrix, precision_recall_fscore_support, roc_curve


def read_csv(path, required):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError('Missing or duplicate CSV header.')
        if not set(required).issubset(reader.fieldnames):
            raise ValueError(f'Required columns: {required}')
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('Empty CSV or malformed row.')
    return rows


def integer(value):
    if isinstance(value, bool) or str(value) != str(int(value)):
        raise ValueError(f'Expected an integer class ID, got {value!r}.')
    return int(value)


def labels_array(values, allowed):
    values = [integer(v) for v in values]
    if not values or any(v not in allowed for v in values):
        raise ValueError('Empty or unknown class labels.')
    return np.asarray(values, dtype=np.int64)


def finite(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError('Scores must be a nonempty finite vector.')
    return result


def binary_metrics(labels, scores):
    y = labels_array(labels, {0, 1})
    s = finite(scores)
    if len(y) != len(s) or set(y) != {0, 1}:
        raise ValueError('ROC requires aligned scores and both binary classes.')
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    return {'auc': float(auc(fpr, tpr)), 'count': len(y),
            'negative_count': int(sum(y == 0)), 'positive_count': int(sum(y == 1)),
            'roc': [{'fpr': float(f), 'tpr': float(t),
                     'threshold': float(c) if np.isfinite(c) else None}
                    for f, t, c in zip(fpr, tpr, thresholds)]}


def select_face_threshold(labels, scores, *, split, max_far=0.01):
    if split != 'validation':
        raise ValueError('Threshold selection is allowed only on validation pairs.')
    if not 0 <= max_far <= 1:
        raise ValueError('max_far must be within [0,1].')
    binary_metrics(labels, scores)
    y, s = labels_array(labels, {0, 1}), finite(scores)
    for threshold in np.unique(s):
        far = float(np.mean(s[y == 0] >= threshold))
        if far <= max_far:
            return {'threshold': float(threshold), 'far': far,
                    'tar': float(np.mean(s[y == 1] >= threshold)),
                    'source': 'validation', 'reject_all': False}
    threshold = float(np.nextafter(s.max(), np.inf))
    if not np.isfinite(threshold):
        raise ValueError('Cannot represent a finite reject-all threshold.')
    return {'threshold': threshold, 'far': 0.0, 'tar': 0.0,
            'source': 'validation', 'reject_all': True}


def read_trials(path, *, official=False):
    trials = []
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        parts = line.split()
        if len(parts) != 3:
            raise ValueError('Every trial must contain left right label; blank rows are invalid.')
        left, right, label = parts
        label = integer(label)
        if label not in (0, 1):
            raise ValueError('Trial labels must be 0 or 1.')
        trials.append((left, right, label))
    if not trials or (official and len(trials) != 8805):
        raise ValueError('Official evaluation requires exactly 8,805 trials; fixtures must be nonempty.')
    return trials


def pair_scores(trials, rows):
    if len(rows) != len(trials):
        raise ValueError('Scores and trial counts differ.')
    result = []
    for (left, right, label), row in zip(trials, rows):
        if (row['left'], row['right'], integer(row['label'])) != (left, right, label):
            raise ValueError('Score paths, labels or row order differ from the trial file.')
        cosine, distance = float(row['cosine']), float(row['euclidean_distance'])
        if not np.isfinite([cosine, distance]).all() or not -1 <= cosine <= 1 or distance < 0:
            raise ValueError('Invalid cosine similarity or Euclidean distance.')
        result.append({'left': left, 'right': right, 'label': label,
                       'cosine': cosine, 'euclidean_distance': distance,
                       'negative_euclidean': -distance})
    return result


def score_embeddings(trials, archive):
    # NPZ contract: paths[N], raw_embeddings[N,D]. Never allow pickle loading.
    with np.load(archive, allow_pickle=False) as data:
        paths, vectors = data['paths'], np.asarray(data['raw_embeddings'], dtype=np.float64)
    if paths.ndim != 1 or paths.dtype.kind not in {'U', 'S'}:
        raise ValueError('Embedding paths must be a one-dimensional string array.')
    names = [p.decode('utf-8') if isinstance(p, bytes) else str(p) for p in paths]
    if len(set(names)) != len(names) or any(not p for p in names):
        raise ValueError('Embedding paths must be unique and nonempty.')
    if vectors.ndim != 2 or vectors.shape[0] != len(names) or vectors.shape[1] < 1 or not np.isfinite(vectors).all():
        raise ValueError('Expected finite raw_embeddings[N,D] aligned with paths.')
    lookup = dict(zip(names, vectors))
    rows = []
    for left, right, label in trials:
        if left not in lookup or right not in lookup:
            raise ValueError('A trial image has no saved embedding.')
        a, b = lookup[left], lookup[right]
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if not np.isfinite([norm_a, norm_b]).all() or min(norm_a, norm_b) <= 0:
            raise ValueError('Cosine requires finite nonzero embedding norms.')
        cosine = float(np.clip(np.dot(a / norm_a, b / norm_b), -1, 1))
        distance = float(np.linalg.norm(a - b))
        rows.append(dict(left=left, right=right, label=label, cosine=cosine, euclidean_distance=distance))
    return pair_scores(trials, rows)


def classification_metrics(truth, predicted, labels, names):
    if len(labels) != len(set(labels)) or len(names) != len(labels) or not labels:
        raise ValueError('Class IDs must be unique and aligned with names.')
    y, p = labels_array(truth, set(labels)), labels_array(predicted, set(labels))
    if len(y) != len(p):
        raise ValueError('Prediction and label counts differ.')
    precision, recall, f1, support = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {'count': len(y), 'correct': int(sum(y == p)), 'accuracy': float(accuracy_score(y, p)),
            'macro_f1': float(np.mean(f1)), 'label_order': list(labels),
            'confusion_matrix': confusion_matrix(y, p, labels=labels).tolist(),
            'confusion_axes': 'rows=true, columns=predicted',
            'per_class': [{'id': label, 'name': name, 'precision': float(pr), 'recall': float(re),
                           'f1': float(f), 'support': int(n)}
                          for label, name, pr, re, f, n in zip(labels, names, precision, recall, f1, support)]}


def liveness_metrics(rows):
    statuses = ['live', 'spoof', 'uncertain', 'failure']
    table = [[0] * 4 for _ in range(2)]
    scored_labels, scores, family_counts = [], [], {}
    seen = set()
    for row in rows:
        if not row['id'] or row['id'] in seen:
            raise ValueError('Liveness IDs must be nonempty and unique.')
        seen.add(row['id'])
        label, status, family = integer(row['label']), row['status'], row['family']
        if label not in (0, 1) or status not in statuses:
            raise ValueError('Invalid liveness label or status.')
        if (label == 0 and family != 'live') or (label == 1 and family not in {'print', 'screen', 'screen_photo', 'video_replay'}):
            raise ValueError('Attack family and binary label disagree.')
        score = row['score']
        if score != '':
            score = float(score)
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('p_spoof must be finite within [0,1].')
            scored_labels.append(label); scores.append(score)
        elif status != 'failure':
            raise ValueError('Only failed attempts may lack an inference score.')
        table[label][statuses.index(status)] += 1
        counts = family_counts.setdefault(family, dict.fromkeys(statuses, 0))
        counts[status] += 1
    if not rows:
        raise ValueError('Liveness attempts must be nonempty.')
    live_n, spoof_n = sum(table[0]), sum(table[1])
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    for family, counts in family_counts.items():
        counts['total'] = sum(counts.values())
        if family != 'live':
            counts['spoof_acceptance_rate'] = ratio(counts['live'], counts['total'])
    roc = binary_metrics(scored_labels, scores) if set(scored_labels) == {0, 1} else None
    return {'count': len(rows), 'scored_count': len(scores), 'unscored_count': len(rows)-len(scores),
            'roc': roc, 'roc_unavailable_reason': None if roc else 'Both scored classes are required.',
            'truth_order': ['live', 'spoof'], 'status_order': statuses, 'outcome_table': table,
            'live_count': live_n, 'spoof_count': spoof_n,
            'spoof_acceptance_rate': ratio(table[1][0], spoof_n),
            'live_rejection_rate': ratio(table[0][1], live_n),
            'genuine_user_blocking_rate': ratio(live_n-table[0][0], live_n),
            'uncertain_count': table[0][2]+table[1][2],
            'uncertain_rate': ratio(table[0][2]+table[1][2],len(rows)),
            'failure_count': table[0][3]+table[1][3],
            'failure_rate': ratio(table[0][3]+table[1][3],len(rows)), 'by_family': family_counts}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save_roc(path, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for name, metrics in curves.items():
        points = metrics['roc']
        ax.plot([p['fpr'] for p in points], [p['tpr'] for p in points], label=f"{name}: AUC={metrics['auc']:.4f}")
    ax.plot([0,1], [0,1], '--', color='gray')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', xlim=(0,1), ylim=(0,1.02))
    ax.legend(loc='lower right'); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
