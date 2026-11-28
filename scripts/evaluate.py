"""Evaluate saved face scores/embeddings, class predictions or liveness outcomes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from facial_attendance.evaluation import metrics as m


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='task', required=True)
    for task in ('verification', 'classification', 'liveness'):
        q = sub.add_parser(task)
        q.add_argument('--output', type=Path, required=True, help='New output directory; never overwritten.')
        q.add_argument('--run-id', required=True)
        q.add_argument('--checkpoint-id', required=True, help='Exact model version/hash; use synthetic for fixtures.')
        if task == 'verification':
            q.add_argument('--trials', type=Path, required=True)
            source = q.add_mutually_exclusive_group(required=True)
            source.add_argument('--scores', type=Path)
            source.add_argument('--embeddings', type=Path)
            q.add_argument('--approach', choices=['classification', 'metric'], required=True)
            q.add_argument('--mode', choices=['fixture', 'validation', 'official'], required=True)
            q.add_argument('--select-threshold', action='store_true')
        else:
            q.add_argument('--predictions', type=Path, required=True)
            q.add_argument('--split', choices=['validation', 'test', 'webcam'], required=True)
            if task == 'classification':
                q.add_argument('--module', choices=['emotion', 'face'], required=True)
                q.add_argument('--label-map', type=Path, help='Face train-derived JSON object: class ID -> name.')
            else:
                q.add_argument('--policy-id', required=True, help='Identifier of the externally selected decision policy.')
    return p


def evaluate(args):
    inputs, copies, curves = {}, {}, {}
    if args.task == 'verification':
        trials = m.read_trials(args.trials, official=args.mode == 'official')
        inputs['trials'] = args.trials
        if args.embeddings:
            rows = m.score_embeddings(trials, args.embeddings)
            inputs['raw_embeddings'] = args.embeddings
        else:
            rows = m.pair_scores(trials, m.read_csv(args.scores, ['left','right','label','cosine','euclidean_distance']))
            inputs['scores'] = args.scores
        y = [r['label'] for r in rows]
        curves = {name: m.binary_metrics(y, [r[name] for r in rows]) for name in ('cosine','negative_euclidean')}
        result = {'approach': args.approach, 'mode': args.mode, 'metrics': curves,
                  'score_representation': 'cosine(raw embeddings), negative Euclidean(raw embeddings)',
                  'input_limitation': 'Saved embeddings/scores are supplied by the caller; image decoding, inference and checkpoint provenance are not independently verified.'}
        if args.select_threshold:
            result['selection'] = m.select_face_threshold(y, [r['cosine'] for r in rows], split=args.mode)
        copies['pair_scores.csv'] = rows
        for name, values in curves.items():
            copies[f'roc_{name}.csv'] = values['roc']
    elif args.task == 'classification':
        if args.split == 'webcam':
            raise ValueError('Classification benchmark inputs must use validation or test.')
        rows = m.read_csv(args.predictions, ['id','true','predicted'])
        ids = [r['id'] for r in rows]
        if any(not v for v in ids) or len(set(ids)) != len(ids):
            raise ValueError('Prediction IDs must be unique and nonempty.')
        if args.module == 'emotion':
            if args.label_map:
                raise ValueError('Emotion uses the fixed three-class mapping, not a custom map.')
            labels, names = [0,1,2], ['Negative','Neutral','Positive']
        else:
            if not args.label_map:
                raise ValueError('Face classification requires the train-derived --label-map.')
            mapping = json.loads(args.label_map.read_text(encoding='utf-8-sig'))
            if not isinstance(mapping, dict) or not mapping or not all(isinstance(v,str) and v.strip() for v in mapping.values()):
                raise ValueError('Label map must be a nonempty JSON object of class IDs to names.')
            labels = sorted(m.integer(k) for k in mapping)
            names = [mapping[str(k)] for k in labels]
            inputs['label_map'] = args.label_map
        result = {'module': args.module, 'split': args.split,
                  'metrics': m.classification_metrics([r['true'] for r in rows], [r['predicted'] for r in rows],labels,names)}
        copies['predictions.csv'] = rows
        inputs['predictions'] = args.predictions
    else:
        rows = m.read_csv(args.predictions, ['id','label','score','status','family'])
        result = {'split': args.split, 'policy_id': args.policy_id, 'metrics': m.liveness_metrics(rows),
                  'policy_note': 'Statuses come from the versioned adapter policy. This evaluator does not select a threshold or replace quality checks.'}
        inputs['predictions'] = args.predictions
        copies['predictions.csv'] = rows
        if result['metrics']['roc']:
            curves = {'p_spoof': result['metrics']['roc']}
            copies['roc_p_spoof.csv'] = curves['p_spoof']['roc']
    if not args.run_id.strip() or not args.checkpoint_id.strip():
        raise ValueError('Run and checkpoint identifiers must be nonempty.')
    import numpy, sklearn
    def git(*options):
        try:
            return subprocess.check_output(['git',*options],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):
            return None
    result.update(task=args.task, run_id=args.run_id, checkpoint_id=args.checkpoint_id,
                  created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  numpy=numpy.__version__, sklearn=sklearn.__version__,
                  git_commit=git('rev-parse','HEAD'), git_tracked_dirty=bool(git('diff','HEAD','--name-only')),
                  git_working_tree_dirty=bool(git('status','--porcelain')),
                  inputs={name:{'sha256':m.file_hash(path),'filename':path.name} for name,path in inputs.items()},
                  source_sha256={p.relative_to(ROOT).as_posix():m.file_hash(p) for p in [Path(__file__).resolve(),ROOT/'src/facial_attendance/evaluation/metrics.py']})
    # All input/metric validation occurs before creating any outputs.
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in copies.items():
        m.write_csv(args.output/name, rows)
    if curves:
        m.save_roc(args.output/'roc.png', curves)
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


def main():
    p = parser(); args = p.parse_args()
    try:
        result = evaluate(args)
    except (ValueError,KeyError,OSError,TypeError) as exc:
        p.error(str(exc))
    print(f"Saved {result['task']} evaluation to {args.output}")


if __name__ == '__main__':
    main()
