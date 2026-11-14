"""Resolve a finite candidate set without treating similarity as ground truth."""

from array import array
from collections import Counter, defaultdict
from itertools import combinations, product


VIEWS = ('full_phash', 'crop_phash')
SPLIT_PRIORITY = {'test': 0, 'validation': 1, 'train': 2}


def candidate_pairs(db):
    """Expand a compact graph deterministically, one pair at a time."""
    for view in VIEWS:
        groups = defaultdict(list)
        for signature, path in db.execute(
                'SELECT signature,path FROM members WHERE view=? ORDER BY signature,path', (view,)):
            groups[signature].append(path)
        for paths in groups.values():
            for a, b in combinations(paths, 2):
                yield view, *sorted((a, b)), 0
        for a, b, distance in db.execute(
                'SELECT left_hash,right_hash,hamming FROM edges WHERE view=? ORDER BY left_hash,right_hash', (view,)):
            for left, right in product(groups[a], groups[b]):
                yield view, *sorted((left, right)), distance


def select_candidates(samples, fingerprints, db, review, rejected=()):
    """Return exclusions, retained paths and an independently checkable policy.

    Only confirmed duplicate edges form duplicate components. Unreviewed
    similarities are handled by direct pair constraints, never by transitive
    components. Exclusion of an ambiguous image is quarantine, not a verdict.
    """
    ordered = sorted(samples, key=lambda r: (SPLIT_PRIORITY[r.split], r.relative_path))
    by_path = {r.relative_path: r for r in ordered}
    rank = {r.relative_path: i for i, r in enumerate(ordered)}
    parent = {p: p for p in by_path}
    decisions = defaultdict(set)
    reviewer = review.get('reviewer', '').strip()
    if review.get('pairs') and not reviewer:
        raise ValueError('Near review needs an attributed reviewer')

    def representative(path):
        while parent[path] != path:
            parent[path] = parent[parent[path]]
            path = parent[path]
        return path

    for row in review.get('pairs', []):
        a, b = sorted((row['path_a'], row['path_b']))
        if (a == b or a not in by_path or b not in by_path or row.get('view') not in VIEWS
                or row.get('verdict') not in {'duplicate', 'distinct', 'related_source'}
                or not row.get('evidence', '').strip()):
            raise ValueError('Invalid or unbound natural-pair review')
        decisions[a, b].add(row['verdict'])
        if row['verdict'] == 'duplicate':
            parent[representative(a)] = representative(b)
    for (a, b), verdicts in decisions.items():
        if 'distinct' in verdicts and representative(a) == representative(b):
            raise ValueError('Contradictory duplicate/distinct review decisions')

    exclusions = {p: ('visual_review_rejection', '') for p in rejected}
    if not set(exclusions) <= set(by_path):
        raise ValueError('Visual rejection is absent from candidate selection')
    for path, item in fingerprints.items():
        if path in by_path and item['error']:
            exclusions[path] = ('invalid_face_input', '')
    components = defaultdict(list)
    for path in by_path:
        components[representative(path)].append(path)
    for members in components.values():
        if len(members) < 2:
            continue
        if len({(by_path[p].source_type, by_path[p].binary_label) for p in members}) > 1:
            exclusions.update((p, ('duplicate_label_conflict', '')) for p in members)
            continue
        eligible = [p for p in members if p not in exclusions]
        if eligible:
            keep = min(eligible, key=rank.__getitem__)
            exclusions.update((p, ('confirmed_near_duplicate', keep)) for p in eligible if p != keep)

    crop_groups = defaultdict(list)
    for path in by_path:
        if fingerprints[path]['crop_hash']:
            crop_groups[fingerprints[path]['crop_hash']].append(path)
    for members in crop_groups.values():
        if len({by_path[p].binary_label for p in members}) > 1:
            exclusions.update((p, ('crop_label_conflict', '')) for p in members)

    def disposition(view, a, b):
        verdicts = decisions.get((a, b), ())
        if 'distinct' in verdicts:
            return 'reviewed_distinct'
        if 'duplicate' in verdicts:
            return 'confirmed_duplicate'
        if 'related_source' in verdicts and by_path[a].split != by_path[b].split:
            return 'reviewed_cross_split_relation'
        if view == 'crop_phash':
            return 'crop_similarity_diagnostic'
        if by_path[a].split != by_path[b].split:
            return 'cross_split_constraint'
        if 'related_source' in verdicts:
            return 'reviewed_same_split_presentation'
        if by_path[a].source_type != by_path[b].source_type:
            return 'same_split_annotation_contrast'
        return 'full_image_constraint'

    actionable = {'cross_split_constraint', 'confirmed_duplicate', 'full_image_constraint', 'reviewed_cross_split_relation'}
    prior = {}
    counts = Counter()
    by_view = Counter()
    for view, a, b, _ in candidate_pairs(db):
        if a not in rank or b not in rank:
            raise ValueError('Candidate graph refers to a different selection')
        kind = disposition(view, a, b)
        counts[kind] += 1
        by_view[view] += 1
        if kind in actionable:
            high, low = sorted((rank[a], rank[b]))
            prior.setdefault(low, array('I')).append(high)
    related_cross_split = [(a, b) for (a, b), verdicts in decisions.items()
                           if 'related_source' in verdicts and 'distinct' not in verdicts
                           and by_path[a].split != by_path[b].split]
    for a, b in related_cross_split:
        high, low = sorted((rank[a], rank[b]))
        prior.setdefault(low, array('I')).append(high)

    kept, kept_indices, crop_keeper = set(), set(), {}
    for i, row in enumerate(ordered):
        path = row.relative_path
        if path in exclusions:
            continue
        crop = fingerprints[path]['crop_hash']
        if crop and crop in crop_keeper:
            exclusions[path] = ('exact_crop_duplicate', crop_keeper[crop])
            continue
        witness = next((j for j in prior.get(i, ()) if j in kept_indices), None)
        if witness is not None:
            exclusions[path] = ('near_candidate_quarantine', ordered[witness].relative_path)
            continue
        kept.add(path)
        kept_indices.add(i)
        if crop:
            crop_keeper[crop] = path

    # Revisit every direct relation; do not equate a successful greedy loop
    # with observed clearance of the retained candidate set.
    remaining = 0
    for view, a, b, _ in candidate_pairs(db):
        remaining += a in kept and b in kept and disposition(view, a, b) in actionable
    remaining += sum(a in kept and b in kept for a, b in related_cross_split)
    policy = {
        'name': 'direct-pair-quarantine-v2', 'search_complete': True,
        'priority': 'test, validation, train; lexical path within each split',
        'candidate_occurrences_by_view': dict(by_view), 'dispositions': dict(counts),
        'remaining_actionable_pairs': remaining,
        'reviewed_cross_split_relations': len(related_cross_split),
        'exclusion_counts': dict(Counter(reason for reason, _ in exclusions.values())),
        'scope': 'Full-image radius-6 candidates across splits and within the same split/source type, with explicit distinct/presentation exemptions; confirmed duplicates and reviewed cross-split source relations; exact face-crop redundancy/conflicts.',
        'limits': 'Unreviewed exclusions are conservative quarantine, not confirmed duplicates. Crop-only pHash similarity is diagnostic regardless of split. No media/session IDs are inferred. This does not establish recall outside radius 6 or independence of unknown source captures.',
    }
    return exclusions, kept, policy
