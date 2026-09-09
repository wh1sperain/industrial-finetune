"""MathVerse grouped sampling and conservative, non-official local scoring."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re


def select_problems(records, count, versions, seed):
    groups = defaultdict(dict)
    ids = set()
    for raw in records:
        row = dict(raw)
        version = row['problem_version']
        if version not in versions:
            continue
        row['sample_index'] = str(row['sample_index'])
        row['problem_index'] = str(row['problem_index'])
        if row['sample_index'] in ids or version in groups[row['problem_index']]:
            raise ValueError('Duplicate sample ID or problem version')
        ids.add(row['sample_index'])
        groups[row['problem_index']][version] = row
    complete = {k: v for k, v in groups.items() if set(v) == set(versions)}
    if count <= 0 or count > len(complete):
        raise ValueError(f'Requested {count} original problems; only {len(complete)} complete groups')
    strata = defaultdict(list)
    for key, group in complete.items():
        row = group[versions[0]]
        category = row.get('metadata', row.get('category', {}))
        strata[(category.get('subject', 'unknown'), row['question_type'])].append(key)
    rank = lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()
    for pool in strata.values():
        pool.sort(key=rank)
    # Largest proportional deficit: representative quotas and interleaved prefix.
    picked, used = [], defaultdict(int)
    for step in range(count):
        available = [k for k, pool in strata.items() if used[k] < len(pool)]
        key = max(sorted(available), key=lambda k: (step + 1) * len(strata[k]) / len(complete) - used[k])
        picked.append(strata[key][used[key]])
        used[key] += 1
    selected = [complete[key][v] for key in picked for v in versions]
    audit = {'seed': seed, 'available_complete_problems': len(complete),
             'excluded_incomplete_problems': len(groups) - len(complete),
             'selected_problems': picked, 'selected_samples': len(selected),
             'strata': {str(k): used[k] for k in sorted(strata)}}
    return selected, audit


def extract_answer(text):
    tags = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, re.S)
    if len(tags) == 1:
        return tags[0].strip()
    if tags:
        return ''
    # Balanced braces handle nested LaTeX such as boxed fractions.
    start = text.rfind(r'\boxed{')
    if start >= 0:
        begin, depth = start + len(r'\boxed{'), 1
        for end in range(begin, len(text)):
            depth += (text[end] == '{') - (text[end] == '}')
            if depth == 0:
                return text[begin:end].strip()
    # Accept a bare short answer, never guess from a long reasoning trace.
    if '\n' not in text.strip() and len(text.strip()) <= 80 and '<think>' not in text:
        return text.strip()
    return ''


def normalize(value):
    return re.sub(r'\s+', '', str(value).strip().strip('$')).replace('−', '-')


def score(prediction, answer):
    extracted = extract_answer(prediction)
    return extracted, bool(extracted and normalize(extracted) == normalize(answer))


def summarize(tables, selected, versions, scoring='local_diagnostic_not_official_MathVerse'):
    expected = {str(r['sample_index']): r for r in selected}
    attempted = tables
    # API errors and unjudged generations must not be counted as wrong answers.
    tables = {name: {k: r for k, r in table.items() if type(r.get('correct')) is bool}
              for name, table in tables.items()}
    common = set(expected).intersection(*(set(t) for t in tables.values()))
    groups = defaultdict(set)
    for key, row in expected.items():
        groups[str(row['problem_index'])].add(key)
    paired = set().union(*(ids for ids in groups.values() if ids <= common))
    def metrics(table, ids):
        rows = [table[k] for k in ids]
        return {'samples': len(rows),
                ('local_answer_match' if scoring.startswith('local_') else 'judge_accuracy'):
                    sum(r['correct'] for r in rows) / len(rows) if rows else None,
                'extraction_rate': sum(bool(r['extraction']) for r in rows) / len(rows) if rows else None,
                'truncation_rate': sum(r['truncated'] for r in rows) / len(rows) if rows else None}
    return {'scoring': scoring,
            'unscored_by_model': {name: len(attempted[name]) - len(table) for name, table in tables.items()},
            'requested_samples_per_model': len(expected), 'paired_samples': len(paired),
            'paired_original_problems': len(paired) // len(versions),
            'complete': all(set(t) == set(expected) for t in tables.values()),
            'models': {name: {'all_completed': metrics(table, set(table)),
                              'paired_ALL': metrics(table, paired),
                              'paired_versions': {v: metrics(table, {k for k in paired if expected[k]['problem_version'] == v})
                                                  for v in versions}}
                       for name, table in tables.items()},
            'paired_ids': sorted(paired)}


def read_predictions(path):
    table = {}
    if path.exists():
        lines = path.read_text(encoding='utf-8').splitlines()
        for index, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:  # supervisor may interrupt the final write
                    break
                raise
            key = str(row['sample_index'])
            if key in table:
                raise ValueError('Duplicate predictions')
            table[key] = row
    return table
