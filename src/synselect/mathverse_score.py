"""Concurrent scoring of saved MathVerse answers; no model loading or inference."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import time

from synselect.utils.mathverse import read_predictions
from synselect.utils.mathverse_judge import DeepSeekJudge, JudgeError


def judge_saved(cfg, source, output, deadline):
    judge = DeepSeekJudge(cfg['judge'], None if deadline is None else deadline - cfg['budget']['save_reserve_seconds'])
    selected = json.loads((source / 'samples.json').read_text(encoding='utf-8'))
    tables = {name: read_predictions(source / f'{name}.jsonl') for name in cfg['models']}
    # Interleave models on the same question to retain paired coverage on timeout.
    jobs = iter((name, tables[name][str(row['sample_index'])])
                for row in selected for name in tables if str(row['sample_index']) in tables[name])
    handles = {}
    try:
        for name in tables:
            handles[name] = (output / f'{name}.judged.jsonl').open('x', encoding='utf-8')
        with ThreadPoolExecutor(max_workers=cfg['judge']['concurrency']) as pool:
            pending = {}
            exhausted = False
            while pending or not exhausted:
                while (not exhausted and len(pending) < cfg['judge']['concurrency']
                       and time.time() < judge.deadline and not judge.fatal.is_set()):
                    job = next(jobs, None)
                    if job is None:
                        exhausted = True
                        break
                    name, row = job
                    pending[pool.submit(judge.score, row)] = (name, row)
                if not pending:
                    break
                done, _ = wait(pending, timeout=None if deadline is None else max(0, judge.deadline - time.time()), return_when=FIRST_COMPLETED)
                if not done:
                    break  # supervisor bounds any remaining blocked network threads
                for future in done:
                    name, row = pending.pop(future)
                    try:
                        result = future.result()
                    except JudgeError as error:
                        result = {**row, 'correct': None, 'extraction': '', 'judge_status': 'error',
                                  'judge_error': str(error), 'judge_model': cfg['judge']['model']}
                    handles[name].write(json.dumps(result, ensure_ascii=False) + '\n')
                    handles[name].flush()
    finally:
        for handle in handles.values():
            handle.close()


def scored_tables(cfg, source, output):
    tables = {}
    for name in cfg['models']:
        raw = read_predictions(source / f'{name}.jsonl')
        judged = read_predictions(output / f'{name}.judged.jsonl')
        # Never silently fall back to the old local score when an API request fails.
        tables[name] = {key: judged.get(key, {**row, 'correct': None, 'extraction': '',
                                             'judge_status': 'pending'}) for key, row in raw.items()}
    return tables
