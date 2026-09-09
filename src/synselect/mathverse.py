"""Standalone, deadline-bounded MathVerse inference. Never launches training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlparse

from industrial_ft.utils.config import REPO_ROOT, load_yaml, resolve_model_name
from synselect.data import make_prompt, read_records
from synselect.run_budget import run_child
from synselect.utils.budget import WallBudget
from synselect.utils.config import local_path
from synselect.utils.mathverse import read_predictions, score, select_problems, summarize


def load_eval_config(name):
    if not re.fullmatch(r'[\w-]+', name):
        raise ValueError('Use an experiment name, not a path')
    directory = Path(os.environ.get('INDUSTRIAL_FT_EXPERIMENTS_DIR') or
                     REPO_ROOT / 'configs/experiments/synselect')
    cfg = load_yaml(directory / f'{name}.yaml')
    b = cfg['budget']
    if cfg['stage'] != 'eval' or 'train' in cfg:
        raise ValueError('This entry accepts evaluation-only configurations')
    if cfg['model_options']['attn_implementation'] != 'sdpa' or cfg['eval']['do_sample'] is not False:
        raise ValueError('Require SDPA and deterministic evaluation')
    if cfg['eval']['scoring'] not in ('local_diagnostic', 'deepseek_api'):
        raise ValueError('Unsupported scoring method')
    if cfg['eval']['scoring'] == 'deepseek_api':
        judge = cfg['judge']
        url = urlparse(judge['base_url'])
        if url.scheme != 'https' or not url.netloc or url.username or url.password or url.query:
            raise ValueError('Judge base_url must be HTTPS without credentials or query parameters')
        for key in ('concurrency', 'request_timeout_seconds', 'retry_backoff_seconds', 'max_tokens'):
            if judge[key] <= 0:
                raise ValueError(f'judge.{key} must be positive')
        if judge['max_retries'] < 0 or b.get('judge_seconds', 0) <= b['save_reserve_seconds']:
            raise ValueError('Invalid API retry/judge budget')
    if not cfg['models'] or any(not re.fullmatch(r'[\w-]+', k) for k in cfg['models']):
        raise ValueError('Provide models with safe output names')
    if set(cfg['models']) & {'prepare', '_judge'}:
        raise ValueError('Model names conflict with reserved worker names')
    if any(b[k] <= 0 for k in b) or b['save_reserve_seconds'] >= b['per_model_seconds']:
        raise ValueError('Invalid time budgets')
    if b['preparation_seconds'] + len(cfg['models']) * b['per_model_seconds'] + b.get('judge_seconds', 0) + b['cleanup_seconds'] > b['total_seconds']:
        raise ValueError('Stage allocations exceed overall budget')
    versions = cfg['dataset']['versions']
    if len(versions) != 5 or len(set(versions)) != 5 or 'Vision Only' not in versions:
        raise ValueError('Use all five visual MathVerse versions')
    if cfg['dataset']['prompt_field'] not in ('query_cot', 'query_wo'):
        raise ValueError('Use version-specific query, never question_for_eval as model input')
    if cfg['eval']['max_new_tokens'] <= 0 or cfg['dataset']['original_problems'] <= 0:
        raise ValueError('Sample and token limits must be positive')
    for spec in (cfg['eval'], cfg.get('unlimited_eval', cfg['eval'])):
        if spec['max_new_tokens'] <= 0 or spec['max_sequence_tokens'] <= spec['max_new_tokens']:
            raise ValueError('Sequence limit must exceed the positive generation token limit')
    cfg['model_name_or_path'] = resolve_model_name(cfg['model'])
    cfg['output_dir'] = str(local_path(cfg['output_dir']))
    return cfg


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(cfg, output):
    spec = cfg['dataset']
    source = local_path(spec['path'])
    selected, audit = select_problems(read_records(source), spec['original_problems'],
                                      spec['versions'], spec['seed'])
    for row in selected:
        if not isinstance(row.get(spec['prompt_field']), str) or not row[spec['prompt_field']].strip():
            raise ValueError(f'Missing version-specific query for {row["sample_index"]}')
        if not isinstance(row.get('image'), str) or not row['image']:
            raise ValueError('Use original MathVerse testmini.json and extracted image files')
        path = Path(row['image'])
        path = path if path.is_absolute() else local_path(spec['image_root']) / path
        if not path.is_file():
            raise FileNotFoundError(path)
        row['image'] = str(path)
        row['answer'] = str(row['answer'])
    audit['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    audit['source'] = str(source)
    write_json(output / 'selection.json', audit)
    write_json(output / 'samples.json', selected)
    write_json(output / 'resolved_config.json', cfg)


def infer(cfg, output, name, deadline):
    budget = WallBudget(deadline=deadline, reserve=cfg['budget']['save_reserve_seconds'])
    import torch
    from PIL import Image
    from transformers import StoppingCriteria, StoppingCriteriaList
    from synselect.runtime import build_model, build_processor

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Expose one bf16-capable CUDA GPU on the server')
    write_json(output / f'{name}.environment.json', {
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'mode': 'inference_only_bf16_sdpa'})
    processor = build_processor(cfg)
    adapter = cfg['models'][name]
    model = build_model(cfg, str(local_path(adapter)) if adapter else None, trainable=False).to('cuda').eval()
    class DeadlineStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.full((input_ids.shape[0],), budget.should_stop(), device=input_ids.device, dtype=torch.bool)

    rows = json.loads((output / 'samples.json').read_text(encoding='utf-8'))
    with (output / f'{name}.jsonl').open('x', encoding='utf-8') as handle, torch.inference_mode():
        for row in rows:
            if budget.should_stop():
                break
            prompt = make_prompt(row[cfg['dataset']['prompt_field']], cfg)
            text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            with Image.open(row['image']) as source:
                image = source.convert('RGB')
            inputs = processor(text=[text], images=[image], return_tensors='pt').to('cuda')
            if inputs.input_ids.shape[1] + cfg['eval']['max_new_tokens'] > cfg['eval']['max_sequence_tokens']:
                raise ValueError(f'Sequence too long: {row["sample_index"]}')
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=cfg['eval']['max_new_tokens'],
                                       use_cache=True, stopping_criteria=StoppingCriteriaList([DeadlineStop()]))
            if budget.should_stop():
                break  # never score a deadline-interrupted generation
            tokens = generated[0, inputs.input_ids.shape[1]:]
            prediction = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            extracted, correct = score(prediction, row['answer']) if cfg['eval']['scoring'] == 'local_diagnostic' else ('', None)
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            result = {**row, 'model_answer': prediction, 'extraction': extracted, 'correct': correct,
                      'category': row.get('category', row.get('metadata', {})),
                      'truncated': len(tokens) == cfg['eval']['max_new_tokens'] and tokens[-1].item() not in eos}
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()


def run(cfg, experiment, output):
    started = time.monotonic()
    unlimited = cfg.get('no_time_limit', False)
    end = started + cfg['budget']['total_seconds']
    work_end = end - cfg['budget']['cleanup_seconds']
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'resolved_config.json', cfg)
    statuses = {}
    try:
        if cfg['eval']['scoring'] == 'deepseek_api':
            from synselect.utils.mathverse_judge import require_key
            require_key(cfg['judge'])  # fail before spending GPU time if the key is missing
        for adapter in cfg['models'].values():
            if adapter and not (local_path(adapter) / 'adapter_config.json').is_file():
                raise FileNotFoundError(f'Missing existing adapter: {adapter}')
        command = [sys.executable, '-m', 'synselect.mathverse', '--experiment', experiment,
                   '--output', str(output)]
        run_child(command + ['--worker', 'prepare'],
                  None if unlimited else min(cfg['budget']['preparation_seconds'], work_end - time.monotonic()), output / 'prepare.log')
        for name in cfg['models']:
            seconds = min(cfg['budget']['per_model_seconds'], work_end - time.monotonic() - cfg['budget'].get('judge_seconds', 0))
            deadline_args = [] if unlimited else ['--deadline', str(time.time() + seconds)]
            begin = time.monotonic()
            try:
                run_child(command + ['--worker', name] + deadline_args, None if unlimited else seconds, output / f'{name}.log')
                statuses[name] = {'status': 'finished'}
            except TimeoutError as error:
                statuses[name] = {'status': 'timeout', 'error': str(error)}
            statuses[name]['seconds'] = time.monotonic() - begin
        if cfg['eval']['scoring'] == 'deepseek_api':
            seconds = min(cfg['budget']['judge_seconds'], work_end - time.monotonic())
            try:
                deadline_args = [] if unlimited else ['--deadline', str(time.time() + seconds)]
                run_child(command + ['--worker', '_judge'] + deadline_args,
                          None if unlimited else seconds, output / 'judge.log')
                statuses['judge'] = {'status': 'finished'}
            except TimeoutError as error:
                statuses['judge'] = {'status': 'timeout', 'error': str(error)}
    except Exception as error:
        statuses['error'] = str(error)
        raise
    finally:
        samples_path = output / 'samples.json'
        if samples_path.exists():
            rows = json.loads(samples_path.read_text(encoding='utf-8'))
            tables = {name: read_predictions(output / f'{name}.jsonl') for name in cfg['models']}
            if cfg['eval']['scoring'] == 'deepseek_api':
                from synselect.mathverse_score import scored_tables
                tables = scored_tables(cfg, output, output)
            report = summarize(tables, rows, cfg['dataset']['versions'], scoring_label(cfg))
            for name, table in tables.items():
                # Compatible model-output array for official extraction / judgment tools.
                write_json(output / f'{name}.official_input.json', list(table.values()))
        else:
            report = {'complete': False}
        report.update({'stages': statuses, 'elapsed_seconds': time.monotonic() - started,
                       'budget_seconds': None if unlimited else cfg['budget']['total_seconds'],
                       'no_time_limit': unlimited,
                       'max_new_tokens': cfg['eval']['max_new_tokens']})
        write_json(output / 'report.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


def scoring_label(cfg):
    return (f"{cfg['judge']['model']}_judge_not_official_MathVerse" if cfg['eval']['scoring'] == 'deepseek_api'
            else 'local_diagnostic_not_official_MathVerse')


def rescore(cfg, experiment, source, output):
    from synselect.mathverse_score import scored_tables
    from synselect.utils.mathverse_judge import require_key
    if cfg['eval']['scoring'] != 'deepseek_api':
        raise ValueError('--rescore requires deepseek_api scoring')
    require_key(cfg['judge'])
    rows = json.loads((source / 'samples.json').read_text(encoding='utf-8'))
    previous = json.loads((source / 'resolved_config.json').read_text(encoding='utf-8'))
    cfg['models'] = previous['models']
    cfg['dataset']['versions'] = previous['dataset']['versions']
    if not any((source / f'{name}.jsonl').is_file() for name in cfg['models']):
        raise FileNotFoundError('No saved prediction JSONL files in the source run')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'resolved_config.json', cfg)
    write_json(output / 'samples.json', rows)
    started = time.monotonic()
    status = 'finished'
    try:
        seconds = cfg['budget']['judge_seconds']
        unlimited = cfg.get('no_time_limit', False)
        deadline_args = [] if unlimited else ['--deadline', str(time.time() + seconds)]
        run_child([sys.executable, '-m', 'synselect.mathverse', '--experiment', experiment,
                   '--worker', '_judge', '--source', str(source), '--output', str(output)] + deadline_args,
                  None if unlimited else seconds, output / 'judge.log')
    except TimeoutError:
        status = 'timeout'
    except Exception:
        status = 'failed'
        raise
    finally:
        tables = scored_tables(cfg, source, output)
        report = summarize(tables, rows, cfg['dataset']['versions'], scoring_label(cfg))
        report.update({'source_run': str(source), 'status': status, 'inference': False,
                       'no_time_limit': cfg.get('no_time_limit', False),
                       'elapsed_seconds': time.monotonic() - started})
        write_json(output / 'report.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='mathverse_1h')
    parser.add_argument('--output', help='New run directory; existing results are never overwritten')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--no-time-limit', action='store_true',
                        help='Disable total/stage deadlines; use unlimited_eval token limits for new inference')
    parser.add_argument('--rescore', help='Existing run directory: score saved answers only, no GPU or dataset needed')
    parser.add_argument('--source', help=argparse.SUPPRESS)
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = load_eval_config(args.experiment)
    cfg['no_time_limit'] = args.no_time_limit
    if args.no_time_limit and not args.rescore:
        cfg['eval'].update(cfg.get('unlimited_eval', {}))
    output = local_path(args.output or (str(local_path(args.rescore) / 'deepseek_scores') if args.rescore else cfg['output_dir']))
    if args.dry_run:
        c = cfg['calibration']
        samples = cfg['dataset']['original_problems'] * len(cfg['dataset']['versions'])
        print(json.dumps({'training': False, 'dataset': 'MathVerse testmini (five visual versions)',
                          'models': cfg['models'], 'samples_per_model': samples,
                          'total_generations': 0 if args.rescore else samples * len(cfg['models']),
                          'linear_estimate_minutes': None if args.rescore or args.no_time_limit else
                          c['previous_total_seconds'] * samples * len(cfg['models']) /
                          (c['previous_samples_per_model'] * c['previous_model_count']) / 60,
                          'estimate_assumption': c['note'],
                          'budget': None if args.no_time_limit else cfg['budget'],
                          'no_time_limit': args.no_time_limit, 'max_new_tokens': cfg['eval']['max_new_tokens'],
                          'scoring': cfg['eval']['scoring'], 'judge': cfg.get('judge'),
                          'rescore_only': bool(args.rescore), 'output': str(output)}, indent=2))
    elif args.rescore:
        rescore(cfg, args.experiment, local_path(args.rescore), output)
    elif args.worker == 'prepare':
        cfg = json.loads((output / 'resolved_config.json').read_text(encoding='utf-8'))
        prepare(cfg, output)
    elif args.worker:
        # Freeze inference settings to the supervisor's audited configuration.
        cfg = json.loads((output / 'resolved_config.json').read_text(encoding='utf-8'))
        if args.worker == '_judge':
            from synselect.mathverse_score import judge_saved
            if args.deadline is None and not cfg.get('no_time_limit'):
                parser.error('Judge worker needs a deadline')
            judge_saved(cfg, local_path(args.source) if args.source else output, output, args.deadline)
            return
        if args.worker not in cfg['models'] or (args.deadline is None and not cfg.get('no_time_limit')):
            parser.error('Worker needs a configured model and deadline')
        infer(cfg, output, args.worker, args.deadline)
    else:
        run(cfg, args.experiment, output)


if __name__ == '__main__':
    main()
