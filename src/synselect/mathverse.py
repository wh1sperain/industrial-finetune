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
    if cfg['eval']['scoring'] != 'local_diagnostic':
        raise ValueError('Only local diagnostic scoring is implemented; export for official judging')
    if not cfg['models'] or any(not re.fullmatch(r'[\w-]+', k) for k in cfg['models']):
        raise ValueError('Provide models with safe output names')
    if any(b[k] <= 0 for k in b) or b['save_reserve_seconds'] >= b['per_model_seconds']:
        raise ValueError('Invalid time budgets')
    if b['preparation_seconds'] + len(cfg['models']) * b['per_model_seconds'] + b['cleanup_seconds'] > b['total_seconds']:
        raise ValueError('Stage allocations exceed overall budget')
    versions = cfg['dataset']['versions']
    if len(versions) != 5 or len(set(versions)) != 5 or 'Vision Only' not in versions:
        raise ValueError('Use all five visual MathVerse versions')
    if cfg['dataset']['prompt_field'] not in ('query_cot', 'query_wo'):
        raise ValueError('Use version-specific query, never question_for_eval as model input')
    if cfg['eval']['max_new_tokens'] <= 0 or cfg['dataset']['original_problems'] <= 0:
        raise ValueError('Sample and token limits must be positive')
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
            extracted, correct = score(prediction, row['answer'])
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            result = {**row, 'model_answer': prediction, 'extraction': extracted, 'correct': correct,
                      'category': row.get('category', row.get('metadata', {})),
                      'truncated': len(tokens) == cfg['eval']['max_new_tokens'] and tokens[-1].item() not in eos}
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()


def run(cfg, experiment, output):
    started = time.monotonic()
    end = started + cfg['budget']['total_seconds']
    work_end = end - cfg['budget']['cleanup_seconds']
    output.mkdir(parents=True, exist_ok=False)
    statuses = {}
    try:
        for adapter in cfg['models'].values():
            if adapter and not (local_path(adapter) / 'adapter_config.json').is_file():
                raise FileNotFoundError(f'Missing existing adapter: {adapter}')
        command = [sys.executable, '-m', 'synselect.mathverse', '--experiment', experiment,
                   '--output', str(output)]
        run_child(command + ['--worker', 'prepare'],
                  min(cfg['budget']['preparation_seconds'], work_end - time.monotonic()), output / 'prepare.log')
        for name in cfg['models']:
            seconds = min(cfg['budget']['per_model_seconds'], work_end - time.monotonic())
            deadline = time.time() + seconds
            begin = time.monotonic()
            try:
                run_child(command + ['--worker', name, '--deadline', str(deadline)], seconds, output / f'{name}.log')
                statuses[name] = {'status': 'finished'}
            except TimeoutError as error:
                statuses[name] = {'status': 'timeout', 'error': str(error)}
            statuses[name]['seconds'] = time.monotonic() - begin
    except Exception as error:
        statuses['error'] = str(error)
        raise
    finally:
        samples_path = output / 'samples.json'
        if samples_path.exists():
            rows = json.loads(samples_path.read_text(encoding='utf-8'))
            tables = {name: read_predictions(output / f'{name}.jsonl') for name in cfg['models']}
            report = summarize(tables, rows, cfg['dataset']['versions'])
            for name, table in tables.items():
                # Compatible model-output array for official extraction / judgment tools.
                write_json(output / f'{name}.official_input.json', list(table.values()))
        else:
            report = {'complete': False}
        report.update({'stages': statuses, 'elapsed_seconds': time.monotonic() - started,
                       'budget_seconds': cfg['budget']['total_seconds']})
        write_json(output / 'report.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='mathverse_1h')
    parser.add_argument('--output', help='New run directory; existing results are never overwritten')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--deadline', type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = load_eval_config(args.experiment)
    output = local_path(args.output or cfg['output_dir'])
    if args.dry_run:
        c = cfg['calibration']
        samples = cfg['dataset']['original_problems'] * len(cfg['dataset']['versions'])
        print(json.dumps({'training': False, 'dataset': 'MathVerse testmini (five visual versions)',
                          'models': cfg['models'], 'samples_per_model': samples,
                          'total_generations': samples * len(cfg['models']),
                          'linear_estimate_minutes': c['previous_total_seconds'] * samples * len(cfg['models']) /
                          (c['previous_samples_per_model'] * c['previous_model_count']) / 60,
                          'estimate_assumption': c['note'], 'budget': cfg['budget'],
                          'scoring': cfg['eval']['scoring'], 'output': str(output)}, indent=2))
    elif args.worker == 'prepare':
        prepare(cfg, output)
    elif args.worker:
        # Freeze inference settings to the supervisor's audited configuration.
        cfg = json.loads((output / 'resolved_config.json').read_text(encoding='utf-8'))
        if args.worker not in cfg['models'] or args.deadline is None:
            parser.error('Worker needs a configured model and deadline')
        infer(cfg, output, args.worker, args.deadline)
    else:
        run(cfg, args.experiment, output)


if __name__ == '__main__':
    main()
