import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from synselect.mathverse import load_eval_config, prepare, run
from synselect.utils.mathverse import extract_answer, read_predictions, score, select_problems, summarize


VERSIONS = ['Text Dominant', 'Text Lite', 'Vision Intensive', 'Vision Dominant', 'Vision Only']


def fixture():
    return [{'sample_index': f'{i}-{v}', 'problem_index': str(i), 'problem_version': v,
             'question_type': 'multi-choice', 'metadata': {'subject': 'Geometry' if i < 8 else 'Algebra'},
             'query_cot': 'Read the image.', 'question': '', 'question_for_eval': 'SECRET',
             'image': 'image.png', 'answer': 'A'} for i in range(10) for v in VERSIONS]


class MathVerseTests(unittest.TestCase):
    def test_group_sampling_stable_and_proportional(self):
        rows, audit = select_problems(fixture(), 5, VERSIONS, 42)
        reverse, _ = select_problems(list(reversed(fixture())), 5, VERSIONS, 42)
        self.assertEqual(rows, reverse)
        self.assertEqual(len(rows), 25)
        self.assertEqual(len(audit['selected_problems']), 5)
        self.assertEqual(sum(r['metadata']['subject'] == 'Algebra' for r in rows), 5)

    def test_incomplete_and_duplicate(self):
        rows, audit = select_problems(fixture()[:-1], 5, VERSIONS, 42)
        self.assertEqual(audit['excluded_incomplete_problems'], 1)
        with self.assertRaises(ValueError):
            select_problems(fixture() + [fixture()[0]], 5, VERSIONS, 42)

    def test_conservative_scoring(self):
        self.assertTrue(score('<think>reason</think><answer>A</answer>', 'A')[1])
        self.assertEqual(extract_answer(r'reason \boxed{\frac{1}{2}}'), r'\frac{1}{2}')
        self.assertFalse(score('Reasoning says A\nBut unsure', 'A')[1])
        self.assertFalse(score('<answer>A</answer><answer>B</answer>', 'A')[1])
        self.assertFalse(score('a', 'A')[1])  # no unsafe symbolic case folding

    def test_paired_only_complete_groups(self):
        rows = fixture()[:10]
        full = {r['sample_index']: dict(r, correct=True, extraction='A', truncated=False) for r in rows}
        partial = dict(list(full.items())[:7])
        result = summarize({'baseline': full, 'sft': partial, 'grpo': full}, rows, VERSIONS)
        self.assertEqual(result['paired_samples'], 5)
        self.assertEqual(result['paired_original_problems'], 1)
        self.assertFalse(result['complete'])
        self.assertEqual(result['models']['sft']['paired_versions']['Vision Only']['samples'], 1)
        self.assertEqual(summarize({'base': {}}, rows, VERSIONS)['paired_samples'], 0)

    def test_prepare_preserves_vision_only_prompt(self):
        cfg = load_eval_config('mathverse_1h')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / 'image.png').write_bytes(b'fixture')
            (path / 'testmini.json').write_text(json.dumps(fixture()), encoding='utf-8')
            cfg['dataset'].update(path=str(path / 'testmini.json'), image_root=str(path), original_problems=2)
            prepare(cfg, path)
            selected = json.loads((path / 'samples.json').read_text(encoding='utf-8'))
            self.assertEqual(len(selected), 10)
            self.assertTrue(all(r['query_cot'] == 'Read the image.' for r in selected))

    def test_supervisor_only_eval_workers_and_recovers_timeout(self):
        cfg = load_eval_config('mathverse_1h')
        cfg['models'] = {'baseline': None, 'sft': None, 'grpo': None}
        calls = []
        def fake_child(command, timeout, log):
            calls.append(command)
            output = Path(command[command.index('--output') + 1])
            worker = command[command.index('--worker') + 1]
            if worker == 'prepare':
                (output / 'samples.json').write_text(json.dumps(fixture()[:5]), encoding='utf-8')
            elif worker == 'sft':
                raise TimeoutError('fixture deadline')
        with tempfile.TemporaryDirectory() as tmp, patch('synselect.mathverse.run_child', fake_child), patch('builtins.print'):
            output = Path(tmp) / 'run'
            run(cfg, 'mathverse_1h', output)
            report = json.loads((output / 'report.json').read_text())
            self.assertEqual(report['stages']['sft']['status'], 'timeout')
            self.assertEqual(len(calls), 4)
            self.assertTrue(all(c[2] == 'synselect.mathverse' for c in calls))
            self.assertFalse(report['complete'])

    def test_incomplete_jsonl_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions.jsonl'
            path.write_text('{"sample_index":"1"}\n{"sam', encoding='utf-8')
            self.assertEqual(set(read_predictions(path)), {'1'})


if __name__ == '__main__':
    unittest.main()
