import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from synselect.mathverse import load_eval_config, rescore
from synselect.mathverse_score import judge_saved, scored_tables
from synselect.utils.mathverse import summarize
from synselect.utils.mathverse_judge import DeepSeekJudge, JudgeError


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.cfg = load_eval_config('mathverse_1h')
        self.cfg['judge']['cache_dir'] = str(self.path / 'cache')
        self.cfg['judge']['max_retries'] = 0
        self.env = patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'secret-test-value'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def response(self, value):
        return io.BytesIO(json.dumps({'model': 'deepseek-v4-flash', 'id': 'id',
            'usage': {'total_tokens': 50}, 'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps(value)}}]}).encode())

    def test_two_phases_no_gold_in_extraction_and_cache(self):
        row = {'model_answer': 'The answer is 0.5', 'answer': 'GOLD_SENTINEL',
               'question_for_eval': 'Question with choices', 'question': ''}
        judge = DeepSeekJudge(self.cfg['judge'], time.time() + 60)
        with patch('synselect.utils.mathverse_judge.build_opener') as opener:
            opener.return_value.open.side_effect = [self.response({'extraction': '0.5'}),
                self.response({'correct': True, 'reason': 'Equivalent'})]
            result = judge.score(row)
            self.assertTrue(result['correct'])
            calls = opener.return_value.open.call_args_list
            first = calls[0].args[0].data.decode()
            second = calls[1].args[0].data.decode()
            self.assertNotIn('GOLD_SENTINEL', first)
            self.assertIn('GOLD_SENTINEL', second)
            self.assertIn('Question with choices', second)
            self.assertEqual(json.loads(first)['thinking'], {'type': 'disabled'})
            self.assertTrue(judge.score(row)['judge_calls']['judge']['cache_hit'])
            self.assertEqual(opener.return_value.open.call_count, 2)
        self.assertTrue(all('secret-test-value' not in p.read_text() for p in (self.path / 'cache').glob('*.json')))

    def test_bad_schema_never_false_and_auth_stops(self):
        judge = DeepSeekJudge(self.cfg['judge'], time.time() + 60)
        with patch('synselect.utils.mathverse_judge.build_opener') as opener:
            opener.return_value.open.return_value = self.response({'correct': 'true', 'reason': ''})
            with self.assertRaises(JudgeError):
                judge.request('judge', {})
            opener.return_value.open.side_effect = HTTPError('url', 401, 'secret server text', {}, None)
            with self.assertRaisesRegex(JudgeError, '^HTTP 401$'):
                judge.request('extract', {})
            self.assertTrue(judge.fatal.is_set())

    def test_expired_deadline_no_request(self):
        judge = DeepSeekJudge(self.cfg['judge'], time.time() - 1)
        with patch('synselect.utils.mathverse_judge.build_opener') as opener:
            with self.assertRaisesRegex(JudgeError, 'deadline'):
                judge.request('extract', {})
            opener.assert_not_called()

    def test_unlimited_keeps_individual_request_timeout(self):
        judge = DeepSeekJudge(self.cfg['judge'], None)
        with patch('synselect.utils.mathverse_judge.build_opener') as opener:
            opener.return_value.open.return_value = self.response({'extraction': 'A'})
            self.assertEqual(judge.request('extract', {})['value']['extraction'], 'A')
            self.assertEqual(opener.return_value.open.call_args.kwargs['timeout'], 30)

    def test_errors_and_pending_excluded_from_scores(self):
        row = {'sample_index': '1', 'problem_index': '1', 'problem_version': 'Vision Only',
               'correct': True, 'extraction': 'A', 'truncated': False, 'model_answer': 'A', 'answer': 'A'}
        self.cfg['models'] = {'base': None}
        (self.path / 'base.jsonl').write_text(json.dumps(row) + '\n')
        (self.path / 'samples.json').write_text(json.dumps([row]))
        with patch('synselect.mathverse_score.DeepSeekJudge.score', side_effect=JudgeError('HTTP 429')):
            judge_saved(self.cfg, self.path, self.path, time.time() + 60)
        tables = scored_tables(self.cfg, self.path, self.path)
        report = summarize(tables, [row], ['Vision Only'], 'deepseek_api')
        self.assertEqual(report['unscored_by_model']['base'], 1)
        self.assertIsNone(report['models']['base']['all_completed']['judge_accuracy'])
        self.assertFalse(report['complete'])

    def test_rescore_uses_saved_models_no_gpu_or_training(self):
        self.cfg['no_time_limit'] = True
        source, output = self.path / 'old', self.path / 'new'
        source.mkdir()
        (source / 'samples.json').write_text('[]')
        (source / 'resolved_config.json').write_text(json.dumps({
            'models': {'old_model': '/nonexistent/adapter'}, 'dataset': {'versions': ['Vision Only']}}))
        (source / 'old_model.jsonl').write_text('')
        with patch('synselect.mathverse.run_child') as child, patch('builtins.print'):
            rescore(self.cfg, 'mathverse_1h', source, output)
            command = child.call_args.args[0]
            self.assertIn('_judge', command)
            self.assertNotIn('prepare', command)
            self.assertEqual(child.call_count, 1)
            self.assertIsNone(child.call_args.args[1])
            self.assertNotIn('--deadline', command)
            self.assertFalse(json.loads((output / 'report.json').read_text())['inference'])


if __name__ == '__main__':
    unittest.main()
