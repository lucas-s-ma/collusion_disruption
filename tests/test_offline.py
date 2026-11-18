import contextlib
import io
import json
import math
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import experiment as e


class OfflineChecks(unittest.TestCase):
    def test_dotenv_key_loading(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'.env'
            path.write_text('DEEPSEEK_API_KEY=synthetic-test-key\n')
            old=os.environ.pop('DEEPSEEK_API_KEY',None)
            try:
                self.assertEqual(e.api_key(path),'synthetic-test-key')
                os.environ['DEEPSEEK_API_KEY']='environment-wins'
                self.assertEqual(e.api_key(path),'environment-wins')
            finally:
                os.environ.pop('DEEPSEEK_API_KEY',None)
                if old is not None:os.environ['DEEPSEEK_API_KEY']=old

    def test_market_and_benchmarks(self):
        b = e.benchmarks()
        self.assertAlmostEqual(b['2']['nash_price'], 1.47293, places=5)
        self.assertAlmostEqual(b['3']['nash_price'], 1.37016, places=5)
        self.assertAlmostEqual(b['2']['joint_profit_price'], 1.92498, places=5)
        self.assertAlmostEqual(b['3']['joint_profit_price'], 2.0, places=5)
        m = e.market([1.9, 1.9, 1.6])
        self.assertAlmostEqual(sum(m['quantity']) + m['outside_quantity'], 100)
        self.assertAlmostEqual(e.myopic_price([1.9, 1.9]), 1.58287, places=5)
        self.assertLessEqual(e.myopic_price([1.9,1.9],1.0,1.4),1.4)
        with self.assertRaises(ValueError):e.validate_response('{"price":1.5,"notes":""}','stop',1.0,1.4)

    def test_prompt_isolation(self):
        cfg = {'window': 100, 'entry_round': 1, 'price_range': [0.0,4.5]}
        past = {1: {'firms': {'seller_1': {'price': 1.5, 'quantity': 20, 'profit': 10, 'notes': 'own secret'},
                              'seller_2': {'price': 2.0, 'quantity': 15, 'profit': 20, 'notes': 'rival secret'}}}}
        msgs = e.observation(cfg,past,'seller_1',['seller_1','seller_2'], 'own secret',2,False)
        prompt = json.dumps(msgs)
        self.assertIn('one other independent pricing agent',msgs[1]['content'])
        self.assertTrue(msgs[0]['content'].startswith(e.P1 + '\n\n'))
        self.assertIn('completed rounds 1 through 1',msgs[0]['content'])
        self.assertIn('for every seller active in each round',msgs[0]['content'])
        self.assertIn('Other sellers\' sales and profits are private',msgs[0]['content'])
        self.assertNotIn('three active sellers',msgs[1]['content'])
        self.assertNotIn('Seller_3 entered',msgs[1]['content'])
        self.assertIn('own secret',prompt)
        self.assertNotIn('rival secret',prompt)
        self.assertNotIn('consumer_surplus',prompt)
        self.assertNotIn('nash',prompt.lower())
        entrant = e.observation(cfg,past,'seller_3',['seller_1','seller_2','seller_3'],'',2,True)
        self.assertIsNone(json.loads(entrant[1]['content'].rsplit('\n',1)[1])['history'][0]['own_profit'])

    def test_strategic_entrant_uses_same_economic_and_output_instructions(self):
        cfg={'window':100,'entry_round':100,'price_range':[0.0,4.5]}
        past={100:{'firms':{
            'seller_1':{'price':1.63,'quantity':47.29,'profit':29.79},
            'seller_2':{'price':1.66,'quantity':41.94,'profit':27.68}}}}
        active=['seller_1','seller_2','seller_3']
        prompts={firm:e.observation(cfg,past,firm,active,'',101,True) for firm in active}
        self.assertEqual(len({p[0]['content'] for p in prompts.values()}),1)
        self.assertEqual(len({p[2]['content'] for p in prompts.values()}),1)
        for p in prompts.values():
            self.assertTrue(p[0]['content'].startswith(e.P1 + '\n\n'))
            self.assertIn('completed rounds 1 through 100',p[0]['content'])
            self.assertIn('two other independent pricing agents',p[1]['content'])
            self.assertIn('Rounds 1 through 100 had only seller_1 and seller_2',p[1]['content'])
            self.assertIn('Seller_3 entered in round 101',p[1]['content'])
            self.assertIn('were generated in a two-seller market',p[1]['content'])
            context=json.loads(p[1]['content'].rsplit('\n',1)[1])
            self.assertEqual(context['active_sellers'],active)
            self.assertIn('A third seller is active',context['market_update'])
        for firm in active:
            later=e.observation(cfg,past,firm,active,'',102,False)
            self.assertIn('two other independent pricing agents',later[1]['content'])
            self.assertIn('Seller_3 entered in round 101',later[1]['content'])
        control=e.observation(cfg,past,'seller_1',active[:2],'',102,False)
        self.assertIn('one other independent pricing agent',control[1]['content'])
        self.assertNotIn('two other independent pricing agents',control[1]['content'])
        self.assertNotIn('Seller_3 entered',control[1]['content'])
        entrant_context=json.loads(prompts['seller_3'][1]['content'].rsplit('\n',1)[1])
        self.assertIsNone(entrant_context['history'][0]['own_sales'])
        self.assertIsNone(entrant_context['history'][0]['own_profit'])

    def test_prompt_modes_keep_market_notices_identical(self):
        cfg={'window':100,'entry_round':100,'price_range':[0.0,4.5]}
        past={100:{'firms':{
            'seller_1':{'price':1.5,'quantity':40,'profit':20},
            'seller_2':{'price':1.6,'quantity':35,'profit':21}}}}
        active=['seller_1','seller_2','seller_3']
        conservative=e.observation({**cfg,'prompt_mode':'conservative'},past,'seller_1',active,'',101,True)
        risky=e.observation({**cfg,'prompt_mode':'risky'},past,'seller_1',active,'',101,True)
        self.assertTrue(conservative[0]['content'].startswith(e.P1 + '\n\n'))
        self.assertTrue(risky[0]['content'].startswith(e.P2 + '\n\n'))
        self.assertIn('risky or aggressive options for data-gathering purposes',risky[0]['content'])
        self.assertEqual(conservative[0]['content'][len(e.P1):],risky[0]['content'][len(e.P2):])
        self.assertEqual(conservative[1:],risky[1:])
        entrant=e.observation({**cfg,'prompt_mode':'risky'},past,'seller_3',active,'',101,True)
        self.assertEqual(risky[0]['content'],entrant[0]['content'])

    def test_v4_resume_keeps_its_original_entry_notice(self):
        cfg={'window':100,'entry_round':100,'price_range':[0.0,4.5],
             'prompt_version':'history_scope_v4','prompt_p1':e.P1}
        past={199:{'firms':{
            'seller_1':{'price':1.4,'quantity':30,'profit':12},
            'seller_2':{'price':1.4,'quantity':30,'profit':12},
            'seller_3':{'price':1.4,'quantity':30,'profit':12}}}}
        messages=e.observation(cfg,past,'seller_1',['seller_1','seller_2','seller_3'],'',200,False)
        self.assertIn('three active sellers',messages[1]['content'])
        self.assertNotIn('Rounds 1 through 100 had only',messages[1]['content'])
        self.assertTrue(messages[0]['content'].startswith(e.P1 + '\n\n'))
        self.assertIn('history_scope_v4',e.RESUMABLE_PROMPT_VERSIONS)

    def test_history_scope_follows_window(self):
        cfg={'window':2,'price_range':[0.0,4.5]}
        first=e.observation(cfg,{},'seller_1',['seller_1','seller_2'],'',1,False)
        self.assertIn('No completed market rounds are available yet.',first[0]['content'])
        past={r:{'firms':{'seller_1':{'price':1.5,'quantity':20,'profit':10}}} for r in range(1,4)}
        later=e.observation(cfg,past,'seller_1',['seller_1','seller_2'],'',4,False)
        self.assertIn('completed rounds 2 through 3',later[0]['content'])
        context=json.loads(later[1]['content'].rsplit('\n',1)[1])
        self.assertEqual([row['round'] for row in context['history']],[2,3])

    def test_attempt_cap_includes_retry(self):
        with tempfile.TemporaryDirectory() as t:
            db=e.connect(Path(t)/'s.sqlite3')
            c=e.Caller(db,{'retries':2,'model':'x','temperature':1.0,'max_tokens':256,'price_range':[0,4.5]},'mock',1)
            calls=[]
            class Failing:
                class chat:
                    class completions:
                        @staticmethod
                        def create(**kwargs):
                            calls.append(1)
                            raise TimeoutError('synthetic')
            c.mode='live';c.client=Failing()
            with self.assertRaises(e.AttemptLimit):
                c.decision(1,'pre',1,'seller_1',[])
            self.assertEqual(len(calls),1)
            self.assertEqual(c.count(),1)

    def test_mock_resume_and_files(self):
        progress_output=io.StringIO()
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(progress_output):
            base=['--mock','--run-id','test','--output-dir',t,'--rounds','4','--entry-round','2',
                  '--histories','1','--prompt-mode','risky']
            self.assertEqual(e.cli(base+['--request-limit','5']),2)
            run=Path(t)/'test'
            saved_config=json.loads((run/'config.json').read_text())
            self.assertEqual(saved_config['prompt_mode'],'risky')
            self.assertEqual(saved_config['prompt_prefix'],e.P2)
            db=e.connect(run/'state.sqlite3')
            self.assertEqual(db.execute('SELECT count(*) FROM rounds').fetchone()[0],2)
            self.assertEqual(e.cli(['--mock','--resume',str(run),'--request-limit','19']),0)
            self.assertEqual(db.execute('SELECT count(*) FROM rounds').fetchone()[0],8)
            self.assertEqual(db.execute('SELECT count(*) FROM rounds WHERE branch="pre"').fetchone()[0],2)
            self.assertEqual(db.execute('SELECT count(*) FROM attempts').fetchone()[0],19)
            ids={json.loads(d)['checkpoint'] for (d,) in db.execute('SELECT data FROM rounds')}
            self.assertEqual(ids,{'history_001_round_2'})
            for name in ['manifest.json','benchmarks.json','seller_round.csv','market_round.csv',
                         'private_notes.csv','request_diagnostics.csv','history_summary.csv','checkpoint_001.json']:
                self.assertTrue((run/name).exists(),name)
            for stem in ['price_trajectories','incumbent_gap_and_paired_difference',
                         'profits_and_consumer_surplus','late_outcomes_by_history']:
                for suffix in ['png','svg']:self.assertTrue((run/'figures'/f'{stem}.{suffix}').exists())
            self.assertEqual(e.cli(['--plot-only','--resume',str(run)]),0)
            with self.assertRaises(SystemExit):
                e.cli(['--mock','--resume',str(run),'--rounds','5'])
            with self.assertRaises(SystemExit):
                e.cli(['--mock','--resume',str(run),'--prompt-mode','conservative'])
            self.assertIn('Completed LLM decisions:',progress_output.getvalue())

    def test_dry_run(self):
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(e.cli(['--dry-run']),0)
        self.assertIn('Pre-entry LLM decisions: 200',out.getvalue())
        self.assertIn('full matched decisions: 900',out.getvalue())
        self.assertIn('No provider requests made',out.getvalue())
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(e.cli(['--dry-run','--pre-entry-only']),0)
        self.assertIn('full matched decisions: 200',out.getvalue())

    def test_completed_pilot_can_branch(self):
        with tempfile.TemporaryDirectory() as t, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            first=['--mock','--run-id','pilot','--output-dir',t,'--rounds','3','--entry-round','2','--pre-entry-only']
            self.assertEqual(e.cli(first),0)
            run=Path(t)/'pilot'
            db=e.connect(run/'state.sqlite3')
            self.assertEqual(db.execute('SELECT count(*) FROM rounds').fetchone()[0],2)
            self.assertEqual(e.cli(['--mock','--resume',str(run)]),0)
            self.assertEqual(db.execute('SELECT count(*) FROM rounds').fetchone()[0],5)
            self.assertEqual({r[0] for r in db.execute('SELECT DISTINCT branch FROM rounds')},
                             {'pre','no_entry','strategic_entry','myopic_entry'})


if __name__=='__main__':unittest.main()
