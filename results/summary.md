| Model | Method | Problems | pass@1 (95% CI) | Final (95% CI) | Gain | Oracle pass@N | Mean sample acc | Avg tokens | Avg sec |
|---|---|---|---|---|---|---|---|---|---|
| allam-2-7b | retry (examples, k=3, fallback to attempt 1) | 50 | 32.0% (21%-46%) | 36.0% (24%-50%) | +4.0% | - | - | 1672 | 34.9 |
| allam-2-7b | retry (self, k=3, fallback to attempt 1) | 50 | 30.0% (19%-44%) | 30.0% (19%-44%) | +0.0% | - | - | 2502 | 47.2 |
| allam-2-7b | consensus (N=10) | 50 | 38.0% (26%-52%) | 52.0% (39%-65%) | +18.6% | 62.0% (48%-74%) | 33.4% | 7046 | 74.1 |
| openai_gpt-oss-120b | retry (self, k=5) | 80 | 96.2% (90%-99%) | 100.0% (95%-100%) | +3.8% | - | - | 1486 | 31.0 |
| openai_gpt-oss-20b | retry (self, k=5) | 80 | 95.0% (88%-98%) | 100.0% (95%-100%) | +5.0% | - | - | 1456 | 24.8 |
| openai_gpt-oss-20b | consensus (N=10) | 5 | 100.0% (57%-100%) | 100.0% (57%-100%) | +4.0% | 100.0% (57%-100%) | 96.0% | 4906 | 27.9 |

Gain: retry = final - pass@1; consensus = final - mean sample acc. pass@1 = first attempt (retry) or one sampled candidate at T>0 (consensus). Mean sample acc = average accuracy over all N candidates (expected pass@1 at that temperature). Oracle = at least one of N candidates passes (upper bound for any selection method).

Paired comparison (same problems):
- **allam-2-7b**, consensus (N=10) vs retry (examples, k=3, fallback to attempt 1) on 50 common problems: consensus fixes 9, breaks 1.
- **openai_gpt-oss-20b**, consensus (N=10) vs retry (self, k=5) on 5 common problems: consensus fixes 0, breaks 0.

Ablation (does clustering add anything beyond the free docstring-example filter?):
- allam-2-7b, consensus (N=10): filter-only (first candidate passing the docstring examples, no clustering) = 52.0% (39%-65%) vs full consensus = 52.0% (39%-65%).

Failure taxonomy:
- **allam-2-7b**, retry (examples, k=3, fallback to attempt 1)
  - final answers: wrong_answer 21, pass 18, runtime_error 7, syntax_error 3, killed 1
  - own tests vs hidden tests: code wrong, tests caught it 28, tests ok, code ok 18, tests too weak (accepted wrong code) 4
- **allam-2-7b**, retry (self, k=3, fallback to attempt 1)
  - own tests vs hidden tests: code wrong, tests caught it 23, tests wrong (rejected correct code) 12, tests too weak (accepted wrong code) 12, tests ok, code ok 3
- **allam-2-7b**, consensus (N=10)
  - final answers: correct 26, no correct candidate, mostly wrong_answer 15, selection miss (a correct candidate existed) 5, no correct candidate, mostly runtime_error 3, no correct candidate, mostly syntax_error 1
  - every candidate: wrong_answer 44%, pass 33%, runtime_error 16%, syntax_error 5%, timeout 1%, killed 0%
  - runtime errors: TypeError 25, NameError 19, IndexError 15, ValueError 7, ModuleNotFoundError 5
- **openai_gpt-oss-120b**, retry (self, k=5)
  - own tests vs hidden tests: tests ok, code ok 65, tests wrong (rejected correct code) 15
- **openai_gpt-oss-20b**, retry (self, k=5)
  - own tests vs hidden tests: tests ok, code ok 62, tests wrong (rejected correct code) 18
- **openai_gpt-oss-20b**, consensus (N=10)
  - final answers: correct 5
