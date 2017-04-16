from vbench.api import BenchmarkRunner
from settings import *

runner = BenchmarkRunner(benchmarks, REPO_PATH, REPO_URL,
                         BUILD, DB_PATH, TMP_DIR, PREPARE,
                         run_option='eod', start_date=START_DATE,
                         module_dependencies=dependencies)
runner.run()

