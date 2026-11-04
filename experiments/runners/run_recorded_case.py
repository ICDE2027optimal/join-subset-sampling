#!/usr/bin/env python3
"""Rerun one recorded case on a supplied prepared query directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from run_controlled import execute_one, write_json

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'experiments/results'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, choices=(
        'stats_original', 'controlled', 'job1a', 'materialized_paired', 'extension'))
    parser.add_argument('--case', required=True, type=int)
    parser.add_argument('--query-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build')
    parser.add_argument('--methods', nargs='+', help='Use selected method IDs from the recorded case')
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    rows = [json.loads(line) for line in (RESULTS/'measurements.jsonl').read_text().splitlines()]
    selected = sorted((r for r in rows if r['batch']==args.batch and r['case_index']==args.case),
                      key=lambda r:r['sequence'])
    if not selected:
        parser.error('Choose an existing case index in the selected batch')
    if args.methods:
        unknown = set(args.methods)-{r['method'] for r in selected}
        if unknown:
            parser.error('Methods outside this case: '+', '.join(sorted(unknown)))
        selected = [r for r in selected if r['method'] in args.methods]
    original = {r['record_key']:r for r in map(json.loads,(RESULTS/'records.jsonl').read_text().splitlines())}
    query = args.query_dir.resolve()
    if not (query/'structure.tsv').is_file():
        parser.error('The query directory must contain structure.tsv')
    output = args.output.resolve()
    if output.exists():
        parser.error('Choose a new output directory')
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else [0]
    cpu = allowed[0] if args.cpu is None else args.cpu
    if args.execute and (not sys.platform.startswith('linux') or cpu not in allowed):
        parser.error('Execution requires Linux and an allowed CPU')
    plan=[]
    for row in selected:
        parameters = original[row['record_key']]['planned_parameters']
        driver = args.build_dir.resolve()/('materialized_explicit_driver'
                     if row['method']=='materialized-explicit' else 'formal_driver')
        if not driver.is_file():
            parser.error('Build the required driver: '+driver.name)
        command=[str(driver),'--query-dir',str(query)]
        for key,value in parameters.items():
            command.extend(['--'+key,str(value)])
        process = {key:row[key] for key in ('process_id','case_index','method','repeat','sequence')}
        process.update(seed=int(row['seed']),cpu=cpu,memory_bytes=row['memory_limit_bytes'],
                       timeout_seconds=row['timeout_seconds'],
                       relative_directory='processes/'+row['process_id'],command=command,
                       source_record_key=row['record_key'],
                       driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest())
        case = dict(chosen_root=row['root'],join_size=row['join_size'],
                    actual_mu=parameters['expected-mu'],max_p=parameters['expected-maxp'],
                    sum_p2=parameters['expected-sump2'],warmup=row['planned_warmup_queries'],
                    queries=row['planned_queries'])
        plan.append((process,case))
    output.mkdir(parents=True)
    write_json(output/'manifest.json',{'batch':args.batch,'case_index':args.case,
                                     'processes':[p for p,c in plan]})
    if not args.execute:
        print(json.dumps({'planned_processes':len(plan),'manifest':str(output/'manifest.json')}))
        return 0
    outcomes=[]
    with (output/'progress.jsonl').open('w',encoding='utf-8') as progress:
        for process,case in plan:
            result=execute_one(process,case,output,Path('/usr/bin/time'))
            stderr=output/process['relative_directory']/'stderr.txt'
            if result['status']=='unexpected_exit' and stderr.exists() and 'std::bad_alloc' in stderr.read_text():
                result.update(status='memory_limit',censored=True,
                              error='Allocation failure under the configured address-space limit')
            write_json(output/process['relative_directory']/'record.json',result)
            progress.write(json.dumps(result)+'\n')
            progress.flush()
            outcomes.append(result['status'])
            print(json.dumps({'process':process['process_id'],'status':result['status']}),flush=True)
            if result['status'] not in ('completed','timeout','memory_limit'):
                break
    summary={status:outcomes.count(status) for status in sorted(set(outcomes))}
    write_json(output/'summary.json',dict(planned_processes=len(plan),outcomes=summary))
    return 0 if len(outcomes)==len(plan) and set(outcomes)<= {'completed','timeout','memory_limit'} else 2


if __name__=='__main__':
    raise SystemExit(main())
