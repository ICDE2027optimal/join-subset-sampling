#!/usr/bin/env python3
"""Verify supplied records and recompute paper comparison statistics."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import median


RESULTS = Path(__file__).resolve().parent / 'results'


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['batch'], row['case_index'], row['method']].append(row)
    summaries = {}
    for key, group in groups.items():
        good = [row for row in group if row['status'] == 'completed']
        item = {'complete': len(group) == len(good) == 4,
                'status_counts': dict(Counter(row['status'] for row in group)),
                'metrics': {}}
        for field in ('B_ms', 'q_ms', 'persistent_bytes', 'max_rss_kib'):
            values = [row[field] for row in good if row[field] is not None]
            item['metrics'][field] = None if not values else {
                'median': median(values), 'min': min(values), 'max': max(values),
                'range': max(values) - min(values)}
        summaries[key] = item
    return groups, summaries


def comparisons(rows, groups, summaries):
    cases = {}
    for row in rows:
        cases.setdefault((row['batch'], row['case_index']), row)
    output = []
    for (batch, case), row in cases.items():
        ours = summaries.get((batch, case, 'candidate'))
        if ours is None or not ours['complete']:
            continue
        k = row['planned_queries']
        item = {'batch': batch, 'case_index': case, 'query_id': row['query_id'],
                'mode': row['mode'], 'mu': float(row['expected_mu']), 'k': k}
        def metric(method, field):
            return summaries[batch, case, method]['metrics'][field]['median']
        if batch == 'materialized_paired':
            methods = ('candidate', 'materialized-explicit')
            if not summaries[batch, case, methods[1]]['complete']:
                continue
            paired = [{r['repeat']: r for r in groups[batch, case, method]} for method in methods]
            for name, value in (
                ('B_ratio', lambda r: r['B_ms']),
                ('q_ratio', lambda r: r['q_ms']),
                ('cost_ratio', lambda r: r['B_ms'] + k * r['q_ms'])):
                ratios = [value(paired[0][i]) / value(paired[1][i]) for i in range(4)]
                item[name] = {'median': median(ratios), 'min': min(ratios), 'max': max(ratios)}
            bo, bm = (metric(m, 'B_ms') for m in methods)
            qo, qm = (metric(m, 'q_ms') for m in methods)
            item.update(B_ours_ms=bo, B_materialized_ms=bm, q_ours_ms=qo, q_materialized_ms=qm)
            item['break_even_calls'] = (bm-bo)/(qo-qm) if qo != qm else None
            item['ours_lower_cost_integer_calls'] = (
                max(0, math.ceil(item['break_even_calls'])-1) if bm>bo and qo>qm else None)
        elif batch in ('stats_original', 'controlled', 'job1a', 'extension'):
            baselines = [method for b, c, method in summaries
                         if b == batch and c == case and method != 'candidate'
                         and summaries[b, c, method]['complete']]
            if not baselines:
                continue
            def cost(method):
                if batch == 'extension':
                    return median(r['B_ms']+k*r['q_ms'] for r in groups[batch, case, method])
                return metric(method, 'B_ms')+k*metric(method, 'q_ms')
            qbase = min(baselines, key=lambda m: metric(m, 'q_ms'))
            cbase = min(baselines, key=cost)
            item.update(q_baseline=qbase, cost_baseline=cbase,
                        q_ratio=metric('candidate', 'q_ms')/metric(qbase, 'q_ms'),
                        cost_ratio=cost('candidate')/cost(cbase),
                        cost_aggregation='median(B+kq)' if batch=='extension' else 'median(B)+k*median(q)')
        else:
            continue
        output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Write the recomputed metrics to a new JSON file')
    args = parser.parse_args()
    manifest = json.loads((RESULTS/'manifest.json').read_text())
    for item in manifest['products']:
        data = (RESULTS/item['name']).read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError('Evidence digest mismatch: '+item['name'])
    rows = [json.loads(line) for line in (RESULTS/'measurements.jsonl').read_text().splitlines()]
    if len(rows) != manifest['total_processes'] or len({r['record_key'] for r in rows}) != len(rows):
        raise ValueError('Process inventory mismatch')
    groups, summaries = summarize(rows)
    reference = json.loads((RESULTS/'aggregates.json').read_text())['groups']
    checked = 0
    for ref in reference:
        actual = summaries[ref['batch'], ref['case_index'], ref['method']]
        if actual['status_counts'] != ref['status_counts']:
            raise ValueError('Status summary mismatch')
        for field, expected in ref['metrics'].items():
            got = actual['metrics'][field]
            if expected is None or expected['n'] == 0:
                if got is not None:
                    raise ValueError('Missing metric mismatch')
            else:
                for stat in ('min', 'median', 'max'):
                    if not math.isclose(got[stat], expected[stat], rel_tol=2e-15, abs_tol=1e-15):
                        raise ValueError('Metric mismatch: '+str((ref['batch'], ref['case_index'], field, stat)))
                    checked += 1
    comp = comparisons(rows, groups, summaries)
    product = [c for c in comp if c['batch']=='stats_original' and c['mode'].startswith('product-')]
    paired = [c for c in comp if c['batch']=='materialized_paired']
    extension = [c for c in comp if c['batch']=='extension']
    overview = {
        'processes': len(rows), 'statuses': dict(Counter(r['status'] for r in rows)),
        'checked_summary_values': checked,
        'stats_product_cases': len(product),
        'stats_product_sampling_wins': sum(c['q_ratio']<1 for c in product),
        'stats_product_cost_wins': sum(c['cost_ratio']<1 for c in product),
        'paired_cases': len(paired),
        'paired_materialized_sampling_wins': sum(c['q_ratio']['median']>1 for c in paired),
        'paired_ours_scheduled_cost_wins': sum(c['cost_ratio']['median']<1 for c in paired),
        'paired_ours_integer_call_benefit_cases': sum((c['ours_lower_cost_integer_calls'] or 0)>0 for c in paired),
        'extension': {query: {
            'cases': sum(c['query_id']==query for c in extension),
            'sampling_wins': sum(c['query_id']==query and c['q_ratio']<1 for c in extension),
            'cost_wins': sum(c['query_id']==query and c['cost_ratio']<1 for c in extension)}
            for query in sorted({c['query_id'] for c in extension})}}
    print(json.dumps(overview, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump({'overview': overview, 'comparisons': comp,
                       'method_summaries': [dict(batch=b,case_index=c,method=m,**value)
                                            for (b,c,m),value in summaries.items()]},stream,indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
