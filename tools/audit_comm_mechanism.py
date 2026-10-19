# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from attribute_comm_free import digest, intersection_ns


def audit_rank(path):
    con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    try:
        strings = dict(con.execute('SELECT id,value FROM STRING_IDS'))
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )]
        calls = [dict(r) for r in con.execute(
            'SELECT o.*,s.value AS name FROM COMMUNICATION_OP o '
            'JOIN STRING_IDS s ON o.opName=s.id ORDER BY startNs,opId'
        )]
        apis = [dict(r) for r in con.execute('SELECT * FROM PYTORCH_API')]
        for api in apis:
            api['name'] = strings.get(api['name'], str(api['name']))
            api['shape_text'] = strings.get(api['inputShapes'])
        reductions = [r for r in apis if 'allreduce' in r['name'].lower()
                      or 'all_reduce' in r['name'].lower()]
        by_name = {}
        for name in sorted({r['name'] for r in reductions}):
            rows = [r for r in reductions if r['name'] == name]
            by_name[name] = {
                'calls': len(rows),
                'nonempty_shapes': sum(bool(r['shape_text']) for r in rows),
                'nonnull_callchain_ids': sum(
                    r['callchainId'] is not None for r in rows
                ),
                'shape_values': dict(Counter(r['shape_text'] for r in rows)),
                'samples': rows[:2],
            }
        graph = [dict(r) for r in con.execute(
            'SELECT c.*,s.value AS api_name FROM CANN_API c '
            'JOIN STRING_IDS s ON c.name=s.id '
            "WHERE s.value LIKE '%Execute%' OR s.value LIKE '%Graph%' "
            "OR s.value LIKE '%Capture%' ORDER BY c.startNs"
        )]
        execute = [r for r in graph if r['api_name'] == 'aclmdlRIExecuteAsync']
        free = [tuple(r) for r in con.execute(
            'SELECT startNs,endNs FROM OVERLAP_ANALYSIS WHERE type=3'
        )]
        valid = [(r['startNs'], r['endNs']) for r in execute
                 if r['endNs'] > r['startNs']]
        free_ns = sum(b-a for a, b in free)
        intersect = intersection_ns(free, valid)
        tasks = [dict(r) for r in con.execute(
            'SELECT modelId,streamId,contextId,COUNT(*) AS tasks,'
            'MIN(startNs) AS startNs,MAX(endNs) AS endNs '
            'FROM TASK GROUP BY modelId,streamId,contextId'
        )]
        result = {
            'db': {'path': str(path), 'sha256': digest(path)},
            'tables': tables,
            'communication_calls': len(calls),
            'allreduce_payload_elements': dict(Counter(
                str(r['count']) for r in calls if r['name'].startswith('hcom_allReduce')
            )),
            'allreduce_api_fields': by_name,
            'all_api_shapes': sum(bool(r['shape_text']) for r in apis),
            'callchain_tables': [t for t in tables if 'CALLCHAIN' in t.upper()],
            'graph_host_api_intervals': graph,
            'execute_intervals': len(execute),
            'execute_intervals_with_positive_duration': len(valid),
            'task_model_stream_context_ranges': tasks,
            'free_ms': free_ns / 1e6,
            'free_intersect_execute_host_api_ms': intersect / 1e6,
            'free_outside_execute_host_api_ms': (free_ns-intersect) / 1e6,
            'boundary_limit': (
                'Async host API end is not device graph completion. '
                'TASK grouped extents are not verified replay boundaries. '
                'No safe overlap follows from these temporal intersections.'
            ),
        }
        return result, [(r['name'], r['count'], r['dataType'], r['rankSize'])
                        for r in calls]
    finally:
        con.close()


def audit(paths):
    rows = [audit_rank(p) for p in paths]
    return {
        'schema': 'comm-mechanism-audit.v1',
        'ranks': [r[0] for r in rows],
        'rank_pair_communication_name_payload_dtype_size_equal': (
            rows[0][1] == rows[1][1]
        ),
        'status': ('blocked pending reliable per-call callsite/phase '
                   'and device graph boundaries'),
        'scope': ('diagnostic only; not L3, performance acceptance '
                  'or overlap feasibility'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.db) != 2 or args.db[0].resolve() == args.db[1].resolve():
        parser.error('two independent rank DB paths required')
    result = audit(args.db)
    with args.output.open('x') as out:
        json.dump(result, out, indent=2)
        out.write('\n')


if __name__ == '__main__':
    main()
