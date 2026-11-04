#!/usr/bin/env python3
"""Check interrupted acquisition and the extension comparison conventions."""

import hashlib
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/preparation'))
sys.path.insert(0, str(ROOT/'experiments'))
import acquire_data
import inspect_results


def digest(data):
    return hashlib.sha256(data).hexdigest()


class InterruptedStream(io.BytesIO):
    def __init__(self, payload, error):
        super().__init__(payload)
        self.error = error
        self.started = False

    def read(self, size=-1):
        if self.started:
            raise self.error
        self.started = True
        return super().read(3)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payload = b'complete source bytes\n'
        self.destination = self.root/'source.csv'

    def download(self, response, expected_hash=None):
        with patch.object(acquire_data.urllib.request, 'urlopen', return_value=response):
            return acquire_data.download('https://example.invalid/source.csv', self.destination,
                                         len(self.payload), expected_hash or digest(self.payload))

    def test_retry_after_network_and_keyboard_interruptions(self):
        for error in (ConnectionResetError('transfer interrupted'), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)):
                    self.download(InterruptedStream(self.payload, error))
                self.assertFalse(self.destination.exists())
                self.assertEqual(list(self.root.glob('*.download')), [])
        self.download(io.BytesIO(self.payload))
        self.assertEqual(self.destination.read_bytes(), self.payload)

    def test_legacy_partial_file_and_completed_file_reuse(self):
        legacy = self.root/'source.csv.download'
        legacy.write_bytes(b'partial previous attempt')
        self.download(io.BytesIO(self.payload))
        self.assertEqual(legacy.read_bytes(), b'partial previous attempt')
        self.assertEqual(self.destination.read_bytes(), self.payload)
        with patch.object(acquire_data.urllib.request, 'urlopen', side_effect=AssertionError('transfer repeated')):
            record = acquire_data.download('https://example.invalid/source.csv', self.destination,
                                           len(self.payload), digest(self.payload))
        self.assertEqual(record['sha256'], digest(self.payload))

    def test_hash_failure_cleans_stage_and_preserves_destination_validation(self):
        with self.assertRaises(ValueError):
            self.download(io.BytesIO(self.payload), '0'*64)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.glob('*.download')), [])
        self.destination.write_bytes(b'existing different bytes')
        with self.assertRaises(ValueError):
            self.download(io.BytesIO(self.payload))
        self.assertEqual(self.destination.read_bytes(), b'existing different bytes')

    def job_fixture(self):
        archive = self.root/'input.tgz'
        files = {'table.csv': self.payload, 'schema.sql': b'CREATE TABLE t (x INT)\n'}
        with tarfile.open(archive, 'w:gz') as bundle:
            for name, data in files.items():
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                bundle.addfile(entry, io.BytesIO(data))
        data = archive.read_bytes()
        config = {'archive': {'bytes': len(data), 'sha256': digest(data), 'schema_member': 'schema.sql'},
                  'tables': [{'member': 'table.csv', 'expected_bytes': len(self.payload),
                              'expected_sha256': digest(self.payload)}]}
        output = self.root/'extracted'
        output.mkdir()
        return archive, config, output

    def test_extraction_retry_with_legacy_partial(self):
        archive, config, output = self.job_fixture()
        legacy = output/'table.csv.extract'
        legacy.write_bytes(b'legacy partial')
        def interrupted_copy(source, target, length):
            target.write(source.read(3))
            raise OSError('extraction interrupted')
        with patch.object(acquire_data.shutil, 'copyfileobj', side_effect=interrupted_copy):
            with self.assertRaises(OSError):
                acquire_data.acquire_job(output, config, archive)
        self.assertFalse((output/'table.csv').exists())
        self.assertEqual(list(output.glob('*.extract')), [legacy])
        acquire_data.acquire_job(output, config, archive)
        self.assertEqual((output/'table.csv').read_bytes(), self.payload)
        self.assertEqual((output/'schema.sql').read_bytes(), b'CREATE TABLE t (x INT)\n')

    def test_extracted_table_hash_checked_before_publish(self):
        archive, config, output = self.job_fixture()
        config['tables'][0]['expected_sha256'] = '0'*64
        with self.assertRaises(ValueError):
            acquire_data.acquire_job(output, config, archive)
        self.assertFalse((output/'table.csv').exists())
        self.assertEqual(list(output.glob('*.extract')), [])


class ExtensionComparisonTests(unittest.TestCase):
    def test_per_process_cost_median_and_incomplete_baseline_exclusion(self):
        rows = []
        measurements = {'candidate': [(1,100),(2,2),(100,1),(101,101)],
                        'root': [(60,5)]*4, 'score': [(0.1,0.1)]*4}
        for method, values in measurements.items():
            for repeat, (build, call) in enumerate(values):
                complete = method!='score' or repeat!=3
                rows.append(dict(batch='extension', case_index=0, query_id='test',
                                 method=method, repeat=repeat, mode='product-real', expected_mu='1',
                                 planned_queries=1, status='completed' if complete else 'memory_limit',
                                 B_ms=build if complete else None, q_ms=call if complete else None,
                                 persistent_bytes=1 if complete else None, max_rss_kib=1))
        groups, summaries = inspect_results.summarize(rows)
        result, = inspect_results.comparisons(rows, groups, summaries)
        self.assertEqual(result['q_baseline'], 'root')
        self.assertEqual(result['cost_baseline'], 'root')
        self.assertEqual(result['cost_aggregation'], 'median(B+kq)')
        self.assertAlmostEqual(result['q_ratio'], 51/5)
        self.assertAlmostEqual(result['cost_ratio'], 101/65)


if __name__=='__main__':
    unittest.main()
