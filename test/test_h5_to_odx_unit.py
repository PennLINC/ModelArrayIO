"""Unit tests for the ODX results write-back (h5_to_odx + export-results routing).

These monkeypatch the ODX writer so they run without the optional ``odx``
package, mirroring ``test_mif_to_h5_unit.py``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from modelarrayio.cli import export_results as export_results_mod
from modelarrayio.cli import h5_to_odx as h5_to_odx_mod


def _results_h5(path: Path) -> None:
    with h5py.File(path, 'w') as f:
        grp = f.create_group('results/myana')
        # 2 metrics x 3 fixels
        rm = grp.create_dataset(
            'results_matrix',
            data=np.array([[1.0, 2.0, 3.0], [0.01, 0.5, 0.9]], dtype=np.float32),
        )
        rm.attrs['colnames'] = ['ses2.estimate', 'ses2.p.value']


def test_h5_to_odx_builds_result_dpf_dict(monkeypatch, tmp_path: Path) -> None:
    h5 = tmp_path / 'results.h5'
    _results_h5(h5)
    captured = {}

    def fake_write(template_odx, results, out_path):
        captured['template'] = template_odx
        captured['results'] = results
        captured['out'] = out_path
        return Path(out_path)

    monkeypatch.setattr(h5_to_odx_mod, 'write_odx_results', fake_write)
    status = h5_to_odx_mod.h5_to_odx(
        example_odx='tmpl.odx',
        in_file=h5,
        analysis_name='myana',
        compress=False,
        output_dir=tmp_path,
    )
    assert status == 0
    assert captured['template'] == 'tmpl.odx'
    assert captured['out'] == tmp_path / 'myana.odx'
    res = captured['results']
    # estimate metric carried through; p.value gets a 1m.p.value companion
    assert np.allclose(res['myana_ses2.estimate'], [1.0, 2.0, 3.0])
    assert np.allclose(res['myana_ses2.p.value'], [0.01, 0.5, 0.9])
    assert np.allclose(res['myana_ses2.1m.p.value'], [0.99, 0.5, 0.1])


def test_export_results_routes_odx(monkeypatch, tmp_path: Path) -> None:
    h5 = tmp_path / 'results.h5'
    _results_h5(h5)
    calls = {}

    def fake_h5_to_odx(**kwargs):
        calls.update(kwargs)
        return 0

    monkeypatch.setattr(export_results_mod, 'h5_to_odx', fake_h5_to_odx)
    status = export_results_mod.export_results(
        in_file=h5,
        analysis_name='myana',
        output_dir=tmp_path / 'out',
        example_file=str(tmp_path / 'group.odx'),  # .odx → odx modality
    )
    assert status == 0
    # routed to the ODX writer (not mis-routed to CIFTI), with the .odx template
    assert calls['example_odx'] == str(tmp_path / 'group.odx')
    assert calls['analysis_name'] == 'myana'
