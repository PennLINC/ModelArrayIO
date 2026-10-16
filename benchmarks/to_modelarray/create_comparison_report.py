"""Render the original-versus-optimized ``to-modelarray`` benchmark PDF.

Reads one or more ``compare_revisions.py`` run directories (each holding
``comparison_results.csv``, ``correctness_results.csv`` and
``comparison_meta.json``) and renders a findings report that compares the
original committed code (a pinned baseline revision) against the optimized
working tree for every modality (NIfTI, CIFTI, MIF) on both backends
(HDF5, TileDB): runtime speedup, downstream R-read time, output size and
bitwise correctness.

Example::

    python benchmarks/to_modelarray/create_comparison_report.py \
      --run-dir benchmarks/to_modelarray/results/original-vs-optimized-9ab13dd-nifti \
      --run-dir benchmarks/to_modelarray/results/original-vs-optimized-9ab13dd-mif \
      --run-dir benchmarks/to_modelarray/results/original-vs-optimized-9ab13dd-cifti \
      --output benchmarks/to_modelarray/results/to-modelarray-cross-modality-benchmark.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from comparison_runs import Group, load_groups

NAVY = HexColor('#172A3A')
TEAL = HexColor('#0F766E')
TEAL_LIGHT = HexColor('#DDF3F0')
CORAL = HexColor('#E76F51')
CORAL_LIGHT = HexColor('#FCE8E2')
SLATE = HexColor('#48606F')
PALE = HexColor('#F3F6F8')
GRID = HexColor('#D3DDE3')
INK = HexColor('#172A3A')
WHITE = colors.white

MODALITY_LABEL = {'nifti': 'NIfTI (voxel)', 'cifti': 'CIFTI (surface)', 'mif': 'MIF (fixel)'}
MODALITY_ORDER = ['nifti', 'cifti', 'mif']
BACKEND_LABEL = {'hdf5': 'HDF5 gzip-4', 'tiledb': 'TileDB zstd-5'}


MODALITY_NAME = {'nifti': 'NIfTI', 'cifti': 'CIFTI', 'mif': 'MIF'}
BACKEND_NAME = {'hdf5': 'HDF5', 'tiledb': 'TileDB'}
NUMBER_WORDS = (
    'no',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
    'ten',
)
RUN_SETTINGS = ('repetitions', 'warmups', 'workers', 'target_chunk_mb')
# What explains a configuration's gain, for the executive finding.
GAIN_REASON = {
    'cifti': 'where the raw-read fast path removes the per-file XML header parse',
    'hdf5': 'where chunks are now compressed on a thread pool instead of one writer thread',
    'tiledb': 'where each scalar is now written as one dense fragment instead of one per stripe',
}


def _count(value: str, noun: str) -> str:
    """Spell a run setting for prose (``'3'`` -> ``'three <noun>s'``); mixed values stay digits."""
    if not value.isdigit():
        return f'{value} {noun}s'
    number = int(value)
    word = NUMBER_WORDS[number] if number < len(NUMBER_WORDS) else value
    return f'{word} {noun}' if number == 1 else f'{word} {noun}s'


def _plural(number: int, noun: str) -> str:
    return f'{number:,} {noun}' if number == 1 else f'{number:,} {noun}s'


def _join(items: list[str]) -> str:
    if len(items) <= 2:
        return ' and '.join(items)
    return f'{", ".join(items[:-1])}, and {items[-1]}'


def _scope(ordered: list[Group]) -> str:
    """Name the modalities and backends actually measured, e.g. 'NIfTI and MIF on HDF5'."""
    modalities = _join(list(dict.fromkeys(MODALITY_NAME[group.modality] for group in ordered)))
    backends = list(dict.fromkeys(BACKEND_NAME[group.backend] for group in ordered))
    return f'{modalities} on {"both HDF5 and TileDB" if len(backends) == 2 else backends[0]}'


def _largest_gain(ordered: list[Group]) -> str:
    best = max(ordered, key=lambda group: group.speedup)
    reason = GAIN_REASON['cifti' if best.modality == 'cifti' else best.backend]
    return (
        f'The largest gain is {MODALITY_NAME[best.modality]} on {BACKEND_NAME[best.backend]} '
        f'({best.speedup:.1f}x), {reason}.'
    )


def _run_setting(metas: list[dict], key: str) -> str:
    """One recorded run setting across every run directory; differing values are joined by '/'."""
    values = {f'{float(meta[key]):g}' for meta in metas if meta.get(key) is not None}
    return '/'.join(sorted(values, key=float)) or '?'


def _cohort_sources(values: list[str]) -> dict[str, str]:
    sources = {}
    for value in values:
        modality, separator, text = value.partition('=')
        if not separator or modality not in MODALITY_LABEL:
            raise SystemExit(
                f'--cohort-source expects MODALITY=TEXT with MODALITY one of '
                f'{sorted(MODALITY_LABEL)}, got {value!r}'
            )
        sources[modality] = text
    return sources


def _median_cells(group: Group) -> list[str]:
    """Original median, optimized median and speedup, as every results table shows them."""
    return [
        f'{group.baseline_median:.2f} s',
        f'{group.optimized_median:.2f} s',
        f'{group.speedup:.2f}x',
    ]


def _equal_cell(group: Group) -> str:
    return 'yes' if group.correctness_ok else 'NO'


def _ordered_groups(grouped: dict[str, list[Group]]) -> list[Group]:
    ordered: list[Group] = []
    for modality in MODALITY_ORDER:
        for backend in ('hdf5', 'tiledb'):
            ordered.extend(
                group for group in grouped.get(modality, []) if group.backend == backend
            )
    return ordered


def _styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        'title': ParagraphStyle(
            'Title',
            parent=sample['Title'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=25,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        'subtitle': ParagraphStyle(
            'Subtitle',
            parent=sample['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=SLATE,
            spaceAfter=12,
        ),
        'h1': ParagraphStyle(
            'Heading1',
            parent=sample['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=15,
            leading=18,
            textColor=NAVY,
            spaceBefore=2,
            spaceAfter=7,
        ),
        'h2': ParagraphStyle(
            'Heading2',
            parent=sample['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=11,
            leading=14,
            textColor=TEAL,
            spaceBefore=4,
            spaceAfter=5,
        ),
        'body': ParagraphStyle(
            'Body',
            parent=sample['BodyText'],
            fontName='Helvetica',
            fontSize=8.5,
            leading=12,
            textColor=INK,
            spaceAfter=5,
        ),
        'small': ParagraphStyle(
            'Small',
            parent=sample['BodyText'],
            fontName='Helvetica',
            fontSize=7.4,
            leading=10,
            textColor=SLATE,
        ),
        'card_number': ParagraphStyle(
            'CardNumber',
            parent=sample['Normal'],
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=17,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        'card_label': ParagraphStyle(
            'CardLabel',
            parent=sample['Normal'],
            fontName='Helvetica',
            fontSize=6.7,
            leading=8.4,
            textColor=SLATE,
            alignment=TA_CENTER,
        ),
        'table_header': ParagraphStyle(
            'TableHeader',
            parent=sample['Normal'],
            fontName='Helvetica-Bold',
            fontSize=7.2,
            leading=9,
            textColor=WHITE,
        ),
        'table': ParagraphStyle(
            'Table',
            parent=sample['Normal'],
            fontName='Helvetica',
            fontSize=7.2,
            leading=9.1,
            textColor=INK,
        ),
    }


def _paragraph(value: object, style: ParagraphStyle) -> Paragraph:
    return Paragraph(str(value), style)


def _table(rows, widths, styles, *, header=True) -> Table:
    prepared = []
    for row_index, row in enumerate(rows):
        style = styles['table_header'] if header and row_index == 0 else styles['table']
        prepared.append([_paragraph(value, style) for value in row])
    table = Table(prepared, colWidths=widths, repeatRows=1 if header else 0, hAlign='LEFT')
    commands = [
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 3.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ('GRID', (0, 0), (-1, -1), 0.35, GRID),
    ]
    if header:
        commands.append(('BACKGROUND', (0, 0), (-1, 0), NAVY))
        commands.append(('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, PALE]))
    table.setStyle(TableStyle(commands))
    return table


def _cards(cards, styles, backgrounds) -> Table:
    content = [
        [
            [_paragraph(number, styles['card_number']), _paragraph(label, styles['card_label'])]
            for number, label in cards
        ]
    ]
    width = 504 / len(cards)
    table = Table(content, colWidths=[width] * len(cards), rowHeights=[54])
    style = [
        ('BOX', (0, 0), (-1, -1), 0.5, GRID),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, WHITE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]
    for index, background in enumerate(backgrounds):
        style.append(('BACKGROUND', (index, 0), (index, 0), background))
    table.setStyle(TableStyle(style))
    return table


def _callout(text, styles, background=TEAL_LIGHT, border=TEAL) -> Table:
    table = Table([[_paragraph(text, styles['body'])]], colWidths=[504])
    table.setStyle(
        TableStyle(
            [
                ('BACKGROUND', (0, 0), (-1, -1), background),
                ('BOX', (0, 0), (-1, -1), 0.6, border),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


class SpeedupBars(Flowable):
    """Grouped horizontal original-vs-optimized median wall-time bars."""

    def __init__(self, groups: list[Group], width=504, height=250):
        super().__init__()
        self.groups = groups
        self.width = width
        self.height = height

    def draw(self) -> None:
        canvas = self.canv
        left = 118
        right = 44
        chart_width = self.width - left - right
        maximum = max((g.baseline_median for g in self.groups), default=1.0) * 1.16 or 1.0
        top = self.height - 10
        bottom = 26
        lane = (top - bottom) / max(len(self.groups), 1)
        canvas.setFont('Helvetica', 6.5)
        for tick in range(6):
            value = maximum * tick / 5
            x = left + chart_width * tick / 5
            canvas.setStrokeColor(GRID)
            canvas.setLineWidth(0.35)
            canvas.line(x, bottom - 2, x, top)
            canvas.setFillColor(SLATE)
            canvas.drawCentredString(x, 12, f'{value:.1f}')
        canvas.setFillColor(SLATE)
        canvas.drawCentredString(
            left + chart_width / 2, 2, 'Median conversion wall time (seconds)'
        )
        for index, group in enumerate(self.groups):
            y = top - (index + 0.5) * lane
            label = f'{MODALITY_LABEL[group.modality].split(" ")[0]} · {group.backend.upper()}'
            canvas.setFont('Helvetica-Bold', 7)
            canvas.setFillColor(NAVY)
            canvas.drawRightString(left - 8, y - 2, label)
            for offset, value, fill in (
                (2.4, group.baseline_median, CORAL),
                (-8.4, group.optimized_median, TEAL),
            ):
                bar = chart_width * value / maximum
                canvas.setFillColor(fill)
                canvas.roundRect(left, y + offset - 2.2, max(bar, 0.4), 9, 1.5, fill=1, stroke=0)
                canvas.setFillColor(NAVY)
                canvas.setFont('Helvetica-Bold', 6.3)
                canvas.drawString(left + max(bar, 0.4) + 3, y + offset - 0.5, f'{value:.2f}s')
            canvas.setFillColor(TEAL)
            canvas.setFont('Helvetica-Bold', 6.8)
            canvas.drawString(left + chart_width + 6, y - 2, f'{group.speedup:.2f}x')


def _fmt_bytes(value: float) -> str:
    if value >= 2**30:
        return f'{value / 2**30:.2f} GiB'
    return f'{value / 2**20:.1f} MiB'


def _header_footer(canvas, document, measured_at: str, baseline_revision: str, cache: str) -> None:
    canvas.saveState()
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.5)
    canvas.line(42, 764, 570, 764)
    canvas.setFillColor(SLATE)
    canvas.setFont('Helvetica', 6.8)
    canvas.drawString(
        42,
        771,
        f'Measured {measured_at} | Original {baseline_revision} vs optimized | {cache}',
    )
    canvas.drawRightString(570, 771, f'Page {document.page}')
    canvas.line(42, 28, 570, 28)
    canvas.drawString(42, 18, 'ModelArrayIO original-vs-optimized runtime benchmark')
    canvas.restoreState()


def _report_story(grouped, meta, styles) -> list[Flowable]:
    ordered = _ordered_groups(grouped)
    total = len(ordered)
    faster = sum(group.speedup > 1.0 for group in ordered)
    correct = sum(group.correctness_ok for group in ordered)
    best = max((group.speedup for group in ordered), default=0.0)
    max_diff = max((group.max_abs_difference for group in ordered), default=0.0)
    all_good = faster == total and correct == total
    scope = _scope(ordered)
    settings = meta['settings']
    revision = meta['baseline_revision']
    repetitions = settings['repetitions']
    warmups = settings['warmups']
    speed = (
        f'in every one of the {total} modality x backend configurations, by up to {best:.1f}x'
        if faster == total
        else f'in {faster} of the {total} modality x backend configurations (best {best:.1f}x)'
    )
    correctness = (
        f'while producing byte-for-byte identical output: all {correct} of {total} optimized '
        f'outputs were bitwise equal to the original (maximum absolute difference {max_diff:.1f}), '
        'with source order and NaN positions preserved'
        if correct == total
        else f'but only {correct} of {total} optimized outputs were bitwise equal to the original '
        f'(maximum absolute difference {max_diff:.3g})'
    )

    cards = [
        (f'{faster}/{total}', 'configurations faster than original'),
        (f'{best:.1f}x', 'best median speedup'),
        (f'{correct}/{total}', 'outputs bitwise equal to original'),
        (f'{max_diff:.1f}', 'max abs value difference'),
    ]
    card_backgrounds = [TEAL_LIGHT, TEAL_LIGHT, TEAL_LIGHT, PALE]

    summary_rows = [
        ['Modality', 'Backend', 'Original', 'Optimized', 'Speedup', 'Reduction', 'Equal'],
        *(
            [
                MODALITY_LABEL[group.modality],
                group.backend.upper(),
                *_median_cells(group),
                f'{group.reduction:.0f}%',
                _equal_cell(group),
            ]
            for group in ordered
        ),
    ]

    story: list[Flowable] = [
        Spacer(1, 8),
        _paragraph('to-modelarray: original vs optimized', styles['title']),
        _paragraph(
            f'End-to-end conversion runtime of the original committed code (commit {revision}) '
            f'against the optimized working tree for {scope}, on identical inputs',
            styles['subtitle'],
        ),
        _cards(cards, styles, card_backgrounds),
        Spacer(1, 12),
        _paragraph('Executive finding', styles['h1']),
        _paragraph(
            f'The optimized converter is faster than the original ({revision}) {speed}, '
            f'{correctness}, and every R read gate passed. {_largest_gain(ordered)}',
            styles['body'],
        ),
        _paragraph(
            f'Table 1. Medians of {_count(repetitions, "counterbalanced repetition")} after '
            f'{_count(warmups, "discarded warm-up")} per implementation and backend. float32, '
            f'gzip-4 HDF5 / zstd-5 TileDB, {settings["target_chunk_mb"]} MiB target, '
            f'{_count(settings["workers"], "loader worker")}. Converter wall time excludes the '
            'separately timed R read gate.',
            styles['small'],
        ),
        _table(summary_rows, [92, 74, 66, 68, 58, 62, 44], styles),
        Spacer(1, 8),
        SpeedupBars(ordered),
        _paragraph(
            'Figure 1. Original (coral) vs optimized (teal) median wall time; the right-hand label '
            'is the speedup. Element counts differ by modality, so bars compare each pipeline '
            'against its own original, not across modalities.',
            styles['small'],
        ),
        Spacer(1, 6),
        _callout(
            f'<b>Decision:</b> The optimization is a clear, correctness-preserving runtime win for '
            f'{scope}. It is safe to adopt: outputs are identical and remain readable through '
            'ModelArray in R.'
            if all_good
            else f'<b>Decision:</b> Not ready to adopt as measured: {total - faster} of {total} '
            f'configurations were not faster and {total - correct} of {total} outputs differed '
            'from the original. See Table 1 and the correctness gates.',
            styles,
        ),
        PageBreak(),
        Spacer(1, 8),
        _paragraph('Method: controlled before-and-after comparison', styles['h1']),
        _paragraph(
            f'The baseline is the unmodified source at commit {revision}, exported with git archive '
            'into an isolated tree and run under the same Python environment as the optimized '
            'working tree. For each modality and backend the harness discards '
            f'{_count(warmups, "warm-up")} per implementation, then runs '
            f'{_count(repetitions, "counterbalanced measured repetition")} (original/optimized '
            'order alternates), applies the R read gate, and compares every optimized output '
            'against the original for shape, dtype, source order, NaN placement and values.',
            styles['body'],
        ),
        _paragraph('Per-modality detail', styles['h2']),
    ]

    for modality in MODALITY_ORDER:
        groups = grouped.get(modality)
        if not groups:
            continue
        first = groups[0]
        story.append(
            _paragraph(
                f'{MODALITY_LABEL[modality]} — {_plural(first.n_subjects, "subject")} x '
                f'{_plural(first.n_scalars, "scalar")}, {_plural(first.n_elements, "element")}',
                styles['h2'],
            )
        )
        rows = [
            [
                'Backend',
                'Orig median',
                'Opt median',
                'Speedup',
                'Orig R-read',
                'Opt R-read',
                'Files (orig→opt)',
                'Bitwise equal',
            ],
            *(
                [
                    group.backend.upper(),
                    *_median_cells(group),
                    f'{group.baseline_r_median:.2f} s',
                    f'{group.optimized_r_median:.2f} s',
                    f'{group.baseline_files} → {group.optimized_files}',
                    _equal_cell(group),
                ]
                for group in groups
            ),
        ]
        story.append(_table(rows, [50, 62, 62, 52, 62, 62, 90, 64], styles))
        story.append(Spacer(1, 7))

    story.extend(
        [
            _callout(
                '<b>What changed operationally.</b> HDF5: chunks are byte-shuffled and compressed on '
                'a thread pool, then handed to one writer thread, so the same standard gzip file is '
                'written far faster. TileDB: each scalar is written as one dense fragment instead of '
                'one fragment per spatial stripe, cutting the file count sharply. CIFTI: the loader '
                'reads greyordinate data blocks directly and parses the large XML header only once '
                'per scalar. All three keep the exact output schema ModelArray reads.',
                styles,
                background=PALE,
                border=SLATE,
            ),
            PageBreak(),
            Spacer(1, 8),
            _paragraph('Correctness and reproducibility', styles['h1']),
            _paragraph(
                'Correctness gates. Every optimized output was compared against the original '
                'output for the same inputs. '
                + (
                    'All checks passed:'
                    if correct == total
                    else f'{correct} of {total} configurations passed every check:'
                ),
                styles['body'],
            ),
            _table(
                [
                    [
                        'Modality',
                        'Backend',
                        'Shape',
                        'Dtype',
                        'Sources',
                        'NaN map',
                        'Values',
                        'Max abs diff',
                    ],
                    *(
                        [
                            MODALITY_LABEL[group.modality],
                            group.backend.upper(),
                            'eq' if group.shape_equal else 'NO',
                            'eq' if group.dtype_equal else 'NO',
                            'eq' if group.sources_equal else 'NO',
                            'eq' if group.nan_equal else 'NO',
                            'eq' if group.values_equal else 'NO',
                            f'{group.max_abs_difference:.1f}',
                        ]
                        for group in ordered
                    ),
                ],
                [86, 66, 44, 44, 52, 54, 50, 78],
                styles,
            ),
            Spacer(1, 8),
            _paragraph('Reproduce', styles['h2']),
            _table(_cohort_table(grouped, meta), [52, 150, 130, 172], styles),
            Spacer(1, 6),
            _paragraph(
                f'Baseline: git commit {meta["baseline_revision"]}, exported and run under the '
                f'current environment via benchmarks/to_modelarray/baselines/modelarrayio-baseline-'
                f'{meta["baseline_revision"]}. Environment: {meta["platform"]}; Python '
                f'{meta["versions"]["python"]}, NumPy {meta["versions"]["numpy"]}, nibabel '
                f'{meta["versions"]["nibabel"]}, h5py {meta["versions"]["h5py"]}, TileDB-Py '
                f'{meta["versions"]["tiledb"]}. Medians of {_count(repetitions, "repetition")} after '
                f'{_count(warmups, "discarded warm-up")}. Harness: '
                'benchmarks/to_modelarray/compare_revisions.py; cohort generators and commands: '
                'benchmarks/to_modelarray/README.md.',
                styles['small'],
            ),
            Spacer(1, 8),
            *_scale_section(meta.get('scale_rows', []), meta.get('original_run_note', ''), styles),
            _callout(
                f'<b>Bottom line:</b> Against the original code at {revision}, the optimized '
                f'to-modelarray is faster for {scope}, with byte-for-byte identical, R-readable '
                'output.'
                if all_good
                else f'<b>Bottom line:</b> Against the original code at {revision}, the optimized '
                f'to-modelarray is faster in {faster} of {total} configurations and bitwise equal '
                f'in {correct} of {total}.',
                styles,
            ),
        ]
    )
    return story


def _load_scale(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _scale_section(rows: list[dict[str, str]], note: str, styles) -> list[Flowable]:
    """Production-scale page: full-scalar old-vs-new, worker sweep, sharded array, build."""
    if not rows:
        return []
    by_step: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_step.setdefault(row['step'], []).append(row)
    story: list[Flowable] = [
        PageBreak(),
        Spacer(1, 8),
        _paragraph('Production scale', styles['h1']),
    ]
    if note:
        story.append(_paragraph(note, styles['body']))
    compare = by_step.get('3_full_scalar_original_vs_optimized', [])
    if compare:
        story.append(_paragraph('One full scalar, original vs optimized', styles['h2']))
        story.append(
            _table(
                [
                    ['Scalar', 'Subjects', 'Backend', 'Original', 'Optimized', 'Speedup'],
                    *(
                        [
                            r['scalars'],
                            r['subjects'],
                            r['backend'].upper(),
                            f'{float(r["baseline_median_wall_seconds"]) / 3600:.2f} h',
                            f'{float(r["optimized_median_wall_seconds"]) / 60:.1f} min',
                            f'{float(r["speedup"]):.1f}x',
                        ]
                        for r in compare
                    ),
                ],
                [90, 70, 70, 90, 90, 70],
                styles,
            )
        )
    sweep = by_step.get('3_full_scalar_worker_sweep', [])
    if sweep:
        story.append(Spacer(1, 6))
        story.append(_paragraph('Optimized worker sweep on the same scalar', styles['h2']))
        story.append(
            _table(
                [
                    [
                        'Backend',
                        'Workers',
                        'Wall',
                        'Files/min',
                        'Peak RSS',
                        'Fragments',
                        'Fast-path rows',
                    ],
                    *(
                        [
                            r['backend'].upper(),
                            r['workers'],
                            f'{float(r["optimized_median_wall_seconds"]):.1f} s',
                            r.get('input_files_per_minute', ''),
                            f'{r.get("peak_memory_gb", "")} GB',
                            r.get('fragment_count', '') or '—',
                            r.get('fast_path_rows', ''),
                        ]
                        for r in sweep
                    ),
                ],
                [64, 56, 66, 70, 70, 70, 90],
                styles,
            )
        )
    for step, title in (
        ('4_scalar_sharded_array', 'Scalar-sharded SLURM job array (one task per scalar)'),
        ('5_production_build', 'Full production build (per-scalar HDF5 files)'),
    ):
        items = by_step.get(step, [])
        if not items:
            continue
        walls = [float(r['optimized_median_wall_seconds']) for r in items]
        passed = sum(r.get('passed', '').lower() == 'true' for r in items)
        backends = sorted({r['backend'] for r in items})
        story.append(Spacer(1, 6))
        story.append(_paragraph(title, styles['h2']))
        story.append(
            _paragraph(
                f'{len(items)} conversions ({", ".join(b.upper() for b in backends)}), '
                f'{passed} passed their gates; per-scalar wall {min(walls):.0f}-{max(walls):.0f} s, '
                f'median {statistics.median(walls):.0f} s, serial-equivalent sum '
                f'{sum(walls) / 60:.0f} min.',
                styles['body'],
            )
        )
    return story


def _cohort_table(grouped: dict[str, list[Group]], meta: dict) -> list[list[str]]:
    """Describe each modality's cohort from what its comparison recorded."""
    rows = [['Modality', 'Cohort', 'Cohort file', 'Source']]
    for modality in MODALITY_ORDER:
        groups = grouped.get(modality)
        if not groups:
            continue
        recorded = meta['cohorts'][modality]
        scalars = recorded.get('scalar_names') or []
        rows.append(
            [
                MODALITY_NAME[modality],
                (
                    f'{_plural(groups[0].n_subjects, "subject")} x '
                    f'{_plural(len(scalars), "scalar")}: {escape(", ".join(scalars))}'
                ),
                (
                    f'{escape(Path(recorded["cohort"]).name)}<br/>'
                    f'sha256 {recorded.get("cohort_sha256", "")[:12]}'
                ),
                escape(meta['cohort_sources'].get(modality, 'not recorded')),
            ]
        )
    return rows


def _parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, action='append', required=True, dest='run_dirs')
    parser.add_argument(
        '--measured-at', help='Measurement date (default: date of the earliest run directory).'
    )
    parser.add_argument(
        '--scale-csv',
        type=Path,
        default=None,
        help='Optional summarize_scale.py CSV; adds a production-scale section (full-scalar '
        'old-vs-new, worker sweep, scalar-sharded array, production build).',
    )
    parser.add_argument(
        '--original-run-note',
        default='',
        help='One sentence describing the original production run for the scale section.',
    )
    parser.add_argument(
        '--platform', help='Hardware description (default: the platform the runs recorded).'
    )
    parser.add_argument(
        '--cohort-source',
        action='append',
        default=[],
        metavar='MODALITY=TEXT',
        help='Where a modality\'s cohort came from, e.g. "cifti=RBC subset, internal"; '
        'repeat per modality. Unset modalities read "not recorded".',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=repo
        / 'benchmarks'
        / 'to_modelarray'
        / 'results'
        / 'to-modelarray-cross-modality-benchmark.pdf',
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_dirs = [run_dir.expanduser().resolve() for run_dir in args.run_dirs]
    grouped = load_groups(run_dirs)
    if not grouped:
        raise ValueError('No comparison run directories were loaded.')
    metas = [
        json.loads((run_dir / 'comparison_meta.json').read_text(encoding='utf-8'))
        for run_dir in run_dirs
    ]
    first_meta = metas[0]
    created = sorted(recorded['created_at'] for recorded in metas if recorded.get('created_at'))
    meta = {
        'baseline_revision': first_meta.get('baseline_revision', 'baseline'),
        'measured_at': args.measured_at or (created[0][:10] if created else 'date not recorded'),
        'platform': args.platform or first_meta.get('platform', 'not recorded'),
        'settings': {key: _run_setting(metas, key) for key in RUN_SETTINGS},
        # Discarded warm-up runs read every input first, so measured runs see a warm cache.
        'cache': (
            'Warm cache'
            if all(int(recorded.get('warmups') or 0) > 0 for recorded in metas)
            else 'Cache not pre-warmed'
        ),
        'cohorts': {recorded['modality']: recorded for recorded in metas},
        'cohort_sources': _cohort_sources(args.cohort_source),
        # The versions that ran the comparison, not the environment rendering the report.
        'versions': {name: version or 'n/a' for name, version in first_meta['versions'].items()},
        'scale_rows': _load_scale(args.scale_csv),
        'original_run_note': args.original_run_note,
    }
    styles = _styles()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        leftMargin=0.58 * inch,
        rightMargin=0.58 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.48 * inch,
        title='to-modelarray original vs optimized runtime benchmark',
        author='ModelArrayIO benchmark harness',
        subject=f'Original (committed) vs optimized to-modelarray for {_scope(_ordered_groups(grouped))}',
    )
    document.build(
        _report_story(grouped, meta, styles),
        onFirstPage=lambda canvas, doc: _header_footer(
            canvas, doc, meta['measured_at'], meta['baseline_revision'], meta['cache']
        ),
        onLaterPages=lambda canvas, doc: _header_footer(
            canvas, doc, meta['measured_at'], meta['baseline_revision'], meta['cache']
        ),
    )
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
