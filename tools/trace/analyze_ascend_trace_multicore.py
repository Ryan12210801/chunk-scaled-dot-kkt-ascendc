#!/usr/bin/env python3
"""
Multi-core Ascend instruction-trace analyzer.

Designed for msOpProf / CAModel Chrome Trace JSON files containing lanes such as:
    core0.cubecore0
    core0.veccore0
    core0.veccore1

It distinguishes:
  * logical AI cores, AIC lanes, and AIV lanes
  * useful Cube work (MMAD) from Scalar/MTE/control instructions on cubecore
  * active and inactive AIV lanes
  * WAIT_FLAG_DEV / WAIT_FLAG_DEVI and B/E WAIT_FLAG intervals
  * data-transfer bytes, source duplication, and possible over-fetch
  * per-AIV critical-path stages
  * instruction and PC repetition

Usage:
    python3 analyze_ascend_trace_multicore.py "trace(2).json"
    python3 analyze_ascend_trace_multicore.py old.json new.json -o analysis
    python3 analyze_ascend_trace_multicore.py trace.json --top 40

Outputs for each trace:
    report.md
    global_summary.json
    process_summary.csv
    core_summary.csv
    instruction_summary.csv
    transfer_summary.csv
    transfer_source_summary.csv
    wait_summary.csv
    aiv_stage_summary.csv
    pc_summary.csv

When multiple traces are supplied:
    comparison.csv
    comparison.md

Time convention:
Chrome Trace numeric ts/dur values are interpreted as microseconds. The root
"displayTimeUnit": "ns" is a viewer display preference, not the storage unit.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PID_RE = re.compile(r"^core(\d+)\.(cubecore|veccore)(\d+)$")
HEX_OR_INT_RE = r"(0x[0-9a-fA-F]+|-?\d+)"

VECTOR_MATH_NAMES = {
    "VADDS", "VADD", "VSUB", "VSUBS", "VMUL", "VMULS", "VDIV", "VDIVS",
    "VEXP", "VLN", "VABS", "VMAX", "VMIN", "VMAXS", "VMINS",
    "VCMP", "VCMPVS", "VSEL", "VREDUCE", "VCONV", "VREC", "VRSQRT",
}
VECTOR_CONTROL_NAMES = {
    "BAR", "MOVEMASK", "VMOVMASK_XN", "MOVEV",
}
TRANSFER_PREFIXES = ("MOV_", "LOAD_", "FIX_")
TRANSFER_NAMES = {
    "SET_2D",
}
WAIT_X_NAMES = {"WAIT_FLAG_DEV", "WAIT_FLAG_DEVI"}


@dataclass(frozen=True)
class PidInfo:
    core: int | None
    engine: str
    lane: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze multi-core Ascend instruction-level Chrome Trace JSON."
    )
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("ascend_trace_analysis"),
        help="Output root directory."
    )
    parser.add_argument(
        "--top", type=int, default=30,
        help="Number of instructions/PCs shown in the Markdown report."
    )
    return parser.parse_args()


def parse_pid(pid: Any) -> PidInfo:
    text = str(pid or "")
    match = PID_RE.match(text)
    if not match:
        return PidInfo(None, "other", None)
    core = int(match.group(1))
    raw_engine = match.group(2)
    lane = int(match.group(3))
    engine = "AIC" if raw_engine == "cubecore" else "AIV"
    return PidInfo(core, engine, lane)


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def integer(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, str):
            return int(value, 0)
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_detail(detail: Any) -> dict[str, str]:
    text = str(detail or "")
    result: dict[str, str] = {}
    for item in text.split(","):
        item = item.strip()
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        result[key.strip()] = value.strip()
    return result


def extract_register_value(detail: Any, reg: str) -> str:
    text = str(detail or "")
    match = re.search(rf"\b{re.escape(reg)}:[^=,]+={HEX_OR_INT_RE}", text)
    return match.group(1) if match else ""


def csv_write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend(
        "| " + " | ".join(clean(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(output)


def pair_begin_end(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    opened: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    result: list[dict[str, Any]] = []

    for event in events:
        phase = event.get("ph")
        if phase not in {"B", "E"}:
            continue
        key = (
            event.get("pid"), event.get("tid"), event.get("name"), event.get("id")
        )
        if phase == "B":
            opened[key].append(event)
            continue
        if not opened[key]:
            continue
        begin = opened[key].pop(0)
        start = number(begin.get("ts"))
        end = number(event.get("ts"))
        result.append({
            "pid": begin.get("pid"),
            "tid": begin.get("tid"),
            "name": begin.get("name"),
            "event_id": begin.get("id"),
            "start_us": start,
            "end_us": end,
            "duration_us": max(0.0, end - start),
            "detail": (begin.get("args") or {}).get("detail", ""),
            "pc_addr": (begin.get("args") or {}).get("pc_addr", ""),
            "source_phase": "B/E",
        })
    return result


def is_transfer(event: dict[str, Any]) -> bool:
    name = str(event.get("name") or "")
    tid = str(event.get("tid") or "")
    args = event.get("args") or {}

    if integer(args.get("process_bytes")) > 0:
        return True

    # Keep transfer/configuration instructions only when they are emitted on
    # data-movement pipelines. This avoids misclassifying scalar MOV_* register
    # instructions as memory transfers.
    if tid not in {"MTE1", "MTE2", "MTE3", "FIXP"}:
        return False
    if name in {"MOV_SPR_XN", "SET_CROSS_CORE", "BAR"}:
        return False
    return name.startswith(TRANSFER_PREFIXES) or name in TRANSFER_NAMES


def process_signature(event_group: list[dict[str, Any]]) -> str:
    vector = Counter(
        str(e.get("name"))
        for e in event_group
        if e.get("tid") == "VECTOR"
    )
    if not vector:
        return ""
    return ";".join(f"{name}:{count}" for name, count in sorted(vector.items()))


def classify_process(
    pid_info: PidInfo,
    events: list[dict[str, Any]],
) -> str:
    names = Counter(str(e.get("name")) for e in events)
    if pid_info.engine == "AIC":
        if names["MMAD"] > 0:
            return "useful_aic"
        return "control_only_aic"
    if pid_info.engine == "AIV":
        if any(names[name] > 0 for name in VECTOR_MATH_NAMES):
            return "active_aiv"
        return "inactive_aiv"
    return "other"


def infer_aiv_stages(
    pid: str,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(events, key=lambda e: (number(e.get("ts")), number(e.get("dur"))))
    starts = [number(e.get("ts")) for e in ordered]
    ends = [number(e.get("ts")) + number(e.get("dur")) for e in ordered]
    start = min(starts) if starts else 0.0
    end = max(ends) if ends else 0.0

    waits = [e for e in ordered if e.get("name") in WAIT_X_NAMES]
    main_wait = max(waits, key=lambda e: number(e.get("dur")), default=None)
    wait_start = number(main_wait.get("ts")) if main_wait else 0.0
    wait_duration = number(main_wait.get("dur")) if main_wait else 0.0
    wait_end = wait_start + wait_duration if main_wait else 0.0

    post_wait_transfers = [
        e for e in ordered
        if e.get("tid") in {"MTE1", "MTE2", "MTE3", "FIXP"}
        and number(e.get("ts")) >= wait_end
        and integer((e.get("args") or {}).get("process_bytes")) > 0
    ]
    first_copyin = min(
        (number(e.get("ts")) for e in post_wait_transfers if e.get("tid") == "MTE2"),
        default=0.0,
    )

    math_events = [
        e for e in ordered
        if e.get("tid") == "VECTOR" and str(e.get("name")) in VECTOR_MATH_NAMES
    ]
    math_start = min((number(e.get("ts")) for e in math_events), default=0.0)
    math_end = max(
        (number(e.get("ts")) + number(e.get("dur")) for e in math_events),
        default=0.0,
    )

    copyouts = [
        e for e in ordered
        if e.get("tid") == "MTE3"
        and integer((e.get("args") or {}).get("process_bytes")) > 0
        and number(e.get("ts")) >= math_start
    ]
    copyout_start = min((number(e.get("ts")) for e in copyouts), default=0.0)
    copyout_end = max(
        (number(e.get("ts")) + number(e.get("dur")) for e in copyouts),
        default=0.0,
    )

    return {
        "pid": pid,
        "start_us": round(start, 6),
        "wait_start_us": round(wait_start, 6),
        "wait_duration_us": round(wait_duration, 6),
        "wait_end_us": round(wait_end, 6),
        "first_copyin_us": round(first_copyin, 6),
        "vector_math_start_us": round(math_start, 6),
        "vector_math_end_us": round(math_end, 6),
        "vector_math_window_us": round(max(0.0, math_end - math_start), 6),
        "copyout_start_us": round(copyout_start, 6),
        "copyout_end_us": round(copyout_end, 6),
        "end_us": round(end, 6),
        "init_before_wait_us": round(max(0.0, wait_start - start), 6) if main_wait else "",
        "post_wait_to_math_us": round(max(0.0, math_start - wait_end), 6)
        if main_wait and math_start else "",
        "tail_after_math_us": round(max(0.0, end - math_end), 6)
        if math_end else "",
    }


def analyze_trace(trace_path: Path, output_dir: Path, top_n: int) -> dict[str, Any]:
    with trace_path.open("r", encoding="utf-8") as handle:
        root = json.load(handle)

    events = root.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError(f"{trace_path}: traceEvents is not a list")

    x_events = [e for e in events if e.get("ph") == "X"]
    be_intervals = pair_begin_end(events)
    phase_counts = Counter(str(e.get("ph")) for e in events)

    if x_events:
        global_start = min(number(e.get("ts")) for e in x_events)
        global_end = max(
            number(e.get("ts")) + number(e.get("dur")) for e in x_events
        )
    else:
        global_start = global_end = 0.0
    global_span = max(0.0, global_end - global_start)

    by_pid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in x_events:
        by_pid[str(event.get("pid"))].append(event)

    process_rows: list[dict[str, Any]] = []
    aiv_stage_rows: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()

    for pid, group in sorted(
        by_pid.items(),
        key=lambda item: (
            parse_pid(item[0]).core
            if parse_pid(item[0]).core is not None else 10**9,
            parse_pid(item[0]).engine,
            parse_pid(item[0]).lane
            if parse_pid(item[0]).lane is not None else 10**9,
        ),
    ):
        info = parse_pid(pid)
        starts = [number(e.get("ts")) for e in group]
        ends = [number(e.get("ts")) + number(e.get("dur")) for e in group]
        name_counts = Counter(str(e.get("name")) for e in group)
        role = classify_process(info, group)
        role_counts[role] += 1

        wait_dev_us = sum(
            number(e.get("dur")) for e in group if e.get("name") in WAIT_X_NAMES
        )
        transfer_bytes = sum(
            integer((e.get("args") or {}).get("process_bytes"))
            for e in group
        )
        vector_math_count = sum(name_counts[name] for name in VECTOR_MATH_NAMES)
        vector_control_count = sum(name_counts[name] for name in VECTOR_CONTROL_NAMES)

        process_rows.append({
            "pid": pid,
            "logical_core": info.core if info.core is not None else "",
            "engine": info.engine,
            "lane": info.lane if info.lane is not None else "",
            "role": role,
            "event_count": len(group),
            "start_us": round(min(starts), 6),
            "end_us": round(max(ends), 6),
            "span_us": round(max(ends) - min(starts), 6),
            "wait_dev_us": round(wait_dev_us, 6),
            "cube_mmad_count": name_counts["MMAD"],
            "vector_math_count": vector_math_count,
            "vector_control_count": vector_control_count,
            "transfer_bytes": transfer_bytes,
            "vector_signature": process_signature(group),
        })

        if role == "active_aiv":
            stage = infer_aiv_stages(pid, group)
            stage.update({
                "logical_core": info.core,
                "lane": info.lane,
            })
            aiv_stage_rows.append(stage)

    # Logical-core summary.
    core_rows: list[dict[str, Any]] = []
    logical_cores = sorted({
        parse_pid(pid).core for pid in by_pid
        if parse_pid(pid).core is not None
    })
    process_by_core: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in process_rows:
        if row["logical_core"] != "":
            process_by_core[int(row["logical_core"])].append(row)

    for core in logical_cores:
        rows = process_by_core[core]
        core_rows.append({
            "logical_core": core,
            "aic_count": sum(row["engine"] == "AIC" for row in rows),
            "aiv_count": sum(row["engine"] == "AIV" for row in rows),
            "useful_aic_count": sum(row["role"] == "useful_aic" for row in rows),
            "active_aiv_count": sum(row["role"] == "active_aiv" for row in rows),
            "inactive_aiv_count": sum(row["role"] == "inactive_aiv" for row in rows),
            "start_us": min(row["start_us"] for row in rows),
            "end_us": max(row["end_us"] for row in rows),
            "span_us": round(
                max(row["end_us"] for row in rows)
                - min(row["start_us"] for row in rows),
                6,
            ),
            "events": sum(int(row["event_count"]) for row in rows),
            "transfer_bytes": sum(int(row["transfer_bytes"]) for row in rows),
        })

    # Instruction summary.
    instruction_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in x_events:
        info = parse_pid(event.get("pid"))
        key = (info.engine, str(event.get("tid")), str(event.get("name")))
        instruction_groups[key].append(event)

    instruction_rows: list[dict[str, Any]] = []
    for (engine, tid, name), group in instruction_groups.items():
        durations = [number(e.get("dur")) for e in group]
        total_bytes = sum(
            integer((e.get("args") or {}).get("process_bytes")) for e in group
        )
        instruction_rows.append({
            "engine": engine,
            "tid": tid,
            "instruction": name,
            "count": len(group),
            "sum_duration_us": round(sum(durations), 6),
            "avg_duration_ns": round(statistics.mean(durations) * 1000.0, 3),
            "max_duration_ns": round(max(durations, default=0.0) * 1000.0, 3),
            "process_bytes": total_bytes,
        })
    instruction_rows.sort(
        key=lambda row: (-float(row["sum_duration_us"]), -int(row["count"]))
    )

    # Transfer event and source summaries.
    transfer_event_rows: list[dict[str, Any]] = []
    source_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)

    for event in x_events:
        if not is_transfer(event):
            continue
        args = event.get("args") or {}
        detail = str(args.get("detail", ""))
        parsed_detail = parse_detail(detail)
        info = parse_pid(event.get("pid"))
        process_bytes = integer(args.get("process_bytes"))
        xd = extract_register_value(detail, "XD")
        xn = extract_register_value(detail, "XN")
        xm = extract_register_value(detail, "XM")
        row = {
            "pid": event.get("pid"),
            "logical_core": info.core if info.core is not None else "",
            "engine": info.engine,
            "lane": info.lane if info.lane is not None else "",
            "tid": event.get("tid"),
            "instruction": event.get("name"),
            "src": parsed_detail.get("Src", ""),
            "dst": parsed_detail.get("Dst", ""),
            "process_bytes": process_bytes,
            "start_us": round(number(event.get("ts")), 6),
            "duration_us": round(number(event.get("dur")), 6),
            "xd_value": xd,
            "xn_value": xn,
            "xm_value": xm,
            "pc_addr": args.get("pc_addr", ""),
        }
        transfer_event_rows.append(row)

        # For OUT -> local-memory transfers, XN commonly contains the OUT source address.
        # Keep raw registers in the key rather than claiming this is universally true.
        source_key = (
            info.engine,
            str(event.get("name")),
            parsed_detail.get("Src", ""),
            parsed_detail.get("Dst", ""),
            process_bytes,
            xn,
        )
        source_groups[source_key].append(event)

    transfer_source_rows: list[dict[str, Any]] = []
    for key, group in source_groups.items():
        engine, name, src, dst, process_bytes, xn = key
        pids = {str(e.get("pid")) for e in group}
        durations = [number(e.get("dur")) for e in group]
        transfer_source_rows.append({
            "engine": engine,
            "instruction": name,
            "src": src,
            "dst": dst,
            "process_bytes_each": process_bytes,
            "xn_value": xn,
            "event_count": len(group),
            "unique_pids": len(pids),
            "total_bytes": process_bytes * len(group),
            "avg_duration_us": round(statistics.mean(durations), 6),
            "max_duration_us": round(max(durations, default=0.0), 6),
        })
    transfer_source_rows.sort(
        key=lambda row: (-int(row["total_bytes"]), -int(row["event_count"]))
    )

    # Wait summary: complete WAIT_FLAG_DEV/DEVI and B/E intervals.
    wait_rows: list[dict[str, Any]] = []
    for event in x_events:
        if event.get("name") not in WAIT_X_NAMES:
            continue
        info = parse_pid(event.get("pid"))
        wait_rows.append({
            "pid": event.get("pid"),
            "logical_core": info.core if info.core is not None else "",
            "engine": info.engine,
            "lane": info.lane if info.lane is not None else "",
            "tid": event.get("tid"),
            "wait_name": event.get("name"),
            "start_us": round(number(event.get("ts")), 6),
            "duration_us": round(number(event.get("dur")), 6),
            "detail": (event.get("args") or {}).get("detail", ""),
            "pc_addr": (event.get("args") or {}).get("pc_addr", ""),
            "source_phase": "X",
        })
    for interval in be_intervals:
        if interval["name"] != "WAIT_FLAG":
            continue
        info = parse_pid(interval["pid"])
        wait_rows.append({
            "pid": interval["pid"],
            "logical_core": info.core if info.core is not None else "",
            "engine": info.engine,
            "lane": info.lane if info.lane is not None else "",
            "tid": interval["tid"],
            "wait_name": interval["name"],
            "start_us": round(interval["start_us"], 6),
            "duration_us": round(interval["duration_us"], 6),
            "detail": interval["detail"],
            "pc_addr": interval["pc_addr"],
            "source_phase": "B/E",
        })
    wait_rows.sort(key=lambda row: -float(row["duration_us"]))

    # PC repetition.
    pc_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in x_events:
        args = event.get("args") or {}
        pc = str(args.get("pc_addr", ""))
        if not pc:
            continue
        key = (
            str(event.get("pid")), str(event.get("tid")),
            str(event.get("name")), pc,
        )
        pc_groups[key].append(event)

    pc_rows: list[dict[str, Any]] = []
    for (pid, tid, name, pc), group in pc_groups.items():
        durations = [number(e.get("dur")) for e in group]
        pc_rows.append({
            "pid": pid,
            "tid": tid,
            "instruction": name,
            "pc_addr": pc,
            "count": len(group),
            "sum_duration_us": round(sum(durations), 6),
            "avg_duration_ns": round(statistics.mean(durations) * 1000.0, 3),
            "first_us": round(min(number(e.get("ts")) for e in group), 6),
            "last_us": round(
                max(number(e.get("ts")) + number(e.get("dur")) for e in group),
                6,
            ),
        })
    pc_rows.sort(key=lambda row: (-int(row["count"]), -float(row["sum_duration_us"])))

    active_aiv_rows = [row for row in process_rows if row["role"] == "active_aiv"]
    inactive_aiv_rows = [row for row in process_rows if row["role"] == "inactive_aiv"]
    useful_aic_rows = [row for row in process_rows if row["role"] == "useful_aic"]
    control_aic_rows = [row for row in process_rows if row["role"] == "control_only_aic"]

    total_mmad = sum(
        1 for event in x_events if event.get("name") == "MMAD"
    )
    total_transfer_bytes = sum(
        integer((event.get("args") or {}).get("process_bytes"))
        for event in x_events
    )
    total_wait_dev = sum(
        number(event.get("dur"))
        for event in x_events if event.get("name") in WAIT_X_NAMES
    )

    summary = {
        "trace": str(trace_path),
        "displayTimeUnit": root.get("displayTimeUnit"),
        "profilingType": root.get("profilingType"),
        "schemaVersion": root.get("schemaVersion"),
        "total_events": len(events),
        "complete_events": len(x_events),
        "phase_counts": dict(phase_counts),
        "global_start_us": round(global_start, 6),
        "global_end_us": round(global_end, 6),
        "global_span_us": round(global_span, 6),
        "logical_core_count": len(logical_cores),
        "process_count": len(by_pid),
        "aic_process_count": sum(row["engine"] == "AIC" for row in process_rows),
        "aiv_process_count": sum(row["engine"] == "AIV" for row in process_rows),
        "useful_aic_count": len(useful_aic_rows),
        "control_only_aic_count": len(control_aic_rows),
        "active_aiv_count": len(active_aiv_rows),
        "inactive_aiv_count": len(inactive_aiv_rows),
        "mmad_count": total_mmad,
        "total_transfer_bytes": total_transfer_bytes,
        "total_wait_dev_us_across_processes": round(total_wait_dev, 6),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "global_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_write(
        output_dir / "process_summary.csv",
        process_rows,
        [
            "pid", "logical_core", "engine", "lane", "role", "event_count",
            "start_us", "end_us", "span_us", "wait_dev_us",
            "cube_mmad_count", "vector_math_count", "vector_control_count",
            "transfer_bytes", "vector_signature",
        ],
    )
    csv_write(
        output_dir / "core_summary.csv",
        core_rows,
        [
            "logical_core", "aic_count", "aiv_count", "useful_aic_count",
            "active_aiv_count", "inactive_aiv_count", "start_us", "end_us",
            "span_us", "events", "transfer_bytes",
        ],
    )
    csv_write(
        output_dir / "instruction_summary.csv",
        instruction_rows,
        [
            "engine", "tid", "instruction", "count", "sum_duration_us",
            "avg_duration_ns", "max_duration_ns", "process_bytes",
        ],
    )
    csv_write(
        output_dir / "transfer_summary.csv",
        transfer_event_rows,
        [
            "pid", "logical_core", "engine", "lane", "tid", "instruction",
            "src", "dst", "process_bytes", "start_us", "duration_us",
            "xd_value", "xn_value", "xm_value", "pc_addr",
        ],
    )
    csv_write(
        output_dir / "transfer_source_summary.csv",
        transfer_source_rows,
        [
            "engine", "instruction", "src", "dst", "process_bytes_each",
            "xn_value", "event_count", "unique_pids", "total_bytes",
            "avg_duration_us", "max_duration_us",
        ],
    )
    csv_write(
        output_dir / "wait_summary.csv",
        wait_rows,
        [
            "pid", "logical_core", "engine", "lane", "tid", "wait_name",
            "start_us", "duration_us", "detail", "pc_addr", "source_phase",
        ],
    )
    csv_write(
        output_dir / "aiv_stage_summary.csv",
        aiv_stage_rows,
        [
            "pid", "logical_core", "lane", "start_us", "wait_start_us",
            "wait_duration_us", "wait_end_us", "first_copyin_us",
            "vector_math_start_us", "vector_math_end_us",
            "vector_math_window_us", "copyout_start_us", "copyout_end_us",
            "end_us", "init_before_wait_us", "post_wait_to_math_us",
            "tail_after_math_us",
        ],
    )
    csv_write(
        output_dir / "pc_summary.csv",
        pc_rows,
        [
            "pid", "tid", "instruction", "pc_addr", "count",
            "sum_duration_us", "avg_duration_ns", "first_us", "last_us",
        ],
    )

    # Report.
    report: list[str] = []
    report.append(f"# Ascend Trace Report: {trace_path.name}\n")
    report.append("## Global summary\n")
    report.append(md_table(
        ["metric", "value"],
        [
            ["global span", f"{global_span:.6f} µs"],
            ["total events", len(events)],
            ["complete X events", len(x_events)],
            ["logical AI cores", len(logical_cores)],
            ["AIC processes", summary["aic_process_count"]],
            ["AIV processes", summary["aiv_process_count"]],
            ["useful AICs containing MMAD", summary["useful_aic_count"]],
            ["control-only AICs", summary["control_only_aic_count"]],
            ["active AIVs", summary["active_aiv_count"]],
            ["inactive AIVs", summary["inactive_aiv_count"]],
            ["MMAD instructions", total_mmad],
            ["transfer bytes across all lanes", total_transfer_bytes],
        ],
    ))
    report.append(
        "\n`cubecore` identifies the process that runs on the AIC side. "
        "Only instructions on the `CUBE` lane, such as `MMAD`, consume the "
        "Cube matrix engine. Scalar, MTE, FIXP, and FLOWCTRL instructions on "
        "the same process are not Cube arithmetic."
    )
    report.append(
        "\nChrome Trace numeric `ts`/`dur` fields are treated as microseconds. "
        "`displayTimeUnit: ns` only controls viewer presentation."
    )

    report.append("\n## Process roles\n")
    report.append(md_table(
        ["role", "count"],
        [[role, count] for role, count in sorted(role_counts.items())],
    ))

    report.append("\n## Logical-core layout\n")
    report.append(md_table(
        [
            "core", "useful AIC", "active AIV", "inactive AIV",
            "span µs", "transfer bytes",
        ],
        [
            [
                row["logical_core"], row["useful_aic_count"],
                row["active_aiv_count"], row["inactive_aiv_count"],
                row["span_us"], row["transfer_bytes"],
            ]
            for row in core_rows
        ],
    ))

    if aiv_stage_rows:
        wait_values = [float(row["wait_duration_us"]) for row in aiv_stage_rows]
        math_values = [float(row["vector_math_window_us"]) for row in aiv_stage_rows]
        post_values = [
            float(row["post_wait_to_math_us"])
            for row in aiv_stage_rows
            if row["post_wait_to_math_us"] != ""
        ]
        report.append("\n## Active-AIV stage statistics\n")
        report.append(md_table(
            ["stage", "minimum µs", "mean µs", "maximum µs"],
            [
                [
                    "device/cross-core wait",
                    f"{min(wait_values):.6f}",
                    f"{statistics.mean(wait_values):.6f}",
                    f"{max(wait_values):.6f}",
                ],
                [
                    "post-wait to first vector math",
                    f"{min(post_values):.6f}",
                    f"{statistics.mean(post_values):.6f}",
                    f"{max(post_values):.6f}",
                ] if post_values else ["post-wait to first vector math", "", "", ""],
                [
                    "vector math window",
                    f"{min(math_values):.6f}",
                    f"{statistics.mean(math_values):.6f}",
                    f"{max(math_values):.6f}",
                ],
            ],
        ))

    report.append(f"\n## Top {top_n} instructions by summed lifetime\n")
    report.append(md_table(
        [
            "engine", "pipe", "instruction", "count",
            "sum µs", "avg ns", "bytes",
        ],
        [
            [
                row["engine"], row["tid"], row["instruction"], row["count"],
                row["sum_duration_us"], row["avg_duration_ns"],
                row["process_bytes"],
            ]
            for row in instruction_rows[:top_n]
        ],
    ))
    report.append(
        "\nSummed instruction lifetimes can exceed wall-clock span because "
        "different cores and pipelines execute concurrently."
    )

    report.append(f"\n## Top {top_n} repeated transfer sources\n")
    report.append(md_table(
        [
            "engine", "instruction", "src→dst", "bytes/event",
            "XN value", "events", "pids", "total bytes", "avg µs",
        ],
        [
            [
                row["engine"], row["instruction"],
                f'{row["src"]}→{row["dst"]}', row["process_bytes_each"],
                row["xn_value"], row["event_count"], row["unique_pids"],
                row["total_bytes"], row["avg_duration_us"],
            ]
            for row in transfer_source_rows[:top_n]
        ],
    ))
    report.append(
        "\nFor many OUT→UB instructions, `XN` is the observed OUT-side address. "
        "The report preserves raw register values because exact register roles "
        "can vary by instruction."
    )

    report.append(f"\n## Longest {top_n} waits\n")
    report.append(md_table(
        [
            "pid", "pipe", "wait", "start µs",
            "duration µs", "detail",
        ],
        [
            [
                row["pid"], row["tid"], row["wait_name"],
                row["start_us"], row["duration_us"], row["detail"],
            ]
            for row in wait_rows[:top_n]
        ],
    ))

    report.append(f"\n## Top {top_n} repeated PCs\n")
    report.append(md_table(
        ["pid", "pipe", "instruction", "PC", "count", "sum µs"],
        [
            [
                row["pid"], row["tid"], row["instruction"],
                row["pc_addr"], row["count"], row["sum_duration_us"],
            ]
            for row in pc_rows[:top_n]
        ],
    ))

    (output_dir / "report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    return summary


def write_comparison(
    summaries: list[dict[str, Any]],
    output_root: Path,
) -> None:
    if len(summaries) < 2:
        return
    fields = [
        "trace", "global_span_us", "total_events", "complete_events",
        "logical_core_count", "aic_process_count", "aiv_process_count",
        "useful_aic_count", "control_only_aic_count",
        "active_aiv_count", "inactive_aiv_count", "mmad_count",
        "total_transfer_bytes", "total_wait_dev_us_across_processes",
    ]
    csv_write(output_root / "comparison.csv", summaries, fields)

    baseline = summaries[0]
    report = ["# Trace comparison\n"]
    rows = []
    for summary in summaries:
        speedup = (
            number(baseline.get("global_span_us"))
            / number(summary.get("global_span_us"))
            if number(summary.get("global_span_us")) > 0 else 0.0
        )
        rows.append([
            Path(summary["trace"]).name,
            summary["global_span_us"],
            f"{speedup:.3f}×",
            summary["useful_aic_count"],
            summary["active_aiv_count"],
            summary["inactive_aiv_count"],
            summary["total_transfer_bytes"],
        ])
    report.append(md_table(
        [
            "trace", "span µs", "speed vs first",
            "useful AIC", "active AIV", "inactive AIV", "transfer bytes",
        ],
        rows,
    ))
    (output_root / "comparison.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    used_names: Counter[str] = Counter()

    for trace in args.traces:
        base = trace.stem
        used_names[base] += 1
        suffix = f"_{used_names[base]}" if used_names[base] > 1 else ""
        output_dir = args.output / f"{base}{suffix}"
        summary = analyze_trace(trace, output_dir, args.top)
        summaries.append(summary)
        print(
            f"{trace}: span={summary['global_span_us']} us, "
            f"useful AIC={summary['useful_aic_count']}, "
            f"active AIV={summary['active_aiv_count']}, "
            f"inactive AIV={summary['inactive_aiv_count']}"
        )

    write_comparison(summaries, args.output)
    print(f"Results written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
