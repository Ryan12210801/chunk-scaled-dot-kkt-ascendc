#!/usr/bin/env python3

import argparse
import csv
import json
import re
from pathlib import Path


FACTOR_OPS = (
    "VMULS", "MULS",
    "VBRCB", "BRCB",
    "VADD",
    "VCMP",
    "VSEL",
    "VEXP",
    "VCONV",
    "CAST",
)


def parse_process_names(events):
    """Schema 1 uses numeric PID plus process_name metadata."""
    result = {}

    for event in events:
        if (
            event.get("ph") == "M"
            and event.get("name") == "process_name"
        ):
            pid = str(event.get("pid"))
            result[pid] = str(
                event.get("args", {}).get("name", pid)
            )

    return result


def get_process_bytes(event):
    args = event.get("args", {})
    raw = args.get("process_bytes")

    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass

    detail = str(args.get("detail", ""))
    match = re.search(
        r"process_bytes[\"']?\s*[:=]\s*[\"']?(\d+)",
        detail,
    )

    return int(match.group(1)) if match else None


def classify_core(process):
    process = process.upper()

    if "CUBECORE" in process:
        return "AIC"

    if "VECCORE" in process:
        return "AIV"

    return "UNKNOWN"


def classify_event(event, process, pipeline):
    name = str(event.get("name", "")).upper()
    detail = str(
        event.get("args", {}).get("detail", "")
    ).upper()
    code = str(
        event.get("args", {}).get("code", "")
    ).upper()

    core = classify_core(process)
    size = get_process_bytes(event)

    if core == "AIC" and name == "SET_CROSS_CORE":
        return "AIC_READY"

    if core == "AIV" and name == "SET_CROSS_CORE":
        return "AIV_FREE"

    if core == "AIV":
        if size == 512:
            return "G_MTE2"

        if size == 256:
            return "BETA_MTE2"

        if size == 16384:
            return "KKT_MTE2"

        if (
            pipeline.upper() in ("VECTOR", "VEC")
            or name.startswith("V")
        ):
            if any(op in name for op in FACTOR_OPS):
                return "FACTOR_VECTOR"

        if (
            pipeline.upper() == "MTE3"
            or "UB_TO_OUT" in name
        ):
            return "OUTPUT_MTE3"

        if (
            name == "WAIT_FLAG"
            and (
                "CROSS" in detail
                or "CROSSCOREWAITFLAG" in code
            )
        ):
            return "AIV_READY_WAIT"

    if core == "AIC" and (
        "MMAD" in name
        or pipeline.upper() in ("CUBE", "FIXP")
    ):
        return "AIC_COMPUTE"

    return None


def load_trace(path):
    data = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    events = data.get("traceEvents", [])
    process_names = parse_process_names(events)

    rows = []

    for index, event in enumerate(events):
        if event.get("ph") != "X":
            continue

        pid = str(event.get("pid", ""))
        process = process_names.get(pid, pid)

        # Schema 2: process itself is core, TID is pipeline.
        pipeline = str(event.get("tid", ""))

        # Schema 1: PID is pipeline name.
        if process.upper() in {
            "CUBE", "FIXP", "VECTOR",
            "MTE1", "MTE2", "MTE3",
            "SCALAR", "FLOWCTRL",
        }:
            pipeline = process

        kind = classify_event(
            event,
            process,
            pipeline,
        )

        if kind is None:
            continue

        ts = float(event.get("ts", 0.0))
        dur = float(event.get("dur", 0.0))

        rows.append({
            "index": index,
            "ts_us": ts,
            "end_us": ts + dur,
            "dur_us": dur,
            "process": process,
            "pipeline": pipeline,
            "name": event.get("name", ""),
            "kind": kind,
            "bytes": get_process_bytes(event),
            "pc": event.get("args", {}).get(
                "pc_addr", ""
            ),
        })

    rows.sort(key=lambda row: (row["ts_us"], row["index"]))
    return rows


def previous_event(rows, index, kind):
    for row in reversed(rows[:index]):
        if row["kind"] == kind:
            return row
    return None


def analyze(rows):
    print(
        f"{'time/us':>10} "
        f"{'duration':>10} "
        f"{'core':>18} "
        f"{'pipeline':>10} "
        f"{'kind':>16} "
        f"{'bytes':>8}"
    )

    for row in rows:
        print(
            f"{row['ts_us']:10.3f} "
            f"{row['dur_us']:10.3f} "
            f"{row['process']:>18} "
            f"{row['pipeline']:>10} "
            f"{row['kind']:>16} "
            f"{str(row['bytes'] or ''):>8}"
        )

    print("\n=== KKT窗口分析 ===")

    for index, row in enumerate(rows):
        if row["kind"] != "KKT_MTE2":
            continue

        ready = previous_event(
            rows,
            index,
            "AIC_READY",
        )

        g_copy = previous_event(
            rows,
            index,
            "G_MTE2",
        )

        beta_copy = previous_event(
            rows,
            index,
            "BETA_MTE2",
        )

        factor = previous_event(
            rows,
            index,
            "FACTOR_VECTOR",
        )

        print(
            f"\nKKT @ {row['ts_us']:.3f} us"
        )

        if ready:
            print(
                "  last AIC READY: "
                f"{ready['ts_us']:.3f} us, "
                f"delay={row['ts_us'] - ready['ts_us']:.3f} us"
            )

        if g_copy:
            print(
                "  last g MTE2:    "
                f"{g_copy['ts_us']:.3f} us"
            )

        if beta_copy:
            print(
                "  last beta MTE2: "
                f"{beta_copy['ts_us']:.3f} us"
            )

        if factor:
            delta = factor["ts_us"] - row["ts_us"]

            print(
                "  last Factor V:  "
                f"{factor['ts_us']:.3f} us"
            )

            if factor["end_us"] > row["ts_us"]:
                print(
                    "  RESULT: Factor与KKT MTE2存在重叠"
                )
            elif delta < 0:
                print(
                    "  RESULT: Factor在KKT前执行，"
                    "但需要确认是否属于同一Task"
                )
            else:
                print(
                    "  RESULT: Factor没有提前到KKT之前"
                )


def write_csv(rows, path):
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "trace",
        type=Path,
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("trace_timeline.csv"),
    )

    args = parser.parse_args()

    rows = load_trace(args.trace)

    analyze(rows)
    write_csv(rows, args.csv)

    print(f"\nCSV written to: {args.csv}")


if __name__ == "__main__":
    main()