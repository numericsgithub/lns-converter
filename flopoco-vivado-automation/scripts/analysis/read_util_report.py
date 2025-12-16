from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from typing import List, Optional

uid_counter = 0


@dataclass
class UtilReportEntry:
    Instance: Optional[str] = None
    Module: Optional[str] = None
    LUTs: Optional[int] = 0
    SRLs: Optional[int] = 0
    FFs: Optional[int] = 0
    BRAMs: Optional[float] = 0.0
    URAMs: Optional[int] = 0
    DSPs: Optional[int] = 0
    Parent: Optional["UtilReportEntry"] = None
    Children: List["UtilReportEntry"] = field(default_factory=list)
    Depth: Optional[int] = None
    UId: Optional[int] = None

    def get_string(self) -> str:
        return f"{self.Instance} -> LUTs: {self.LUTs} FFs: {self.FFs} DSPs: {self.DSPs} BRAMs: {self.BRAMs}"


class UtilReport:
    def __init__(self, all_entries: List[UtilReportEntry]):
        self.all_entries = all_entries

    def get_top(self) -> UtilReportEntry:
        return self.all_entries[0]

    def get_by_instance_name(self, instance_name: str, depth: Optional[int] = None) -> UtilReportEntry:
        for entry in self.all_entries:
            if entry.Instance == instance_name and (depth is None or depth == entry.Depth):
                return entry
        all_names = [entry.Instance for entry in self.all_entries]
        raise ValueError(
            f"Cannot find util entry with instance name '{instance_name}'. "
            f"Available instances: {all_names}"
        )

    def get_by_instance_name_or_none(self, instance_name: str, depth: Optional[int] = None) -> Optional[UtilReportEntry]:
        try:
            return self.get_by_instance_name(instance_name, depth)
        except ValueError:
            return None


def read_vivado_util_report(report_path: str) -> UtilReport:
    global uid_counter

    def parse_table_row(line: str) -> List[str]:
        splitted = [segment.strip() for segment in line.split("|")]
        return [fragment for fragment in splitted if fragment]

    def parse_table_row_depth(line: str) -> int:
        instance = line.split("|")[1]
        whitespace_at_front = re.sub(r"^(\s*)(.*)$", r"\1", instance)
        number_of_spaces = len(whitespace_at_front)
        return max((number_of_spaces - 1) // 2, 0)

    with open(report_path, "r", encoding="utf-8") as report_file:
        all_lines = report_file.readlines()

    cur_line_index = -1
    for i, line in enumerate(all_lines):
        if "Total LUTs" in line:
            cur_line_index = i
            break
    if cur_line_index == -1:
        raise RuntimeError("Unable to locate utilization table header in report.")

    header_fields = parse_table_row(all_lines[cur_line_index])
    cur_line_index += 2

    entries: List[UtilReportEntry] = []
    for line_idx in range(cur_line_index, len(all_lines)):
        cur_line = all_lines[line_idx]
        if "-+-" in cur_line:
            break
        if cur_line.strip() == "":
            continue

        values = parse_table_row(cur_line)
        record = UtilReportEntry()
        record_map = {key: value for key, value in zip(header_fields, values)}
        record.Instance = record_map.get("Instance")
        record.Module = record_map.get("Module")
        if "Total LUTs" in record_map:
            record.LUTs = int(record_map["Total LUTs"])
        if "SRLs" in record_map:
            record.SRLs = int(record_map["SRLs"])
        if "FFs" in record_map:
            record.FFs = int(record_map["FFs"])
        ramb36 = float(record_map.get("RAMB36", 0))
        ramb18 = float(record_map.get("RAMB18", 0))
        record.BRAMs = ramb36 + ramb18 * 0.5
        if "URAM" in record_map:
            record.URAMs = int(record_map["URAM"])
        if "DSP Blocks" in record_map:
            record.DSPs = int(record_map["DSP Blocks"])

        record.Depth = parse_table_row_depth(cur_line)
        record.UId = uid_counter
        uid_counter += 1
        entries.append(record)

    return UtilReport(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Vivado utilization report produced by project_free_flow.tcl.")
    parser.add_argument("report", help="Path to util_post_route.rpt or util_post_synth.rpt.")
    parser.add_argument("--instance", help="Optional instance name to display.", default=None)
    args = parser.parse_args()

    util = read_vivado_util_report(args.report)
    if args.instance:
        entry = util.get_by_instance_name_or_none(args.instance)
        if entry:
            print(entry.get_string())
        else:
            print(f"Instance '{args.instance}' not found.")
    else:
        for entry in util.all_entries:
            print(entry.get_string())


if __name__ == "__main__":
    main()

