"""
Records a per client timeline of run_fedrbe for performance reports.
run_fedrbe marks each span of work with start(kind, round_name, idx) and
end(kind, round_name, idx). Spans of the same role follow each other, spans of
different roles can overlap: the coordinator awaits the aggregated result as a
client (client_await) while it gathers, aggregates and broadcasts it as the
aggregator.
Span kinds:
    client role: client_compute, client_send, client_await
    aggregator role (coordinator only): agg_compute, agg_gather, agg_broadcast
The report only contains raw timestamps (time.time(), so they are comparable
between containers on the same host) and memory values, the interpretation is
done by evaluation_utils/performance_report.py.
"""
import json
import os
import time

REPORT_FILENAME = "performance_report.json"
_STATUS_FILE = "/proc/self/status"


def _status_kib(field: str) -> int:
    """Reads a memory field (in kiB) of this process from /proc/self/status."""
    with open(_STATUS_FILE) as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    raise RuntimeError(f"{field} not found in {_STATUS_FILE}")


class NullRecorder():
    """Recorder used when no performance report is wanted, does nothing."""
    def __init__(self):
        self.info = dict()

    def start(self, kind: str, round_name: str, idx=None) -> None:
        pass

    def end(self, kind: str, round_name: str, idx=None) -> None:
        pass


class PerfRecorder():
    """
    Records the timeline spans and the peak resident memory of
    this process. Create it right before run_fedrbe, as it resets the peak
    resident memory (VmHWM) of the process.
    Raises:
        OSError: if the peak resident memory cannot be reset
    """
    def __init__(self):
        # Writing 5 resets VmHWM to the current resident memory, so VmHWM
        # becomes the peak of the protocol run only
        self.pre_fedrbe_peak_rss_kib = _status_kib("VmHWM")
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        self.baseline_rss_kib = _status_kib("VmRSS")
        self.info = dict()
        self.segments = list()  # closed spans, in the order they ended
        self._open = dict()  # (kind, round_name, idx) -> start time

    def start(self, kind: str, round_name: str, idx=None) -> None:
        """Opens the span (kind, round_name, idx), which must not be open."""
        key = (kind, round_name, idx)
        if key in self._open:
            raise RuntimeError(f"Span {key} is already open")
        self._open[key] = time.time()

    def end(self, kind: str, round_name: str, idx=None) -> None:
        """Closes the open span (kind, round_name, idx)."""
        # read the memory before the end time, so the time of the read
        # belongs to this span instead of a gap to the next span
        rss_kib = _status_kib("VmRSS")
        now = time.time()
        key = (kind, round_name, idx)
        if key not in self._open:
            raise RuntimeError(f"Span {key} is not open")
        self.segments.append({"kind": kind, "round": round_name, "idx": idx,
                              "t_start": self._open.pop(key), "t_end": now,
                              "rss_end_kib": rss_kib})

    def write(self, outputfolder: str, client_id: str, is_coordinator: bool,
              clients: list) -> None:
        """
        Writes the report as REPORT_FILENAME to outputfolder.
        Raises:
            RuntimeError: if a span is still open
        """
        if self._open:
            raise RuntimeError(f"Spans still open: {list(self._open)}")
        # the peak is read last, so it covers the final resident memory
        final_rss_kib = _status_kib("VmRSS")
        report = {
            "client_id": client_id,
            "is_coordinator": bool(is_coordinator),
            "clients": list(clients),
            "info": self.info,
            "baseline_rss_kib": self.baseline_rss_kib,
            "pre_fedrbe_peak_rss_kib": self.pre_fedrbe_peak_rss_kib,
            "final_rss_kib": final_rss_kib,
            "peak_rss_kib": _status_kib("VmHWM"),
            "segments": self.segments,
        }
        with open(os.path.join(outputfolder, REPORT_FILENAME), "w") as f:
            json.dump(report, f, indent=1)
