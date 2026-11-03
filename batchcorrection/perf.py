"""
Records a per client timeline of run_fedrbe for performance reports.
run_fedrbe calls switch(kind, round_name, idx) whenever it changes what it is
doing, which closes the open segment and opens the next one. So the segments
are contiguous and every moment of the run belongs to exactly one segment.
Segment kinds:
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

    def switch(self, kind: str, round_name: str, idx=None) -> None:
        pass

    def end(self) -> None:
        pass


class PerfRecorder():
    """
    Records contiguous timeline segments and the peak resident memory of
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
        self.segments = list()
        self._open = None

    def switch(self, kind: str, round_name: str, idx=None) -> None:
        now = time.time()
        self._close(now)
        self._open = {"kind": kind, "round": round_name, "idx": idx, "t_start": now}

    def end(self) -> None:
        self._close(time.time())

    def _close(self, now: float) -> None:
        if self._open is None:
            return
        self._open["t_end"] = now
        self._open["rss_end_kib"] = _status_kib("VmRSS")
        self.segments.append(self._open)
        self._open = None

    def write(self, outputfolder: str, client_id: str, is_coordinator: bool,
              clients: list) -> None:
        """Writes the report as REPORT_FILENAME to outputfolder."""
        self.end()
        report = {
            "client_id": client_id,
            "is_coordinator": bool(is_coordinator),
            "clients": list(clients),
            "info": self.info,
            "baseline_rss_kib": self.baseline_rss_kib,
            "pre_fedrbe_peak_rss_kib": self.pre_fedrbe_peak_rss_kib,
            "peak_rss_kib": _status_kib("VmHWM"),
            "final_rss_kib": _status_kib("VmRSS"),
            "segments": self.segments,
        }
        with open(os.path.join(outputfolder, REPORT_FILENAME), "w") as f:
            json.dump(report, f, indent=1)
