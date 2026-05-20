#!/usr/bin/env python3
"""
VITRIOL TUI Dashboard — Textual-based terminal UI for live inference monitoring.

Usage:
  vitriol tui [--log /path/to/server.log]

Requires: textual (pip install textual)
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Header, Footer, Static, Label, RichLog, LoadingIndicator
from textual.widget import Widget


# ── Helpers ────────────────────────────────────────────────────

def get_vram() -> tuple[float, float]:
    """Return (used_mib, total_mib) via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=3, text=True
        )
        parts = out.strip().split(", ")
        return float(parts[0]), float(parts[1])
    except Exception:
        return 0.0, 0.0

LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "ansi-logo.txt"
def load_logo() -> str:
    """Return ANSI logo as a string, or empty if not found."""
    if not LOGO_PATH.exists():
        return ""
    with open(LOGO_PATH) as f:
        # First line is a comment; second line is the echo command with -e
        lines = f.readlines()
        for line in lines:
            if "echo -e" in line or r"\033" in line:
                # Extract the ANSI escape sequence
                idx = line.find(r"\033")
                if idx >= 0:
                    return line[idx:].strip().rstrip("'").replace(r"\033", "\033").replace("'", "")
        return ""


# ── Log Parser ──────────────────────────────────────────────────

TIMING_RE = re.compile(
    r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+tokens"
)
PROMPT_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+tokens"
)
DRAFT_RE = re.compile(
    r"draft acceptance rate\s*=\s*([\d.]+)"
)

class ServerLogWatcher:
    """Watches server log file for new timing lines."""

    def __init__(self, log_path: str | Path):
        self.path = Path(log_path)
        self._pos = self.path.stat().st_size if self.path.exists() else 0
        self.last_tps: float = 0.0
        self.last_prompt_tps: float = 0.0
        self.last_acceptance: float = 0.0

    def poll(self) -> bool:
        """Read new lines, return True if any new timing data was parsed."""
        if not self.path.exists():
            return False
        with open(self.path, errors="replace") as f:
            f.seek(self._pos)
            new_data = f.read()
            self._pos = f.tell()

        updated = False
        for line in new_data.splitlines():
            m = TIMING_RE.search(line)
            if m:
                ms, tokens = float(m.group(1)), int(m.group(2))
                self.last_tps = (tokens / ms) * 1000 if ms > 0 else 0.0
                updated = True
            m = PROMPT_RE.search(line)
            if m:
                ms, tokens = float(m.group(1)), int(m.group(2))
                self.last_prompt_tps = (tokens / ms) * 1000 if ms > 0 else 0.0
                updated = True
            m = DRAFT_RE.search(line)
            if m:
                self.last_acceptance = float(m.group(1))
                updated = True
        return updated


# ── Widgets ─────────────────────────────────────────────────────

class LogoWidget(Static):
    """ANSI VITRIOL logo."""

    def on_mount(self):
        logo = load_logo()
        if logo:
            self.update(logo)
        else:
            self.update("VITRIOL")


class Gauge(Widget):
    """Simple horizontal bar gauge."""

    DEFAULT_CSS = """
    Gauge {
        height: 1;
        margin: 0 1;
    }
    .gauge-track {
        width: 100%;
        height: 1;
    }
    """

    def __init__(self, value: float, max_value: float, label: str = "",
                 width: int = 20, **kwargs):
        super().__init__(**kwargs)
        self._value = value
        self._max = max_value
        self._label = label
        self._w = width

    def render(self) -> str:
        ratio = self._value / self._max if self._max > 0 else 0
        filled = int(ratio * self._w)
        bar = "█" * filled + "░" * (self._w - filled)
        pct = int(ratio * 100)
        return f"{self._label} {bar} {pct}%"


class StatCard(Static):
    """A bordered card showing a metric."""

    DEFAULT_CSS = """
    StatCard {
        border: solid $primary;
        width: 1fr;
        height: auto;
        padding: 0 1;
        margin: 0 1;
    }
    StatCard > .label {
        color: $text-muted;
    }
    StatCard > .value {
        text-style: bold;
        color: $text;
    }
    """

    def __init__(self, label: str, value: str = "--", **kwargs):
        super().__init__(**kwargs)
        self._label = label
        self._value = value

    def compose(self):
        yield Label(f"{self._label}", classes="label")
        yield Label(f"{self._value}", classes="value", id="stat-value")


class LogView(RichLog):
    """Scrolling log of server events."""

    DEFAULT_CSS = """
    LogView {
        height: 1fr;
        border: solid $surface;
    }
    """


class BottleneckChart(Static):
    """Per-layer bottleneck visualization (placeholder)."""

    DEFAULT_CSS = """
    BottleneckChart {
        height: auto;
        border: solid $secondary;
        padding: 0 1;
    }
    """

    def render(self) -> str:
        return (
            "Layer   FFN         Attn        PCIe\n"
            " 0-10   ████████   ████       ████████\n"
            "11-20   ████████   ████       ████████\n"
            "21-30   ████████   ████       ░░░░ (pin)\n"
            "31-39   ████████   ████       ░░░░ (pin)\n"
        )


# ── Main App ───────────────────────────────────────────────────

class VitriolTUI(App):
    """VITRIOL Live Dashboard."""

    TITLE = "VITRIOL Dashboard"
    SUB_TITLE = "Live inference monitor"
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-rows: auto 1fr auto;
        grid-columns: 1fr 2fr;
    }

    /* Wide layout (>80 cols) */
    @media (min-width: 81) {
        Screen {
            grid-size: 2 3;
            grid-columns: 1fr 2fr;
        }
        #logo-panel { row-span: 1; }
        #stats-panel { row-span: 1; }
        #bottleneck-panel { row-span: 1; column-span: 1; }
        #config-panel { row-span: 1; }
        #log-panel { row-span: 2; column-span: 1; }
    }

    /* Narrow layout (≤80 cols) */
    @media (max-width: 80) {
        Screen {
            grid-size: 1;
            grid-columns: 1fr;
        }
        #logo-panel { display: none; }
        #logo-banner { display: block; }
        #bottleneck-panel { display: none; }
        #stats-panel { }
        #config-panel { }
        #log-panel { }
    }

    #logo-panel {
        border: solid $primary;
        height: auto;
        padding: 0 0;
        content-align: center middle;
    }
    #stats-panel {
        height: auto;
        border: solid $primary;
    }
    #bottleneck-panel {
        height: auto;
        border: solid $secondary;
    }
    #config-panel {
        height: auto;
        border: solid $surface;
    }
    #log-panel {
        height: 1fr;
        border: solid $surface;
    }
    #logo-banner {
        display: none;
        text-style: bold;
        color: $primary;
        height: 1;
    }
    StatRow {
        height: auto;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(self, log_path: str | Path = ""):
        super().__init__()
        if log_path:
            self.watcher = ServerLogWatcher(log_path)
        else:
            config_dir = Path(os.environ.get("HOME", "~")) / ".vitriol"
            self.watcher = ServerLogWatcher(config_dir / "server.log")

    def compose(self):
        with Vertical(id="logo-panel"):
            yield LogoWidget()
        with Vertical(id="stats-panel"):
            yield Label("Status", classes="label")
            yield StatCard("Gen t/s", id="card-tps")
            yield StatCard("Prompt t/s", id="card-ptps")
            yield StatCard("MTP Acc", id="card-mtp")
            yield Static(id="vram-bar")
            yield Static(id="ctx-bar")
        with Vertical(id="bottleneck-panel"):
            yield Label("Bottlenecks", classes="label")
            yield BottleneckChart()
        with Vertical(id="config-panel"):
            yield Label("Config", classes="label")
            yield Static(" [1] Model     [2] GPU\n [3] VITRIOL  [4] Server")
        with Vertical(id="log-panel"):
            yield Label("Log", classes="label")
            yield LogView(id="log")
        yield Static(id="logo-banner", classes="-- VITRIOL --")

    def on_mount(self):
        self.poll_loop()

    @work(thread=True, exit_if_already_running=True)
    async def poll_loop(self):
        while True:
            self.call_from_thread(self._update)
            time.sleep(1)

    def _update(self):
        # Update VRAM
        used, total = get_vram()
        try:
            bar = self.query_one("#vram-bar")
            pct = int((used / total) * 100) if total > 0 else 0
            bar.update(f"VRAM  {'█' * (pct // 5)}{'░' * (20 - pct // 5)} {pct}%")
        except NoMatches:
            pass

        # Update watcher
        updated = self.watcher.poll()

        # Update stats
        try:
            card_tps = self.query_one("#card-tps")
            card_tps._value = f"{self.watcher.last_tps:.1f} t/s"
            card_tps.refresh()
        except NoMatches:
            pass
        try:
            card_ptps = self.query_one("#card-ptps")
            card_ptps._value = f"{self.watcher.last_prompt_tps:.1f} t/s"
            card_ptps.refresh()
        except NoMatches:
            pass
        try:
            card_mtp = self.query_one("#card-mtp")
            acc = self.watcher.last_acceptance
            card_mtp._value = f"{acc*100:.1f}%" if acc > 0 else "--"
            card_mtp.refresh()
        except NoMatches:
            pass

        # Update log
        if updated:
            try:
                log = self.query_one("#log")
                now = time.strftime("%H:%M:%S")
                if self.watcher.last_tps > 0:
                    log.write(f"[{now}] Gen: {self.watcher.last_tps:.1f} t/s")
                if self.watcher.last_acceptance > 0:
                    log.write(f"[{now}] MTP: {self.watcher.last_acceptance*100:.1f}% acc")
            except NoMatches:
                pass


# ── Entry point ────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="VITRIOL TUI Dashboard")
    parser.add_argument("--log", default="", help="Path to server.log")
    args = parser.parse_args()

    from textual import __version__ as textual_version
    app = VitriolTUI(log_path=args.log)
    app.run()


if __name__ == "__main__":
    main()
