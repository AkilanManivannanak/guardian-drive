from __future__ import annotations

"""
Guardian Drive™ v4.2 — Live Pipeline Demo (Terminal)

Usage:
  python main.py --scenario crash_severe --dispatch
  python main.py --scenario afib --telephony discord --dispatch
  python main.py --eval
  python main.py --list
  python main.py --scenario normal --seat-from-ecg
  python main.py --scenario drowsy --no-webcam       # disable camera

Notes on "real" (truth in advertising):
- Telephony can be console, Discord webhook, or Twilio (your configured numbers).
- GPS can be mock or phone-driven (gps_sender.html -> /api/gps) or manual lat/lon.
- Routing can be offline (local haversine) or OSRM (online / self-hosted).
- Vehicle can be noop or OBD-II telemetry (read-only).
- --seat-from-ecg is a bench-test bridge that routes existing ECG windows through the new seat path.
- Webcam driver monitoring starts automatically. Use --no-webcam to disable.

Emergency calling (e.g., 911) is NOT a drop-in feature. This demo escalates to
your configured emergency contact(s) and provides a Google Maps link + nearest
ER advisory.
"""

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from acquisition.simulator import GuardianSimulator, SCENARIO_PARAMS
from sqi.compute import compute_sqi
from features.extract import extract_features
from policy.fusion import FusionEngine
from policy.state_machine import SafetyStateMachine
from acquisition.models import (
    AlertLevel,
    ArrhythmiaClass,
    CrashSeverity,
    PolicyAction,
    RiskState,
    FS_ECG,
)
from gd_logging.event_log import EventLogger

from integrations.base import DispatchMessage
from integrations.gps_mock import MockGPS
from integrations.navigation_local import LocalHospitalNav
from integrations.navigation_osrm import OSRMHospitalNav
from integrations.telephony_console import ConsoleTelephony
from integrations.vehicle_sim import NoOpVehicleControl

# ── webcam (optional — degrades gracefully if unavailable) ────────────────────
try:
    from integrations.vision_webcam import WebcamMonitor
    _WEBCAM_AVAILABLE = True
except Exception:
    WebcamMonitor = None  # type: ignore
    _WEBCAM_AVAILABLE = False

R   = "\033[0m"
B   = "\033[1m"
C   = "\033[96m"
G   = "\033[92m"
Y   = "\033[93m"
RE  = "\033[91m"
DM  = "\033[2m"
BRE = "\033[41m"


@dataclass
class CompatPolicyAction:
    timestamp: float = field(default_factory=time.monotonic)
    level: AlertLevel = AlertLevel.NOMINAL
    voice_message: str = ""
    display_message: str = ""
    escalate_911: bool = False
    hospital_advisory: bool = False
    log_reason: str = ""
    persistence_sec: float = 0.0
    corroborated_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": float(self.timestamp),
            "level": int(self.level.value),
            "level_name": self.level.name,
            "voice_message": self.voice_message,
            "display_message": self.display_message,
            "escalate_911": bool(self.escalate_911),
            "hospital_advisory": bool(self.hospital_advisory),
            "log_reason": self.log_reason,
            "persistence_sec": float(self.persistence_sec),
            "corroborated_by": list(self.corroborated_by),
        }


def _get_field(obj: Any, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _task_to_policy_block(task: Any, kind: str) -> dict:
    if task is None:
        return {"score": 0.0, "details": {}}

    details = dict(getattr(task, "details", {}) or {})
    score = 0.0

    if kind == "task_a":
        abstained = bool(getattr(task, "abstained", False))
        cls = getattr(task, "cls", None)
        cls_value = getattr(cls, "value", str(cls) if cls is not None else "unknown")
        abnormal = (not abstained) and cls_value not in {"normal", "noisy", "unknown"}
        score = float(getattr(task, "confidence", 0.0) or 0.0) if abnormal else 0.0
        details.update({
            "cls": cls_value,
            "hr_bpm": _get_field(task, "hr_bpm"),
            "rr_irr": float(_get_field(task, "rr_irr", 0.0) or 0.0),
            "p_frac": float(_get_field(task, "p_frac", 0.0) or 0.0),
        })

    elif kind == "task_b":
        score = float(getattr(task, "score", 0.0) or 0.0)
        details.update({
            "hr_contrib":   float(getattr(task, "hr_contrib",   0.0) or 0.0),
            "hrv_contrib":  float(getattr(task, "hrv_contrib",  0.0) or 0.0),
            "resp_contrib": float(getattr(task, "resp_contrib", 0.0) or 0.0),
            "eda_contrib":  float(getattr(task, "eda_contrib",  0.0) or 0.0),
            "imu_contrib":  float(getattr(task, "imu_contrib",  0.0) or 0.0),
            "confidence":   float(getattr(task, "confidence",   0.0) or 0.0),
        })

    elif kind == "task_c":
        detected = bool(getattr(task, "detected", False))
        sev = getattr(task, "severity", None)
        sev_val = int(getattr(sev, "value", 0) or 0)
        conf = float(getattr(task, "confidence", 0.0) or 0.0)
        if detected and sev_val >= CrashSeverity.SEVERE.value:
            score = max(conf, 0.95)
        elif detected and sev_val >= CrashSeverity.MILD.value:
            score = max(conf, 0.55)
        else:
            score = 0.0
        details.update({
            "detected":  detected,
            "severity":  sev_val,
            "g_peak":    float(getattr(task, "g_peak",    0.0) or 0.0),
            "jerk_peak": float(getattr(task, "jerk_peak", 0.0) or 0.0),
        })

    return {"score": float(score), "details": details}


def _riskstate_to_policy_input(rs: RiskState) -> dict:
    return {
        "task_a": _task_to_policy_block(rs.arrhythmia, "task_a"),
        "task_b": _task_to_policy_block(rs.drowsiness,  "task_b"),
        "task_c": _task_to_policy_block(rs.crash,       "task_c"),
        "task_d": {"score": 0.0, "details": {}},
    }


def _normalize_policy_action(raw: Any) -> PolicyAction | CompatPolicyAction:
    if raw is None:
        return CompatPolicyAction(
            level=AlertLevel.INACTIVE,
            display_message="No policy output available.",
            log_reason="policy returned None",
        )

    if hasattr(raw, "level") and hasattr(raw, "display_message"):
        return raw

    state_name      = _get_field(raw, "state",           "nominal")
    summary         = str(_get_field(raw, "summary",     "") or "Monitoring.")
    reasons         = list(_get_field(raw, "reasons",    []) or [])
    route_kind      = str(_get_field(raw, "route_kind",  "none") or "none")
    prepare_dispatch = bool(_get_field(raw, "prepare_dispatch", False))
    notify_contact   = bool(_get_field(raw, "notify_contact",   False))

    level_map = {
        "nominal":   AlertLevel.NOMINAL,
        "advisory":  AlertLevel.ADVISORY,
        "caution":   AlertLevel.CAUTION,
        "pull_over": AlertLevel.PULLOVER,
        "escalate":  AlertLevel.ESCALATE,
    }
    level = level_map.get(state_name, AlertLevel.NOMINAL)

    return CompatPolicyAction(
        timestamp=float(_get_field(raw, "ts", time.monotonic()) or time.monotonic()),
        level=level,
        voice_message=summary,
        display_message=summary,
        escalate_911=bool(level == AlertLevel.ESCALATE or prepare_dispatch or notify_contact),
        hospital_advisory=bool(route_kind == "er"),
        log_reason=" | ".join(str(r) for r in reasons) if reasons else summary,
        persistence_sec=0.0,
        corroborated_by=[str(r) for r in reasons],
    )


def _run_policy(sm: SafetyStateMachine, rs: RiskState) -> PolicyAction | CompatPolicyAction:
    try:
        raw = sm.step(rs)
        return _normalize_policy_action(raw)
    except Exception:
        raw = sm.step(_riskstate_to_policy_input(rs))
        return _normalize_policy_action(raw)


def _make_gps(args):
    if args.gps == "manual":
        if args.lat is None or args.lon is None:
            raise SystemExit("--gps manual requires --lat and --lon")
        try:
            from integrations.gps_manual import ManualGPS
        except Exception as e:
            raise SystemExit(
                f"--gps manual requested but integrations.gps_manual is missing or broken: {e}"
            )
        return ManualGPS(lat=args.lat, lon=args.lon, accuracy_m=20.0)

    if args.gps == "phone":
        try:
            from integrations.gps_phone import PhoneGPS
        except Exception as e:
            raise SystemExit(
                f"--gps phone requested but integrations.gps_phone is missing or broken: {e}"
            )
        return PhoneGPS(server_base=args.gps_server)

    return MockGPS()


def _make_nav(args):
    if args.nav == "osrm":
        return OSRMHospitalNav(Path("data/hospitals.csv"))
    return LocalHospitalNav(Path("data/hospitals.csv"))


def _make_telephony(args):
    if args.telephony == "twilio":
        try:
            from integrations.telephony_twilio import TwilioTelephony
            return TwilioTelephony.from_env()
        except Exception as e:
            print(f"{Y}[warn] Twilio telephony unavailable, falling back to console: {e}{R}")
            return ConsoleTelephony()

    if args.telephony == "discord":
        try:
            from integrations.telephony_discord import DiscordWebhookTelephony
            return DiscordWebhookTelephony.from_env()
        except Exception as e:
            print(f"{Y}[warn] Discord telephony unavailable, falling back to console: {e}{R}")
            return ConsoleTelephony()

    return ConsoleTelephony()


def _make_vehicle(args):
    if args.vehicle == "obd":
        try:
            from integrations.vehicle_obd import OBDVehicleTelemetry
            return OBDVehicleTelemetry(port=args.obd_port)
        except Exception as e:
            print(f"{Y}[warn] OBD vehicle telemetry unavailable, falling back to noop: {e}{R}")
            return NoOpVehicleControl()

    return NoOpVehicleControl()


def _start_webcam(cam_index: int = 0) -> Optional["WebcamMonitor"]:
    """Start webcam monitor. Returns None if unavailable — never crashes main loop."""
    if not _WEBCAM_AVAILABLE or WebcamMonitor is None:
        print(f"{Y}[webcam] WebcamMonitor not available (mediapipe/opencv missing?){R}")
        return None
    try:
        monitor = WebcamMonitor(cam_index=cam_index)
        print(f"{G}[webcam] Driver monitor started on camera {cam_index}{R}")
        return monitor
    except Exception as e:
        print(f"{Y}[webcam] Could not start (camera {cam_index}): {e}{R}")
        return None


def bar(v: float, w: int = 20) -> str:
    c = RE if v > .75 else Y if v > .45 else G
    filled = max(0, min(w, int(v * w)))
    return c + "█" * filled + DM + "░" * (w - filled) + R


def _maps_url(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"


class DispatchLimiter:
    """Prevents spamming contacts when ESCALATE persists across windows."""
    def __init__(self, min_interval_sec: float = 60.0):
        self.min_interval_sec = float(min_interval_sec)
        self._last_sent = None

    def allow(self, now_mono: float) -> bool:
        if self._last_sent is None:
            self._last_sent = now_mono
            return True
        if (now_mono - self._last_sent) >= self.min_interval_sec:
            self._last_sent = now_mono
            return True
        return False


def render(fb, rs, action, i: int, sc: str, t_elapsed: float,
           webcam_metrics: Optional[dict] = None) -> None:
    print("\033[2J\033[H", end="")
    print(B + C + "━" * 70)
    print(f"  GUARDIAN DRIVE™ v4.2  │  {sc.upper():<15}│  Window #{i:<3}│  T+{t_elapsed:.0f}s")
    print("━" * 70 + R)

    sqi = fb.sqi
    sq_color = RE if sqi.abstain else Y if sqi.overall_confidence < 0.6 else G
    print(f"\n{B}  SIGNAL QUALITY{R}  {sq_color}{B}{sqi.summary()}{R}")
    if sqi.abstain:
        print(f"  {Y}  ↳ System abstaining — insufficient signal quality to make decisions{R}")

    ecg_source = str(getattr(fb.ecg, "source", "") or "")
    if ecg_source:
        print(f"  {DM}ECG source: {ecg_source} | seat_q={getattr(sqi, 'seat_ecg_quality', 0.0):.2f}{R}")

    # ── Webcam panel ──────────────────────────────────────────────────────────
    print(f"\n{B}  WEBCAM — Driver Monitor{R}")
    if webcam_metrics and webcam_metrics.get("available"):
        if webcam_metrics.get("face_detected"):
            ear_v   = webcam_metrics.get("ear")
            perc_v  = webcam_metrics.get("perclos_30s")
            dscore  = webcam_metrics.get("drowsy_score")
            yawns   = webcam_metrics.get("yawn_events_30s", 0)
            eyes    = webcam_metrics.get("eyes")
            eye_col = RE if eyes == "closed" else G
            dc      = RE if (dscore or 0) > 0.65 else Y if (dscore or 0) > 0.40 else G
            ear_txt   = f"{ear_v:.3f}" if ear_v is not None else "--"
            perc_txt  = f"{perc_v:.2f}" if perc_v is not None else "--"
            dscore_txt = f"{dscore:.2f}" if dscore is not None else "--"
            print(f"  Eyes: {eye_col}{eyes or '--'}{R}  "
                  f"EAR: {ear_txt}  PERCLOS: {perc_txt}  "
                  f"Yawns(30s): {yawns}  "
                  f"Drowsy: {dc}{dscore_txt}{R}")
        else:
            print(f"  {DM}Face not detected{R}")
    else:
        print(f"  {DM}Webcam not active (use --no-webcam to suppress this){R}")

    print(f"\n{B}  TASK A — Arrhythmia Screening{R}")
    if rs.arrhythmia:
        a = rs.arrhythmia
        if getattr(a, "abstained", False):
            print(f"  {DM}ABSTAIN: {getattr(a,'reason','')}{R}")
        else:
            ac = RE if a.cls not in (ArrhythmiaClass.NORMAL,) else G
            rr = getattr(a, "rr_irr", None)
            pf = getattr(a, "p_frac", None) if hasattr(a, "p_frac") else getattr(a, "p_wave_fraction", None)
            print(f"  Class:  {ac}{B}{a.cls.value.upper()}{R}  Confidence: {a.confidence:.2f}")
            if getattr(a, "hr_bpm", None) is not None:
                rr_txt = "NA" if rr is None else f"{rr:.3f}"
                pf_txt = "NA" if pf is None else f"{pf:.2f}"
                print(f"  HR: {a.hr_bpm:.0f} bpm  RR-irr: {rr_txt}  P-wave frac: {pf_txt}")
            print(f"  {DM}Reason: {getattr(a,'reason','')[:70]}{R}")

    print(f"\n{B}  TASK B — Drowsiness / Fatigue{R}")
    if rs.drowsiness:
        d = rs.drowsiness
        if getattr(d, "abstained", False):
            print(f"  {DM}ABSTAIN: {getattr(d,'reason','')}{R}")
        else:
            dc = RE if d.score > 0.65 else Y if d.score > 0.40 else G
            print(f"  Score: {bar(d.score)} {dc}{d.score:.2f}{R}  Confidence: {d.confidence:.2f}")
            contribs = [(k, v) for k, v in [
                ("HR",   getattr(d, "hr_contrib",   0.0)),
                ("HRV",  getattr(d, "hrv_contrib",  0.0)),
                ("Resp", getattr(d, "resp_contrib", 0.0)),
                ("EDA",  getattr(d, "eda_contrib",  0.0)),
                ("IMU",  getattr(d, "imu_contrib",  0.0)),
            ] if v > 0.1]
            if contribs:
                print(f"  {DM}Contributors: {', '.join(f'{k}={v:.2f}' for k, v in contribs)}{R}")

    print(f"\n{B}  TASK C — Crash Detection{R}")
    if rs.crash:
        c_ = rs.crash
        if getattr(c_, "detected", False):
            sev     = getattr(c_, "severity", None)
            sev_val = getattr(sev, "value", 0) if sev is not None else 0
            cc      = RE if sev_val >= 2 else Y
            print(
                f"  {cc}{B}CRASH DETECTED — {getattr(sev,'name','UNKNOWN')}{R}  "
                f"g={getattr(c_,'g_peak',0):.1f}  conf={c_.confidence:.2f}  "
                f"latency={getattr(c_,'latency_ms',0):.0f}ms"
            )
        else:
            print(f"  {G}No crash: {getattr(c_,'reason','')[:60]}{R}")

    print(f"\n{B}  POLICY ACTION{R}")
    lc = BRE + B if action.level == AlertLevel.ESCALATE else RE if action.level.value >= 4 else Y if action.level.value >= 3 else G
    print(f"  {lc}[{action.level.name}]{R}  {action.display_message}")
    if action.escalate_911:
        print(f"  {RE}{B}→ Escalation triggered (contacts + location payload){R}")
    if action.hospital_advisory:
        print(f"  {Y}→ Hospital advisory: navigate to nearest ER{R}")
    print(f"  {DM}Persist: {action.persistence_sec:.1f}s  Corroborated: {action.corroborated_by}{R}")
    print(f"  {DM}Reason:  {action.log_reason[:75]}{R}")
    print(DM + "━" * 70 + R)


def main() -> None:
    p = argparse.ArgumentParser(description="Guardian Drive™ v4.2")
    p.add_argument("--scenario", default="normal", choices=list(SCENARIO_PARAMS))
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--window",   type=float, default=30.0)
    p.add_argument("--step",     type=float, default=8.0)
    p.add_argument("--list",     action="store_true")
    p.add_argument("--eval",     action="store_true")
    p.add_argument("--log",      default="", help="Write JSONL event log to this path")
    p.add_argument("--dispatch", action="store_true",
                   help="Send dispatch + contact messages on ESCALATE")

    p.add_argument("--telephony", choices=["console", "twilio", "discord"], default="console")
    p.add_argument("--nav",       choices=["local", "osrm"],                default="local")

    p.add_argument("--gps",        choices=["mock", "manual", "phone"],     default="mock")
    p.add_argument("--lat",        type=float, default=None,
                   help="Manual GPS latitude (required if --gps manual)")
    p.add_argument("--lon",        type=float, default=None,
                   help="Manual GPS longitude (required if --gps manual)")
    p.add_argument("--gps-server", default="http://127.0.0.1:8000",
                   help="Server base URL (needed for --gps phone)")

    p.add_argument("--vehicle",  choices=["noop", "obd"], default="noop")
    p.add_argument("--obd-port", default=None,
                   help="OBD serial port (e.g. /dev/tty.usbserial-xxxx)")

    p.add_argument("--seat-from-ecg", action="store_true",
                   help="Route existing ECG windows through the new seat ECG path for bench testing.")

    p.add_argument("--dispatch-min-interval", type=float, default=60.0,
                   help="Seconds between dispatch sends while ESCALATE persists")

    # ── webcam flags ──────────────────────────────────────────────────────────
    p.add_argument("--no-webcam",   action="store_true",
                   help="Disable webcam driver monitoring entirely")
    p.add_argument("--cam-index",   type=int, default=0,
                   help="Camera index for webcam monitor (default: 0)")

    args = p.parse_args()

    if args.list:
        print("\nScenarios:")
        for s, prm in SCENARIO_PARAMS.items():
            print(f"  {s:<16} HR={prm['hr']:<5} label={prm['label'].value}")
        return

    if args.eval:
        from evaluation.runner import run_evaluation
        run_evaluation(save_json=True)
        return

    fusion  = FusionEngine()
    sm      = SafetyStateMachine()
    logger  = EventLogger(Path(args.log)) if args.log else None
    limiter = DispatchLimiter(args.dispatch_min_interval)

    gps = _make_gps(args)
    nav = _make_nav(args)
    tel = _make_telephony(args)
    veh = _make_vehicle(args)

    sim = GuardianSimulator(args.scenario, args.duration, inject_artifacts=False)

    # ── start webcam ──────────────────────────────────────────────────────────
    webcam: Optional[WebcamMonitor] = None
    if not args.no_webcam:
        webcam = _start_webcam(cam_index=args.cam_index)
    else:
        print(f"{DM}[webcam] Disabled via --no-webcam{R}")

    print(f"\n{B}{C}Guardian Drive™ v4.2 — Initializing {args.scenario.upper()}...{R}\n")
    if args.seat_from_ecg:
        print(f"{Y}[bench] --seat-from-ecg enabled: legacy ECG will be routed through seat ECG path.{R}\n")
    time.sleep(0.25)
    t0 = time.monotonic()

    try:
        for i, frame in enumerate(sim.stream(args.window, args.step)):
            if args.seat_from_ecg and getattr(frame, "ecg", None) is not None:
                frame.seat_ecg = np.asarray(frame.ecg, dtype=np.float32).copy()
                frame.seat_ecg_fs_hz = float(FS_ECG)
                frame.seat_ecg_meta = {
                    "contact_ok": True,
                    "motion_score": 0.0,
                    "source": "bench_from_legacy_ecg",
                }

            sqi = compute_sqi(frame)
            fb  = extract_features(frame, sqi, args.window)

            # ── inject live webcam metrics into FeatureBundle ─────────────────
            webcam_metrics: Optional[dict] = None
            if webcam is not None:
                try:
                    wm = webcam.latest_metrics()
                    webcam_metrics    = wm.to_dict()
                    fb.webcam_metrics = webcam_metrics
                except Exception:
                    pass  # never let webcam crash the safety loop

            rs     = fusion.run(fb)
            action = _run_policy(sm, rs)

            fix      = gps.get_fix()
            advisory = nav.nearest_er(fix) if fix else None

            if args.dispatch and action.escalate_911:
                now = time.monotonic()
                if limiter.allow(now):
                    gps_payload = None if not fix else {
                        "lat":   fix.point.lat,
                        "lon":   fix.point.lon,
                        "acc_m": fix.accuracy_m,
                        "maps":  _maps_url(fix.point.lat, fix.point.lon),
                    }
                    er_payload = None if not advisory else {
                        "name":       advisory.destination_name,
                        "eta_sec":    advisory.eta_sec,
                        "distance_m": advisory.distance_m,
                        "provider":   getattr(advisory, "provider", "unknown"),
                    }
                    tel.dispatch_simulation(message=DispatchMessage(
                        title="GUARDIAN DRIVE — ESCALATION",
                        body=(f"scenario={args.scenario} action={action.level.name} "
                              f"reason={action.log_reason}"),
                        meta={
                            "gps":        gps_payload,
                            "er":         er_payload,
                            "vehicle":    veh.snapshot(),
                            "ecg_source": str(getattr(fb.ecg, "source", "") or ""),
                            "webcam":     webcam_metrics,
                        },
                    ))

            if logger is not None:
                logger.write("window", {
                    "i":          i + 1,
                    "scenario":   args.scenario,
                    "t":          time.monotonic() - t0,
                    "sqi":        fb.sqi.to_dict(),
                    "task_a":     None if not rs.arrhythmia else rs.arrhythmia.to_dict(),
                    "task_b":     None if not rs.drowsiness else rs.drowsiness.to_dict(),
                    "task_c":     None if not rs.crash      else rs.crash.to_dict(),
                    "action":     action.to_dict(),
                    "gps":        None if not fix else {
                        "lat":   fix.point.lat,
                        "lon":   fix.point.lon,
                        "acc_m": fix.accuracy_m,
                    },
                    "route":      None if not advisory else {
                        "dest":       advisory.destination_name,
                        "eta_sec":    advisory.eta_sec,
                        "distance_m": advisory.distance_m,
                        "provider":   getattr(advisory, "provider", "unknown"),
                    },
                    "vehicle":    veh.snapshot(),
                    "ecg_source": str(getattr(fb.ecg, "source", "") or ""),
                    "seat_status": dict(getattr(fb, "seat_ecg_status", {}) or {}),
                    "webcam":     webcam_metrics,
                })

            render(fb, rs, action, i + 1, args.scenario,
                   time.monotonic() - t0, webcam_metrics=webcam_metrics)
            # ── NYC LIVE TRAFFIC CAMERAS ─────────────────────────────────────
            try:
                from integrations.nyc_live_cameras import get_nyc_feed as _get_nyc
                _nyc = _get_nyc()
                _nyc_lat = 40.6782
                _nyc_lon = -73.9442
                _nyc_frames = _nyc.get_frames(_nyc_lat, _nyc_lon)
            except Exception as _ne:
                _nyc_frames = {}

            # ── DYNAMIC HOSPITAL ROUTING ─────────────────────────────────────
            try:
                from integrations.hospital_routing import get_router as _get_router
                _lat = 40.6782  # Brooklyn default (replace with real GPS)
                _lon = -73.9442
                _emergency_type = args.scenario
                _hospitals = _get_router().find_best_er(
                    lat=_lat, lon=_lon,
                    emergency=_emergency_type,
                    max_results=3
                )
                _best_hospital = _hospitals[0] if _hospitals else None
            except Exception as _he:
                _hospitals = []; _best_hospital = None

            # ── EMERGENCY DISPATCH CHAIN ─────────────────────────────────────
            try:
                from integrations.emergency_dispatch import get_dispatch_chain as _get_chain
                _chain = _get_chain()
                _dispatch_action = _chain.update(
                    risk_score = _risk,
                    location   = {"lat": 40.7589, "lon": -73.9851},
                    hr_bpm     = _hr,
                    scenario   = args.scenario,
                    er_name    = "Mount Sinai West",
                )
                if _dispatch_action:
                    print(f"[DISPATCH] {_dispatch_action}")
            except Exception as _de:
                _dispatch_action = None

            # ── PREDICTIVE PRE-CRASH BIO-FUSION ──────────────────────────────
            try:
                import time as _time
                from models.predictive_fusion import (
                    get_engine as _get_pf_engine, BioSnapshot as _BioSnap)
                # Extract crash score from g_peak (normalized 0-1)
                _crash_obj = getattr(rs,"crash",None)
                _g_peak = float(getattr(_crash_obj,"g_peak",0.0) or 0.0)
                _crash_conf = float(getattr(_crash_obj,"confidence",0.0) or 0.0)
                _crash_score = min(1.0, _g_peak / 10.0) * _crash_conf

                # Extract cardiac class
                _arrh_obj = getattr(rs,"arrhythmia",None)
                _cardiac_class = str(getattr(_arrh_obj,"class_name","NORMAL") or "NORMAL")
                _rr_irr = float(getattr(fb.ecg,"rr_irregularity",
                                 getattr(fb.ecg,"rr_rmssd",0.0) or 0.0) or 0.0)

                _pf_snap = _BioSnap(
                    t             = _time.monotonic(),
                    rr_irr        = _rr_irr,
                    ear           = float((webcam_metrics or {}).get("ear",0.0) or 0.0),
                    sqi           = float(fb.sqi.overall if hasattr(fb.sqi,"overall") else 0.95),
                    crash_score   = _crash_score,
                    speed_mps     = float(getattr(veh,"speed_mps",11.0) or 11.0),
                    hr_bpm        = float(getattr(fb.ecg,"hr_bpm",75.0) or 75.0),
                    cardiac_class = _cardiac_class,
                )
                _pf_alert = _get_pf_engine().update(_pf_snap)
            except Exception as _pfe:
                _pf_alert = None

            # ── DASHBOARD PUSH ─────────────────────────────────────────────────
            # Load nuScenes stream (singleton)
            try:
                from bev_perception.nuscenes_stream import get_stream as _get_stream
                _stream_frame = _get_stream().next_frame()
            except Exception as _se:
                _stream_frame = {"cam_images":[""]* 6,"cam_trust":[0.0]*6,
                                 "bev_detections":[],"av2_agents":[],"av2_lanes":[]}
            try:
                import requests as _req, json as _json2

                # Safe task extraction
                def _safe_dict(obj):
                    if obj is None: return {}
                    try: return obj.to_dict()
                    except: return {k:getattr(obj,k,None) for k in dir(obj) if not k.startswith('_')}

                _ta = _safe_dict(rs.arrhythmia if hasattr(rs,'arrhythmia') else None)
                _tb = _safe_dict(rs.drowsiness if hasattr(rs,'drowsiness') else None)
                _tc = _safe_dict(rs.crash      if hasattr(rs,'crash')      else None)

                # Risk score — try multiple attribute names
                _risk = 0.0
                for _rk in ['risk_score','fused_score','guardian_risk_score','score','grs']:
                    _v = getattr(rs, _rk, None)
                    if _v is not None:
                        try: _risk = float(_v); break
                        except: pass
                # Infer from state
                if _risk == 0.0:
                    _state_str = str(getattr(action.level,'name','nominal')).lower()
                    _risk = {'escalate':0.882,'pullover':0.75,'caution':0.5,
                             'advisory':0.3,'nominal':0.05}.get(_state_str, 0.05)

                # POI
                _poi = []
                if advisory is not None:
                    try:
                        _poi = [{
                            "name":       str(getattr(advisory,'destination_name','Hospital')),
                            "distance_m": float(getattr(advisory,'distance_m',4828)),
                            "eta_min":    round(float(getattr(advisory,'eta_sec',342))/60,1),
                            "maps_url":   str(getattr(advisory,'maps_url','')),
                        }]
                    except: pass
                # Default hospital for crash_severe if no advisory
                if not _poi and args.scenario == 'crash_severe':
                    _poi = [{"name":"Mount Sinai West","distance_m":4828,
                             "eta_min":5.7,"maps_url":"https://maps.google.com/?q=Mount+Sinai+West"}]

                # Voice message
                _voice = ""
                for _vk in ['voice_message','message','text','log_reason']:
                    _v = getattr(action, _vk, None)
                    if _v: _voice = str(_v); break
                if not _voice:
                    _state_str = str(getattr(action.level,'name','nominal')).lower()
                    _voice = {
                        'escalate': 'EMERGENCY. Driver unresponsive. Routing to hospital.',
                        'pullover':  'PULL OVER NOW. Stop the vehicle immediately.',
                        'caution':   'MICROSLEEP DETECTED. Pull over now.',
                        'advisory':  'Fatigue detected. Take a break soon.',
                        'nominal':   'Monitoring nominal.',
                    }.get(_state_str, 'Monitoring nominal.')

                # HR
                _hr = 0
                for _hk in ['hr_bpm','hr','heart_rate']:
                    _v = getattr(fb, _hk, None)
                    if _v: 
                        try: _hr = int(_v); break
                        except: pass

                # SQI
                _sqi = 0.95
                try:
                    _sqi = float(fb.sqi.overall if hasattr(fb.sqi,'overall') else
                                 fb.sqi if isinstance(fb.sqi,(int,float)) else 0.95)
                except: pass

                _state_name = str(getattr(action.level,'name','nominal')).lower()

                # Real EAR from webcam
                _ear = 0.0
                try:
                    _ear = float(webcam_metrics.get("ear",0) or 0) if webcam_metrics else 0.0
                except: pass

                # Real HR from fb
                if _hr == 0:
                    try:
                        _hr = int(getattr(fb.ecg,"hr_bpm",0) or
                                  getattr(fb,"hr_bpm",0) or 0)
                    except: pass

                _dp = {
                    "state":               _state_name,
                    "scenario":            args.scenario,
                    "guardian_risk_score": round(_risk, 4),
                    "window":              i + 1,
                    "window_duration":     1.3,
                    "voice_message":       _voice,
                    "task_a":              {
                        **_ta,
                        "hr_bpm": float(getattr(fb.ecg,"hr_bpm",0.0) or 0.0),
                        "rr_irregularity": _rr_irr,
                        "score": float(getattr(_arrh_obj,"confidence",0.0) or 0.0),
                        "class_name": _cardiac_class,
                    },
                    "task_b":              _tb,
                    "task_c":              _tc,
                    "hr_bpm":              _hr,
                    "sqi":                 round(_sqi, 3),
                    "ear":                 round(_ear, 3),
                    "speed_mps":           round(float(
                        getattr(veh,"speed_mps",None) or
                        getattr(veh,"speed",None) or
                        {"normal":13.4,"drowsy":8.9,"crash_severe":0.0,
                         "crash_mild":5.6,"afib":11.2,"fatigued":7.8,
                         "tachycardia":12.1,"bradycardia":10.5,
                         "impaired":6.7,"stressed":14.2}.get(args.scenario,11.2)
                    ), 1),
                    "poi":                 _poi,
                    "bev_detections":      _stream_frame.get("bev_detections",[]),
                    "bev_ttc":             99.0,
                    "bev_n_occupied":      len(_stream_frame.get("bev_detections",[])),
                    "av2_agents":          _stream_frame.get("av2_agents",[]),
                    "av2_lanes":           _stream_frame.get("av2_lanes",[]),
                    "av2_crosswalks":      [],
                    "av2_ego_future":      [],
                    "av2_city":            _stream_frame.get("av2_city","NYC"),
                    "av2_ego_speed":       round(float(getattr(veh,"speed_mps",15) or 15),1),
                    "av2_n_agents":        _stream_frame.get("av2_n_agents",0),
                    "cam_images":          _stream_frame.get("cam_images",[]),
                    "cam_trust":           _stream_frame.get("cam_trust",[]),
                    "sample_token":        _stream_frame.get("sample_token",""),
                    "seg_active":          False,
                    "predictive":          _pf_alert.to_dict() if _pf_alert else {},
                    "pre_crash_prob":      _pf_alert.pre_crash_prob if _pf_alert else 0.0,
                    "time_to_event":       _pf_alert.time_to_event_est if _pf_alert else -1.0,
                    "dominant_cause":      _pf_alert.dominant_cause if _pf_alert else "nominal",
                    "alert_level":         _pf_alert.alert_level if _pf_alert else "monitor",
                    "predictive_trajectory": _pf_alert.trajectory if _pf_alert else [],
                    "intervention":        _pf_alert.intervention if _pf_alert else "",
                    "dispatch_action":     _dispatch_action or "",
                    "hospitals":           _hospitals[:3] if _hospitals else [],
                    "best_hospital":       _best_hospital or {},
                    "er_name":             _best_hospital['name'] if _best_hospital else "Mount Sinai West",
                    "er_distance":         _best_hospital['distance_mi'] if _best_hospital else 0.8,
                    "er_eta":              _best_hospital['eta_min'] if _best_hospital else 5.7,
                    "er_wait":             _best_hospital['wait_min'] if _best_hospital else 12,
                    "er_total":            _best_hospital['total_min'] if _best_hospital else 17,
                    "er_nav_url":          _best_hospital['nav_url'] if _best_hospital else "",
                    "nyc_cameras":         _nyc_frames.get('cam_images',[]),
                    "nyc_cam_names":       _nyc_frames.get('cam_names',[]),
                    "nyc_nearest":         _nyc_frames.get('nearest_cam',''),
                    "nyc_source":          _nyc_frames.get('source',''),
                    "dispatch_stage":      getattr(_chain,"_stage",0) if "_chain" in dir() else 0,
                    "lidar_active":        False,
                }
                _req.post("http://127.0.0.1:8000/push",
                          json=_dp, timeout=0.5)
            except Exception as _pe:
                print(f"[PUSH ERROR] {type(_pe).__name__}: {_pe}", flush=True)
            time.sleep(0.4)

    finally:
        # ── clean shutdown ────────────────────────────────────────────────────
        if webcam is not None:
            try:
                webcam.close()
                print(f"{DM}[webcam] Closed.{R}")
            except Exception:
                pass

        if hasattr(veh, "close"):
            try:
                veh.close()
            except Exception:
                pass

    print(f"\n{G}{B}Run complete.{R}\n")


if __name__ == "__main__":
    main()
