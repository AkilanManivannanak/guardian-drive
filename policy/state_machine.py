from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, MutableMapping, Optional


def _score(task_obj: object) -> float:
    if task_obj is None:
        return 0.0
    if isinstance(task_obj, Mapping):
        return float(task_obj.get("score", 0.0) or 0.0)
    return float(getattr(task_obj, "score", 0.0) or 0.0)


def _details(task_obj: object) -> Dict[str, object]:
    if task_obj is None:
        return {}
    if isinstance(task_obj, Mapping):
        return dict(task_obj.get("details", {}) or {})
    return dict(getattr(task_obj, "details", {}) or {})


@dataclass
class PolicyDecision:
    ts: float
    state: str
    route_kind: str
    reasons: List[str] = field(default_factory=list)
    summary: str = ""
    speak_keys: List[str] = field(default_factory=list)
    haptic_pattern: Optional[str] = None
    request_pull_over: bool = False
    notify_contact: bool = False
    notify_discord: bool = False
    prepare_dispatch: bool = False
    open_navigation: bool = False
    truth: Dict[str, object] = field(default_factory=dict)
    debug: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

    # Compatibility with the PolicyAction interface used by the replay / evaluation code.
    @property
    def level(self):
        from acquisition.models import AlertLevel
        return AlertLevel(_STATE_TO_LEVEL.get(self.state, 1))

    @property
    def log_reason(self) -> str:
        return "; ".join(self.reasons) if self.reasons else self.summary


_STATE_TO_LEVEL = {"nominal": 1, "advisory": 2, "caution": 3, "pull_over": 4, "escalate": 5}


def _risk_state_to_tasks(rs: object) -> Dict[str, object]:
    """Adapter: FusionEngine's RiskState dataclass -> the task mapping consumed by step()."""
    out: Dict[str, object] = {}
    arr = getattr(rs, "arrhythmia", None)
    if arr is not None and not getattr(arr, "abstained", False):
        cls = getattr(getattr(arr, "cls", None), "value", "")
        if cls in ("afib", "tachycardia", "bradycardia"):
            out["task_a"] = {"score": float(arr.confidence), "details": {"class": cls}}
    drw = getattr(rs, "drowsiness", None)
    if drw is not None and not getattr(drw, "abstained", False):
        out["task_b"] = {"score": float(drw.score), "details": dict(getattr(drw, "details", {}) or {})}
    cr = getattr(rs, "crash", None)
    if cr is not None and getattr(cr, "detected", False):
        sev = getattr(getattr(cr, "severity", None), "name", "NONE")
        factor = 1.0 if sev == "SEVERE" else 0.6  # a mild crash alone is not an emergency
        out["task_c"] = {"score": float(cr.confidence) * factor, "details": {"severity": sev}}
    return out


class SafetyStateMachine:
    """
    Conservative policy.

    Key changes vs a naive demo:
    - drowsiness alone does not jump to emergency fast
    - severe drowsiness requires stronger webcam evidence + persistence
    - drowsiness routes to rest stops, not ER
    - neurologic screen is treated as possible emergency, not diagnosis
    """

    def __init__(
        self,
        *,
        drowsy_advisory_score: float = 0.48,
        drowsy_caution_score: float = 0.66,
        drowsy_pull_score: float = 0.84,
        cardiac_emergency_score: float = 0.92,
        neuro_emergency_score: float = 0.74,
        crash_emergency_score: float = 0.90,
        caution_persist_sec: float = 5.0,
        pull_persist_sec: float = 12.0,
        step_period_sec: float = 1.0,
    ) -> None:
        self.drowsy_advisory_score = drowsy_advisory_score
        self.drowsy_caution_score = drowsy_caution_score
        self.drowsy_pull_score = drowsy_pull_score
        self.cardiac_emergency_score = cardiac_emergency_score
        self.neuro_emergency_score = neuro_emergency_score
        self.crash_emergency_score = crash_emergency_score
        self.caution_persist_steps = max(1, int(round(caution_persist_sec / step_period_sec)))
        self.pull_persist_steps = max(self.caution_persist_steps, int(round(pull_persist_sec / step_period_sec)))
        self._state = "nominal"
        self._counters: MutableMapping[str, int] = {
            "drowsy_caution": 0,
            "drowsy_pull": 0,
        }

    def _strong_webcam_drowsy(self, task_b: object) -> bool:
        d = _details(task_b)
        webcam_score = float(d.get("webcam_score", d.get("vision_score", 0.0)) or 0.0)
        perclos = float(d.get("perclos", 0.0) or 0.0)
        eyes_closed_sec = float(d.get("eyes_closed_sec", d.get("closure_sec", 0.0)) or 0.0)
        yawn_score = float(d.get("yawn_score", 0.0) or 0.0)
        posture_score = float(d.get("posture_score", 0.0) or 0.0)

        # Posture alone should not be enough.
        if webcam_score >= 0.78 and (perclos >= 0.32 or eyes_closed_sec >= 2.5 or yawn_score >= 0.72):
            return True
        if webcam_score >= 0.85 and posture_score >= 0.55 and perclos >= 0.25:
            return True
        return False

    def step(self, rs: object) -> PolicyDecision:
        now = time.time()
        if not isinstance(rs, Mapping):
            rs = _risk_state_to_tasks(rs)
        task_a = rs.get("task_a")
        task_b = rs.get("task_b")
        task_c = rs.get("task_c")
        task_d = rs.get("task_d")

        cardiac = _score(task_a)
        drowsy = _score(task_b)
        crash = _score(task_c)
        neuro = _score(task_d)

        reasons: List[str] = []
        speak_keys: List[str] = []
        haptic: Optional[str] = None
        route_kind = "none"
        notify_contact = False
        notify_discord = False
        prepare_dispatch = False
        open_navigation = False
        request_pull_over = False
        summary = "Monitoring."

        if crash >= self.crash_emergency_score:
            self._state = "escalate"
            reasons.append(f"crash score {crash:.2f} >= {self.crash_emergency_score:.2f}")
            speak_keys = ["crash_emergency"]
            haptic = "urgent_cardiac"
            route_kind = "er"
            notify_contact = True
            notify_discord = True
            prepare_dispatch = True
            open_navigation = True
            request_pull_over = True
            summary = "Possible crash emergency. Emergency workflow active."
        elif cardiac >= self.cardiac_emergency_score:
            self._state = "escalate"
            reasons.append(f"cardiac score {cardiac:.2f} >= {self.cardiac_emergency_score:.2f}")
            speak_keys = ["cardiac_urgent"]
            haptic = "urgent_cardiac"
            route_kind = "er"
            notify_contact = True
            notify_discord = True
            prepare_dispatch = True
            open_navigation = True
            request_pull_over = True
            summary = "Severe cardiac risk. Pull over and route to emergency care."
        elif neuro >= self.neuro_emergency_score:
            self._state = "escalate"
            reasons.append(f"neuro score {neuro:.2f} >= {self.neuro_emergency_score:.2f}")
            speak_keys = ["neuro_urgent"]
            haptic = "neurologic_emergency"
            route_kind = "er"
            notify_contact = True
            notify_discord = True
            prepare_dispatch = True
            open_navigation = True
            request_pull_over = True
            summary = "Possible neurologic emergency. Emergency guidance active."
        else:
            strong_webcam = self._strong_webcam_drowsy(task_b)

            if drowsy >= self.drowsy_caution_score:
                self._counters["drowsy_caution"] += 1
            else:
                self._counters["drowsy_caution"] = max(0, self._counters["drowsy_caution"] - 1)

            if drowsy >= self.drowsy_pull_score and strong_webcam:
                self._counters["drowsy_pull"] += 1
            else:
                self._counters["drowsy_pull"] = max(0, self._counters["drowsy_pull"] - 1)

            if self._counters["drowsy_pull"] >= self.pull_persist_steps:
                self._state = "pull_over"
                reasons.append("persistent severe drowsiness with strong webcam evidence")
                speak_keys = ["drowsy_pull_over"]
                haptic = "pull_over_drowsy"
                route_kind = "rest_stop"
                notify_discord = True
                request_pull_over = True
                open_navigation = True
                summary = "Persistent severe drowsiness. Pull over at the nearest safe stop."
            elif self._counters["drowsy_caution"] >= self.caution_persist_steps:
                self._state = "caution"
                reasons.append("persistent drowsiness above caution threshold")
                speak_keys = ["drowsy_caution"]
                haptic = "caution_drowsy"
                route_kind = "rest_stop"
                open_navigation = True
                summary = "Fatigue detected. Prepare for a safe rest stop."
            elif drowsy >= self.drowsy_advisory_score:
                self._state = "advisory"
                reasons.append("drowsiness advisory threshold crossed")
                speak_keys = ["drowsy_advisory"]
                route_kind = "rest_stop"
                summary = "Stay alert and plan a short break."
            else:
                self._state = "nominal"
                summary = "Monitoring."

        truth = {
            "cardiac_score": round(cardiac, 4),
            "drowsy_score": round(drowsy, 4),
            "crash_score": round(crash, 4),
            "neuro_score": round(neuro, 4),
            "routing_rule": route_kind,
            "state": self._state,
        }
        debug = {
            "drowsy_caution_counter": self._counters["drowsy_caution"],
            "drowsy_pull_counter": self._counters["drowsy_pull"],
            "drowsy_caution_needed": self.caution_persist_steps,
            "drowsy_pull_needed": self.pull_persist_steps,
            "task_b_details": _details(task_b),
            "task_d_details": _details(task_d),
        }

        return PolicyDecision(
            ts=now,
            state=self._state,
            route_kind=route_kind,
            reasons=reasons,
            summary=summary,
            speak_keys=speak_keys,
            haptic_pattern=haptic,
            request_pull_over=request_pull_over,
            notify_contact=notify_contact,
            notify_discord=notify_discord,
            prepare_dispatch=prepare_dispatch,
            open_navigation=open_navigation,
            truth=truth,
            debug=debug,
        )
