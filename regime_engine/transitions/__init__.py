"""Regime Transition Detection package (F4).

PRD Section 15 (F4) & Appendix B.
Detects monsoon phase transitions and low-pressure system life cycle events.
"""

from regime_engine.transitions.detector import TransitionDetector, PHASES

__all__ = [
    "TransitionDetector",
    "PHASES",
]
