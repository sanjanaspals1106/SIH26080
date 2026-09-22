"""Historical Analog Finder package (F3).

PRD Section 15 (F3) & Appendix B.
Finds top-5 analogous weather states across development seasons.
"""

from regime_engine.analogs.finder import AnalogFinder, ANALOG_VECTOR_KEYS

__all__ = [
    "AnalogFinder",
    "ANALOG_VECTOR_KEYS",
]
