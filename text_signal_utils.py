"""
Lightweight copy of engagement-text helpers for offline unit tests.

The production implementation remains in llm_analyzer.py (unchanged API).
This module exists so tests can run without importing torch / transformers.
"""


def extract_engagement_level(analysis: str) -> str:
    a = analysis.lower()
    disengaged_keywords = [
        "using mobile",
        "on mobile",
        "using phone",
        "on the phone",
        "talking on phone",
        "talking on mobile",
        "sleeping",
        "dozing",
        "distracted",
        "not paying attention",
        "looking away",
        "looking at phone",
        "texting",
        "checking phone",
        "scrolling",
    ]
    strong_disengaged = any(
        kw in a
        for kw in ["using mobile", "on mobile", "using phone", "sleeping", "dozing"]
    )
    if strong_disengaged:
        return "DISENGAGED"

    assembly_keywords = [
        "assembling",
        "aligning",
        "screwing",
        "tightening",
        "connecting",
        "plugging",
        "inserting",
        "placing",
        "adjusting",
        "handling",
        "manipulating",
        "working on",
        "operating",
        "using tool",
    ]
    if any(kw in a for kw in assembly_keywords):
        if any(
            w in a
            for w in [
                "highly engaged",
                "very engaged",
                "completely focused",
                "fully engaged",
                "intensely focused",
                "deeply concentrated",
                "carefully",
                "precisely",
            ]
        ):
            return "HIGHLY_ENGAGED"
        if any(
            w in a
            for w in ["actively", "engaged", "focusing", "concentrating", "attentive"]
        ):
            return "ENGAGED"
        return "ENGAGED"

    if any(
        w in a
        for w in [
            "preparing",
            "organizing",
            "setting up",
            "arranging",
            "getting ready",
            "selecting",
            "reaching for",
            "positioning",
        ]
    ):
        return "PREPARING"

    if any(
        w in a
        for w in [
            "idle",
            "waiting",
            "resting",
            "paused",
            "not active",
            "not working",
            "standing by",
            "inactive",
            "stationary",
        ]
    ):
        return "IDLE"

    weak_disengaged = any(
        kw in a for kw in ["looking away", "distracted", "not paying attention"]
    )
    if weak_disengaged:
        count = sum(1 for kw in disengaged_keywords if kw in a)
        return "DISENGAGED" if count >= 2 else "ENGAGED"

    return "ENGAGED"
