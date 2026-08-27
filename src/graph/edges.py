def route_after_intake(state):
    """Route after intake validation."""
    if state.get("state") == "intake_failed":
        return "error_step"
    return "transcribe_step"


def route_after_transcription(state):
    """Route after transcription to injection check."""
    if state.get("state") == "transcription_failed":
        return "error_step"
    return "injection_check_step"


def route_after_injection_check(state):
    """Route after injection check."""
    if state.get("state") == "flagged_for_review":
        return "error_step"
    return "pii_redact_step"


def route_after_qa(state):
    """Route after QA to determine next node."""
    if state.get("error"):
        return "error_step"
    compliance_flags = state.get("compliance_flags", [])
    for flag in compliance_flags:
        if flag.get("severity") == "critical":
            return "supervisor_step"
    return "report_step"
