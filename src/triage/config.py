"""Shared constants for the incident triage project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
DOCS = ROOT / "docs"

RANDOM_SEED = 42

# Observation window for the synthetic incident log.
WINDOW_START = "2024-11-01"
WINDOW_END = "2026-10-31"

# Time-based split: train on the first 18 months, test on the last 6.
TEST_START = "2026-05-01"
VALID_START = "2026-02-01"  # model selection on Feb to Apr 2026
VENDOR_SHIFT = "2026-07-01"  # NovaPay goes live (used only for reporting)

SEVERITIES = ["SEV-1", "SEV-2", "SEV-3", "SEV-4"]
SEVERITY_NAMES = {
    "SEV-1": "Critical",
    "SEV-2": "High",
    "SEV-3": "Medium",
    "SEV-4": "Low",
}

# Resolution targets (hours) used to measure SLA breaches.
SLA_HOURS = {"SEV-1": 4, "SEV-2": 8, "SEV-3": 24, "SEV-4": 72}

TEAMS = [
    "Network Operations",
    "Database Administration",
    "Payments Application Support",
    "Digital Channels Support",
    "Identity and Security Operations",
    "End-User Computing",
    "Data Platform Engineering",
    "Third-Party Vendor Management",
]

SERVICES = {
    # service: business criticality (1 = low, 3 = high)
    "Payments Gateway": 3,
    "Card Authorization": 3,
    "Core Banking Ledger": 3,
    "Trade Settlement": 3,
    "Online Banking Portal": 2,
    "Mobile Banking App": 2,
    "Single Sign-On": 2,
    "Customer Contact Center": 2,
    "Enterprise Data Warehouse": 1,
    "Corporate VPN": 1,
    "Email and Collaboration": 1,
    "HR and Payroll": 1,
}

# Cost of under-triage versus over-triage, used for cost-weighted evaluation.
# Row = true severity, column = predicted severity.
MISCLASSIFICATION_COST = [
    # SEV-1 SEV-2 SEV-3 SEV-4  (predicted)
    [0, 20, 60, 100],  # true SEV-1
    [3, 0, 15, 30],  # true SEV-2
    [2, 1, 0, 5],  # true SEV-3
    [2, 1, 0.5, 0],  # true SEV-4
]
