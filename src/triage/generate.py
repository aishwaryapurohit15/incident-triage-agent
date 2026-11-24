"""Synthetic operational incident generator.

Produces a realistic IT incident log for a mid-sized bank, including the
quality problems found in real ticketing systems: inconsistent category
labels, duplicate reports of the same outage, missing resolution notes,
recording errors in severity and invalid timestamps.

The hidden root cause and true duplicate links are written to a separate
ground-truth file so that clustering and duplicate detection can be scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_RAW, RANDOM_SEED, SERVICES, SEVERITIES, WINDOW_END, WINDOW_START

# ---------------------------------------------------------------------------
# Root-cause catalogue
# ---------------------------------------------------------------------------
# Each root cause carries: owning team, services it affects, base severity
# probabilities, median hours to resolve, probability of customer impact,
# lognormal loss parameters, probability it follows a change, Basel II event
# type, relative frequency, short-description and description templates, and
# resolution note templates.

ROOT_CAUSES = {
    "certificate_expiry": dict(
        team="Identity and Security Operations",
        services=["Payments Gateway", "Online Banking Portal", "Mobile Banking App", "Single Sign-On", "Card Authorization"],
        sev=[0.30, 0.40, 0.25, 0.05], mttr=3.0, impact=0.75, loss=(7.6, 1.9), change=0.05,
        basel="Business disruption and system failures", weight=0.035,
        short=["SSL certificate expired on {host}", "TLS handshake failures on {service}", "Certificate error when connecting to {service}",
               "Browser shows certificate warning for {service}", "Expired cert breaking API calls to {service}"],
        desc=["Clients receive TLS handshake errors when calling {host}. The certificate appears to have expired at {time}.",
              "Users see a 'your connection is not private' warning on {service}. Certificate chain validation is failing.",
              "API consumers report SSL errors from {host}. Monitoring shows certificate expiry alerts.",
              "Partner integrations to {service} are failing with x509 certificate has expired errors."],
        notes=["Renewed and deployed the certificate on {host}. Added the certificate to the expiry monitoring dashboard.",
               "Certificate rotated and the load balancer reloaded. Root cause: renewal reminder went to a former employee."],
    ),
    "db_connection_pool": dict(
        team="Database Administration",
        services=["Core Banking Ledger", "Payments Gateway", "Trade Settlement", "Online Banking Portal"],
        sev=[0.20, 0.45, 0.30, 0.05], mttr=2.5, impact=0.6, loss=(7.2, 1.8), change=0.15,
        basel="Business disruption and system failures", weight=0.06,
        short=["Database connection pool exhausted on {service}", "Timeouts connecting to {service} database",
               "{service} slow, DB connections maxed out", "Too many connections error on {db}", "Intermittent DB timeouts for {service}"],
        desc=["Application logs show 'unable to acquire connection from pool' on {service}. Response times above {n} seconds.",
              "The {db} instance hit the max connections limit. {service} transactions are queuing and timing out.",
              "Users report {service} hanging. DBA dashboard shows active sessions at 100% of the pool.",
              "Long-running queries are holding connections on {db}, causing timeouts for {service} users."],
        notes=["Killed blocking sessions on {db} and raised the pool size from 200 to 300. Query fix raised with the development team.",
               "Restarted the application tier to release leaked connections. Connection leak fixed in the next release."],
    ),
    "failed_deployment": dict(
        team="Digital Channels Support",
        services=["Mobile Banking App", "Online Banking Portal", "Payments Gateway", "Card Authorization", "Customer Contact Center"],
        sev=[0.15, 0.40, 0.35, 0.10], mttr=2.0, impact=0.65, loss=(7.0, 2.0), change=0.92,
        basel="Execution, delivery and process management", weight=0.08,
        short=["{service} errors after last night's release", "Deployment broke login on {service}", "HTTP 500 errors on {service} after change {chg}",
               "Rollback needed for {service} release {ver}", "New version of {service} failing health checks"],
        desc=["Since release {ver} was deployed, {service} returns HTTP 500 errors for about {n}% of requests.",
              "Change {chg} was implemented during the weekend window. Users now cannot complete transactions on {service}.",
              "Health checks for {service} are failing after the configuration change. Error rate jumped from 0.1% to {n}%.",
              "A feature flag enabled in release {ver} is causing a null pointer exception in {service}."],
        notes=["Rolled back release {ver}. Defect logged and the fix is scheduled for the next change window.",
               "Configuration value reverted under emergency change. Post-implementation review requested."],
    ),
    "vpn_gateway": dict(
        team="Network Operations",
        services=["Corporate VPN"],
        sev=[0.05, 0.35, 0.45, 0.15], mttr=3.5, impact=0.05, loss=(5.5, 1.7), change=0.2,
        basel="Business disruption and system failures", weight=0.06,
        short=["VPN not connecting", "Remote staff cannot connect to VPN", "VPN gateway {region} down", "VPN keeps dropping every few minutes",
               "Cannot reach internal sites over VPN"],
        desc=["Multiple remote employees in {region} cannot establish a VPN tunnel. The client times out at authentication.",
              "VPN sessions drop every few minutes for users connected to the {region} gateway.",
              "Monitoring shows the {region} VPN concentrator CPU at 100%. Around {n} users are unable to connect.",
              "Staff working from home report the VPN client stuck on 'connecting'."],
        notes=["Failed over to the secondary VPN concentrator in {region}. Hardware fault logged with the vendor.",
               "Increased the license pool and restarted the VPN service."],
    ),
    "disk_full": dict(
        team="Data Platform Engineering",
        services=["Enterprise Data Warehouse", "Core Banking Ledger", "Trade Settlement", "HR and Payroll"],
        sev=[0.05, 0.30, 0.45, 0.20], mttr=1.5, impact=0.15, loss=(5.8, 1.8), change=0.05,
        basel="Business disruption and system failures", weight=0.07,
        short=["Disk space full on {host}", "{host} out of disk space", "Filesystem /data at 100% on {host}", "Low disk alert on {service} server",
               "Write failures due to no space left on device"],
        desc=["Monitoring alert: /data on {host} at 100%. {service} jobs are failing with 'no space left on device'.",
              "Log files have filled the disk on {host}. The {service} application cannot write temporary files.",
              "Disk usage on {host} grew {n}% overnight. Database writes are failing.",
              "Batch loads into {service} failed because the staging volume is full."],
        notes=["Archived old logs and extended the volume by 500 GB. Log rotation policy corrected.",
               "Cleared temporary files and added a 90% disk usage alert."],
    ),
    "third_party_api": dict(
        team="Third-Party Vendor Management",
        services=["Payments Gateway", "Card Authorization", "Customer Contact Center", "Online Banking Portal", "Mobile Banking App"],
        sev=[0.25, 0.40, 0.28, 0.07], mttr=4.0, impact=0.8, loss=(7.8, 2.0), change=0.03,
        basel="Business disruption and system failures", weight=0.06,
        short=["{vendor} API outage affecting {service}", "Third-party provider {vendor} not responding", "{service} failing - {vendor} returning 503",
               "Vendor {vendor} degraded performance", "Card network timeouts via {vendor}"],
        desc=["Calls to {vendor} are returning HTTP 503. {service} transactions are failing for customers.",
              "{vendor} status page reports a major incident. Our {service} integration is degraded.",
              "Latency from {vendor} increased to {n} seconds. {service} timeouts are rising.",
              "The external provider {vendor} is rejecting requests from our {service} integration."],
        notes=["{vendor} resolved the outage on their side. Vendor incident report requested under the SLA.",
               "Traffic switched to the backup provider until {vendor} recovered. Vendor scorecard updated."],
    ),
    "batch_failure": dict(
        team="Data Platform Engineering",
        services=["Enterprise Data Warehouse", "Trade Settlement", "Core Banking Ledger", "HR and Payroll"],
        sev=[0.10, 0.35, 0.40, 0.15], mttr=5.0, impact=0.3, loss=(6.6, 1.9), change=0.1,
        basel="Execution, delivery and process management", weight=0.07,
        short=["Overnight batch job {job} failed", "End-of-day reconciliation did not complete", "{job} abended with return code 12",
               "Settlement file not generated", "Nightly ETL for {service} failed"],
        desc=["The overnight job {job} failed at step {n}. Downstream {service} reports are missing this morning.",
              "End-of-day reconciliation for {service} did not finish. Breaks are reported in the ledger.",
              "Scheduler shows {job} in an error state. The input file from upstream arrived late and was incomplete.",
              "The {service} ETL pipeline failed with a schema mismatch on the source extract."],
        notes=["Reran {job} after the upstream file was corrected. Added a file completeness check before load.",
               "Fixed the schema mapping and reprocessed the missing day."],
    ),
    "dns_failure": dict(
        team="Network Operations",
        services=["Online Banking Portal", "Email and Collaboration", "Payments Gateway", "Single Sign-On", "Mobile Banking App"],
        sev=[0.25, 0.40, 0.25, 0.10], mttr=1.5, impact=0.55, loss=(7.0, 1.8), change=0.3,
        basel="Business disruption and system failures", weight=0.03,
        short=["DNS resolution failing for {service}", "Cannot resolve {host}", "Name lookup errors across multiple apps",
               "{service} unreachable - DNS issue", "Intermittent NXDOMAIN responses"],
        desc=["Servers cannot resolve {host}. Several services including {service} are unreachable by name.",
              "Internal DNS servers are returning SERVFAIL. Applications that depend on {service} are failing.",
              "After the DNS zone update, {service} records are missing and users cannot reach the site.",
              "Intermittent DNS timeouts are causing failed requests to {service}."],
        notes=["Restored the missing DNS records and flushed resolver caches.",
               "Restarted the failed DNS server and rebalanced the resolvers."],
    ),
    "sso_outage": dict(
        team="Identity and Security Operations",
        services=["Single Sign-On", "Online Banking Portal", "Email and Collaboration", "HR and Payroll"],
        sev=[0.20, 0.45, 0.28, 0.07], mttr=2.0, impact=0.5, loss=(6.8, 1.8), change=0.25,
        basel="Business disruption and system failures", weight=0.045,
        short=["Users cannot log in via SSO", "Single sign-on login loop", "MFA prompts not arriving", "Authentication service down for {service}",
               "SAML login errors on {service}"],
        desc=["Users are stuck in a redirect loop when logging in to {service} through SSO.",
              "MFA push notifications are not being delivered. Around {n} staff cannot authenticate.",
              "The identity provider is returning SAML assertion errors for {service}.",
              "Login to {service} fails with 'invalid token' after the identity provider upgrade."],
        notes=["Identity provider signing key was out of sync. Metadata refreshed on all service providers.",
               "MFA provider issue resolved. Fallback authentication method documented."],
    ),
    "memory_leak": dict(
        team="Payments Application Support",
        services=["Payments Gateway", "Card Authorization", "Trade Settlement", "Core Banking Ledger"],
        sev=[0.15, 0.40, 0.35, 0.10], mttr=3.0, impact=0.55, loss=(7.1, 1.9), change=0.2,
        basel="Business disruption and system failures", weight=0.06,
        short=["{service} crashing with OutOfMemoryError", "{service} JVM restarts every few hours", "Memory usage climbing on {host}",
               "{service} pods in CrashLoopBackOff", "Payment processing service unresponsive"],
        desc=["{service} nodes are crashing with java.lang.OutOfMemoryError. Heap usage climbs steadily until restart.",
              "Kubernetes reports {service} pods restarting {n} times in the last hour due to OOMKilled.",
              "Memory usage on {host} increases about 2% per hour. The service becomes unresponsive around 95%.",
              "Payment authorizations are timing out while {service} performs long garbage collection pauses."],
        notes=["Scheduled rolling restarts as a workaround. Memory leak traced to a cache with no eviction and fixed.",
               "Increased the heap and deployed a patch for the leaking session cache."],
    ),
    "data_feed": dict(
        team="Data Platform Engineering",
        services=["Enterprise Data Warehouse", "Trade Settlement", "Core Banking Ledger"],
        sev=[0.05, 0.30, 0.45, 0.20], mttr=6.0, impact=0.2, loss=(6.4, 2.0), change=0.1,
        basel="Execution, delivery and process management", weight=0.05,
        short=["Market data feed delayed", "Incorrect values in {service} feed", "Data feed from {vendor} missing records",
               "Duplicate records in today's load", "Stale prices in {service}"],
        desc=["The daily feed from {vendor} arrived {n} minutes late. {service} reports are using yesterday's values.",
              "Reconciliation found {n} records missing from the {service} load compared with the source.",
              "Prices in {service} have not updated since the morning. The upstream feed appears stale.",
              "Today's file contains duplicate trade records, inflating totals in {service}."],
        notes=["Reloaded the feed after the provider resent the file. Added row count reconciliation.",
               "Deduplicated the load and added a primary key constraint on the staging table."],
    ),
    "phishing_alert": dict(
        team="Identity and Security Operations",
        services=["Email and Collaboration", "Single Sign-On"],
        sev=[0.10, 0.35, 0.40, 0.15], mttr=4.0, impact=0.1, loss=(6.9, 2.2), change=0.0,
        basel="External fraud", weight=0.05,
        short=["Suspicious phishing email reported", "Possible credential compromise for user account", "Phishing campaign targeting staff",
               "User clicked link in suspicious email", "Mailbox sending spam to customers"],
        desc=["Several employees received an email impersonating IT asking them to reset passwords via an external link.",
              "A user entered credentials on a fake login page. Unusual sign-ins detected from a foreign IP address.",
              "Security monitoring flagged {n} inbound emails with a malicious attachment.",
              "A staff mailbox is sending unsolicited emails to customers. The account may be compromised."],
        notes=["Blocked the sender domain, purged the emails and reset credentials for affected users.",
               "Account disabled, sessions revoked and MFA re-registered. Awareness note sent to staff."],
    ),
    "end_user": dict(
        team="End-User Computing",
        services=["Email and Collaboration", "HR and Payroll", "Customer Contact Center", "Corporate VPN"],
        sev=[0.00, 0.02, 0.28, 0.70], mttr=8.0, impact=0.02, loss=(4.0, 1.6), change=0.02,
        basel="Business disruption and system failures", weight=0.20,
        short=["Laptop running slow", "Password reset request", "Printer not working on floor {n}", "Outlook not syncing", "Need access to shared drive",
               "Headset not working for contact center agent", "Teams camera not detected", "Cannot open payslip in HR portal", "Monitor flickering"],
        desc=["My laptop has been very slow since this morning and applications keep freezing.",
              "I am locked out of my account and need a password reset.",
              "The printer on floor {n} shows an error and will not print.",
              "Outlook stopped syncing emails on my laptop. Webmail works fine.",
              "Please grant me access to the shared finance drive. My manager has approved.",
              "My headset is not recognized by the softphone, so I cannot take customer calls.",
              "Cannot download my payslip from the HR portal. The page keeps loading."],
        notes=["Password reset and the user confirmed access.", "Replaced the faulty hardware.", "Cleared the local cache and re-synced the profile.",
               "Access granted after approval was verified.", "Reinstalled the driver and the device now works."],
    ),
    "network_latency": dict(
        team="Network Operations",
        services=["Trade Settlement", "Payments Gateway", "Customer Contact Center", "Core Banking Ledger", "Corporate VPN"],
        sev=[0.10, 0.35, 0.40, 0.15], mttr=3.0, impact=0.4, loss=(6.5, 1.9), change=0.15,
        basel="Business disruption and system failures", weight=0.05,
        short=["High latency between data centers", "Packet loss on {region} link", "{service} slow due to network", "Calls dropping in contact center",
               "WAN circuit {region} degraded"],
        desc=["Latency between the primary and secondary data centers increased to {n} ms. {service} replication is lagging.",
              "Network monitoring shows {n}% packet loss on the {region} WAN link.",
              "Contact center calls are dropping and the voice quality is poor. Jitter is high on the {region} circuit.",
              "{service} response times doubled. The network team sees congestion on the core switch."],
        notes=["Carrier repaired a fiber cut on the {region} circuit. Traffic rerouted during the outage.",
               "Replaced a faulty line card on the core switch."],
    ),
    "storage_replication": dict(
        team="Database Administration",
        services=["Core Banking Ledger", "Trade Settlement", "Enterprise Data Warehouse"],
        sev=[0.15, 0.40, 0.35, 0.10], mttr=4.5, impact=0.25, loss=(6.7, 1.9), change=0.1,
        basel="Business disruption and system failures", weight=0.03,
        short=["Database replication lag on {db}", "Standby database out of sync", "Storage array alert on {host}", "Replica {db} behind by hours",
               "Disaster recovery copy not updating"],
        desc=["The standby for {db} is {n} minutes behind the primary. Recovery point objective is at risk.",
              "Storage array {host} reports degraded RAID. Write latency is high for {service}.",
              "Log shipping to the DR site for {service} stopped overnight.",
              "Replication from {db} to the reporting replica failed with a checksum error."],
        notes=["Rebuilt the replica from a fresh snapshot and resumed replication.",
               "Replaced the failed disk and the array rebuilt successfully."],
    ),
    "maintenance_overrun": dict(
        team="Payments Application Support",
        services=["Payments Gateway", "Core Banking Ledger", "Card Authorization", "Trade Settlement"],
        sev=[0.10, 0.35, 0.40, 0.15], mttr=2.5, impact=0.5, loss=(6.8, 1.8), change=0.95,
        basel="Execution, delivery and process management", weight=0.035,
        short=["Planned maintenance overran into business hours", "{service} still unavailable after maintenance window", "Change {chg} not completed on time",
               "Services not restored after patching", "Maintenance on {service} running late"],
        desc=["The maintenance window for change {chg} ended at 06:00 but {service} is still unavailable.",
              "Patching on {host} took longer than planned. {service} did not come back online before opening.",
              "The database upgrade for {service} is still running and the service is unavailable to customers.",
              "Post-maintenance validation failed for {service}. The implementation team is still working on it."],
        notes=["Change completed and services validated. The window was too short; future changes need a longer window.",
               "Backed out the change and restored service. Rescheduled with a revised plan."],
    ),
}

# Messy category labels used by different reporters for the same thing.
CATEGORY_VARIANTS = {
    "certificate_expiry": ["Security", "security", "Certificates", "SEC"],
    "db_connection_pool": ["Database", "database", "DB", "Databse"],
    "failed_deployment": ["Application", "application", "App", "Software", "Change"],
    "vpn_gateway": ["Network", "network", "NW", "Netwrk", "Connectivity"],
    "disk_full": ["Infrastructure", "infra", "Storage", "Server"],
    "third_party_api": ["Vendor", "Third Party", "3rd party", "External", "Application"],
    "batch_failure": ["Batch", "Data", "batch job", "Application"],
    "dns_failure": ["Network", "network", "DNS", "Connectivity"],
    "sso_outage": ["Access", "Security", "Login", "access"],
    "memory_leak": ["Application", "app", "Performance", "Software"],
    "data_feed": ["Data", "data", "Feeds", "Batch"],
    "phishing_alert": ["Security", "security", "Phishing", "SEC", "Email"],
    "end_user": ["Hardware", "Desktop", "Access", "Email", "Other", "hardware"],
    "network_latency": ["Network", "network", "NW", "Performance", "Telecom"],
    "storage_replication": ["Database", "Storage", "Infrastructure", "DB"],
    "maintenance_overrun": ["Change", "change", "Application", "Maintenance"],
}

HOSTS = ["prd-app-01", "prd-app-07", "pay-gw-02", "cbl-db-03", "edw-etl-11", "auth-idp-01", "web-fe-04", "trd-set-02", "hr-app-02", "api.bank.example"]
DBS = ["LEDGERPRD", "PAYDB01", "SETTLEDB", "EDWPRD", "CRMDB02"]
VENDORS = ["CardNet", "PayFlow", "MarketLink", "TeleVox", "CloudMail", "FinData"]
REGIONS = ["US-East", "US-West", "London", "Singapore", "Frankfurt"]
JOBS = ["EOD_RECON_01", "GL_POST_NIGHTLY", "SETTLE_T1_BATCH", "EDW_LOAD_CUST", "PAYROLL_CALC"]
OPENERS = ["", "", "", "Hi team, ", "Hello, ", "URGENT: ", "Hi, ", "FYI - ", "Please help: "]
CLOSERS = ["", "", "", " Please advise.", " Customers are complaining.", " Thanks.", " This is impacting business.", " Please look asap."]
CHANNELS = ["Monitoring Alert", "Service Portal", "Email", "Phone"]

# Impact statements reporters add. They carry most of the severity signal, with noise.
IMPACT_PHRASES = {
    "SEV-1": ["All customers are affected.", "Complete outage, no transactions are going through.", "Customers cannot pay or log in at all.",
              "Thousands of customers impacted, social media complaints rising.", "Regulatory reporting deadline at risk today.",
              "Total loss of service across all channels.", "Major incident declared, bridge call opened."],
    "SEV-2": ["Many users are affected.", "Service is degraded for a large group of customers.", "About {n}% of transactions are failing.",
              "Several branches are reporting the same problem.", "Partial outage, some customers can still complete transactions.",
              "Business is impacted but there is a slow workaround."],
    "SEV-3": ["A few users are affected.", "A workaround is available.", "Internal users only, no customer impact so far.",
              "Intermittent, not blocking work.", "Impact is limited to one team.", "Can wait until business hours."],
    "SEV-4": ["Only affects me.", "No business impact.", "Low priority, whenever someone is free.", "Single user request.",
              "Not urgent.", "Cosmetic issue only."],
}


def _fill(template: str, rng: np.random.Generator, service: str, vendor: str | None = None) -> str:
    return template.format(
        service=service,
        host=rng.choice(HOSTS),
        db=rng.choice(DBS),
        vendor=vendor or rng.choice(VENDORS),
        region=rng.choice(REGIONS),
        job=rng.choice(JOBS),
        chg=f"CHG{rng.integers(10000, 99999)}",
        ver=f"{rng.integers(3, 9)}.{rng.integers(0, 20)}.{rng.integers(0, 9)}",
        n=int(rng.integers(2, 60)),
        time=f"{rng.integers(0, 24):02d}:{rng.choice(['00', '15', '30', '45'])}",
    )


def _typo(text: str, rng: np.random.Generator, rate: float) -> str:
    """Introduce realistic typing noise: swapped and dropped characters."""
    words = text.split(" ")
    for i, w in enumerate(words):
        if len(w) > 4 and rng.random() < rate:
            j = int(rng.integers(1, len(w) - 2))
            if rng.random() < 0.5:
                w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
            else:
                w = w[:j] + w[j + 1:]
            words[i] = w
    return " ".join(words)


def _messy(text: str, rng: np.random.Generator, closer: bool = True) -> str:
    text = rng.choice(OPENERS) + text + (rng.choice(CLOSERS) if closer else "")
    if rng.random() < 0.15:
        text = text.lower()
    replacements = {"database": "db", "production": "prod", "please": "pls", "application": "app"}
    if rng.random() < 0.3:
        for k, v in replacements.items():
            text = text.replace(k, v)
    return _typo(text, rng, rate=0.04)


def _daily_rate(dates: pd.DatetimeIndex) -> np.ndarray:
    """Expected incidents per day with weekly, monthly and trend effects."""
    base = np.full(len(dates), 21.0)
    base *= np.where(dates.dayofweek < 5, 1.15, 0.62)  # quieter weekends
    base *= np.where(dates.is_month_end | (dates.day <= 2), 1.25, 1.0)  # month-end close
    t = np.arange(len(dates)) / len(dates)
    base *= 1.0 + 0.12 * t  # gradual growth in digital volume
    return base


def generate(n_days: int | None = None, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    if n_days:
        dates = dates[:n_days]
    rates = _daily_rate(dates)
    causes = list(ROOT_CAUSES)

    # Regime shift: a new payments vendor (NovaPay) goes live in July 2026 and
    # raises third-party incidents. The monitoring module should detect it.
    shift_date = pd.Timestamp("2026-07-01")

    # A handful of major outage days with bursts of related tickets.
    outage_days = set(rng.choice(len(dates), size=10, replace=False).tolist())

    rows, truth = [], []
    inc_no = 1
    for d_idx, day in enumerate(dates):
        n = rng.poisson(rates[d_idx])
        w = np.array([ROOT_CAUSES[c]["weight"] for c in causes], dtype=float)
        if day >= shift_date:
            w[causes.index("third_party_api")] *= 3.0
            w[causes.index("failed_deployment")] *= 1.3
        w /= w.sum()
        day_causes = [(c, False) for c in rng.choice(causes, size=n, p=w)]
        if d_idx in outage_days:
            burst_cause = str(rng.choice(["failed_deployment", "third_party_api", "dns_failure", "db_connection_pool"]))
            day_causes += [(burst_cause, True)] * int(rng.integers(12, 30))
        burst = None  # (service, first incident id, opened time) for the major outage of the day

        for cause, in_burst in day_causes:
            spec = ROOT_CAUSES[cause]
            service = rng.choice(spec["services"])
            if in_burst and burst is not None:
                service = burst[0]
            crit = SERVICES[service]

            # Opening time: change-related incidents cluster after night windows.
            change_related = rng.random() < spec["change"]
            if change_related:
                hour = rng.choice([1, 2, 3, 6, 7, 8, 9], p=[0.1, 0.1, 0.1, 0.15, 0.2, 0.2, 0.15])
            else:
                hour = rng.choice(np.arange(24), p=_hour_profile())
            opened = day + pd.Timedelta(hours=int(hour), minutes=int(rng.integers(0, 60)))
            if in_burst and burst is not None:
                opened = burst[2] + pd.Timedelta(minutes=int(rng.integers(2, 180)))

            # True severity: base mix shifted by service criticality and business hours.
            p = np.array(spec["sev"], dtype=float)
            shift = {3: np.array([1.6, 1.25, 0.8, 0.5]), 2: np.ones(4), 1: np.array([0.4, 0.7, 1.15, 1.4])}[crit]
            if 8 <= hour <= 18 and opened.dayofweek < 5:
                shift = shift * np.array([1.2, 1.1, 1.0, 0.9])
            # Global calibration toward a realistic mix (few SEV-1 incidents).
            p = p * shift * np.array([0.35, 0.75, 1.1, 1.2])
            p /= p.sum()
            sev_idx = int(rng.choice(4, p=p))
            true_sev = SEVERITIES[sev_idx]

            # Recorded severity has human recording error (about 9% off by one).
            rec_idx = sev_idx
            if rng.random() < 0.09:
                rec_idx = int(np.clip(sev_idx + rng.choice([-1, 1]), 0, 3))
            recorded_sev = SEVERITIES[rec_idx]

            # Time to resolve: lognormal around the cause median, faster for SEV-1.
            sev_speed = [0.55, 0.8, 1.0, 1.6][sev_idx]
            mttr_h = float(rng.lognormal(np.log(spec["mttr"] * sev_speed), 0.75))
            resolved = opened + pd.Timedelta(hours=mttr_h)

            impact = rng.random() < spec["impact"] * [1.2, 1.0, 0.7, 0.3][sev_idx]
            customers = int(rng.lognormal(6.5 - sev_idx, 1.2)) if impact else 0
            # Only some customer-impacting incidents produce a quantified financial loss.
            has_loss = impact and rng.random() < 0.45
            loss = float(rng.lognormal(*spec["loss"]) * [3.0, 1.4, 0.6, 0.2][sev_idx]) if has_loss else 0.0

            vendor = "NovaPay" if (cause == "third_party_api" and day >= shift_date and rng.random() < 0.6) else None
            short = _messy(_fill(str(rng.choice(spec["short"])), rng, service, vendor), rng, closer=False)
            body = _fill(str(rng.choice(spec["desc"])), rng, service, vendor)
            # Most reporters describe the impact; about 20% of the time it hints at a neighbouring level.
            if rng.random() < 0.8:
                hint_idx = sev_idx if rng.random() < 0.8 else int(np.clip(sev_idx + rng.choice([-1, 1]), 0, 3))
                body += " " + _fill(str(rng.choice(IMPACT_PHRASES[SEVERITIES[hint_idx]])), rng, service)
            desc = _messy(body, rng)
            # Some reporters write vague tickets with almost no signal.
            if rng.random() < 0.06:
                desc = rng.choice(["Not working.", "System is down, please check.", "Issue with " + service + ".", "Getting errors.", "slow"])

            note = _fill(str(rng.choice(spec["notes"])), rng, service)
            team = spec["team"]
            # Misrouting: about 7% of tickets are assigned to a plausible wrong team.
            if rng.random() < 0.07:
                team = str(rng.choice([t for t in _team_list() if t != team]))

            iid = f"INC{inc_no:07d}"
            inc_no += 1
            channel = "Monitoring Alert" if (cause in {"disk_full", "storage_replication", "memory_leak"} and rng.random() < 0.6) else str(
                rng.choice(CHANNELS, p=[0.2, 0.4, 0.25, 0.15])
            )
            rows.append(dict(
                incident_id=iid, opened_at=opened, resolved_at=resolved, channel=channel, service=service,
                category=str(rng.choice(CATEGORY_VARIANTS[cause])), short_description=short, description=desc,
                assigned_team=team, severity=recorded_sev, customer_impact=bool(impact), customers_affected=customers,
                financial_loss_usd=round(loss, 2), change_related=bool(change_related), resolution_notes=note,
                basel_event_type=spec["basel"], reopened=bool(rng.random() < 0.04),
            ))
            parent = None
            if in_burst:
                if burst is None:
                    burst = (service, iid, opened)
                else:
                    parent = burst[1]
            truth.append(dict(incident_id=iid, root_cause=cause, true_severity=true_sev, true_team=spec["team"], duplicate_of=parent))
            if parent:  # follow-on reports of a major outage carry no separate loss
                rows[-1]["financial_loss_usd"], rows[-1]["customers_affected"] = 0.0, 0

            # Duplicate reports: the same outage reported again within hours.
            dup_p = 0.10 if impact and sev_idx <= 1 else 0.025
            for _ in range(int(rng.random() < dup_p) + int(rng.random() < dup_p / 3)):
                dup_open = opened + pd.Timedelta(minutes=int(rng.integers(3, 240)))
                did = f"INC{inc_no:07d}"
                inc_no += 1
                rows.append(dict(
                    incident_id=did, opened_at=dup_open, resolved_at=resolved, channel=str(rng.choice(CHANNELS[1:])), service=service,
                    category=str(rng.choice(CATEGORY_VARIANTS[cause])),
                    short_description=_messy(_fill(str(rng.choice(spec["short"])), rng, service), rng, closer=False),
                    description=_messy(_fill(str(rng.choice(spec["desc"])), rng, service), rng),
                    assigned_team=team, severity=recorded_sev, customer_impact=bool(impact), customers_affected=0,
                    financial_loss_usd=0.0, change_related=bool(change_related), resolution_notes="Duplicate of " + iid,
                    basel_event_type=spec["basel"], reopened=False,
                ))
                truth.append(dict(incident_id=did, root_cause=cause, true_severity=true_sev, true_team=spec["team"], duplicate_of=iid))

    df = pd.DataFrame(rows).sort_values("opened_at").reset_index(drop=True)
    gt = pd.DataFrame(truth)

    # Data-quality defects typical of ticketing exports.
    n = len(df)
    idx = rng.choice(n, size=int(0.22 * n), replace=False)
    df.loc[idx, "resolution_notes"] = np.nan  # missing resolution notes
    idx = rng.choice(n, size=int(0.03 * n), replace=False)
    df.loc[idx, "assigned_team"] = np.nan  # unassigned tickets
    idx = rng.choice(n, size=int(0.004 * n), replace=False)
    df.loc[idx, "resolved_at"] = df.loc[idx, "opened_at"] - pd.Timedelta(hours=2)  # resolved before opened
    idx = rng.choice(n, size=int(0.01 * n), replace=False)
    df.loc[idx, "resolved_at"] = pd.NaT  # still open or never closed
    # Exact export duplicates (same row twice).
    extra = df.sample(n=int(0.005 * n), random_state=seed)
    df = pd.concat([df, extra]).sort_values("opened_at").reset_index(drop=True)
    return df, gt


def _hour_profile() -> np.ndarray:
    hours = np.arange(24)
    prof = 0.3 + np.exp(-((hours - 10.5) ** 2) / 18) + 0.6 * np.exp(-((hours - 15) ** 2) / 10)
    return prof / prof.sum()


def _team_list() -> list[str]:
    from .config import TEAMS

    return TEAMS


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df, gt = generate()
    df.to_csv(DATA_RAW / "incidents.csv", index=False)
    gt.to_csv(DATA_RAW / "ground_truth.csv", index=False)
    print(f"Wrote {len(df):,} incident rows and {len(gt):,} ground-truth rows to {DATA_RAW}")


if __name__ == "__main__":
    main()
