import csv
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
NUM_OPERATORS = 50
OPERATOR_POOL = [fake.name() for _ in range(NUM_OPERATORS)]


FIELDS = [
    "Alarm timestamp","Alarm type","Alarm Point number","Alarm name",
    "Monitor number","Alarm direction","Mode","Acknowledge by",
    "Tag","Full comment","Assessment timestamp","Correct Tag"
]

ALARM_TYPES = ["Intrusion", "Sensor_Fault", "Tamper"]
MODES = ["Secured", "Unsecured", "Armed", "Disarmed"]
TAGS = ["Nuisance", "Valid", "False"]

# --- Scenario-driven truth (ground truth is defined here) ---
TAG_SCENARIOS = {
    # Nuisance = environmental / benign triggers for intrusion
    "Nuisance": {
        "alarm_types": ["Intrusion"],
        "comment_prefix": ["Rain", "Wind", "Animal", "Tree movement", "Storm"],
        "modes": ["Secured", "Armed"],
        "weight": 0.45,  # adjust base rate
    },

    # Valid = real incident OR real device problem (fault/tamper)
    "Valid": {
        "alarm_types": ["Intrusion", "Sensor_Fault", "Tamper"],
        "comment_prefix": [
            "Dispatch", "Police notified", "Door forced", "Verified",
            "Sensor fault confirmed", "Tamper confirmed", "Battery failure", "Maintenance"
        ],
        "modes": ["Secured", "Armed"],
        "weight": 0.45,
    },

    # False = operator assessed as false alarm / all clear
    "False": {
        "alarm_types": ["Intrusion"],  # keep False mostly for intrusion
        "comment_prefix": ["All In Order", "No issue found", "False alarm", "Site checked"],
        "modes": ["Secured", "Armed"],
        "weight": 0.10,
    },
}

# Confusions for intentional human label mistakes (operator mis-tag)
CONFUSIONS = {
    "Nuisance": ["False", "Valid"],
    "Valid": ["False", "Nuisance"],
    "False": ["Nuisance", "Valid"],
}

def fmt_ts(dt: datetime) -> str:
    return dt.strftime("%m/%d/%Y %I:%M:%S.") + f"{dt.microsecond//1000:03d} " + dt.strftime("%p")

def make_point_number(point_id: int) -> str:
    fdb = random.randint(1, 6)
    ipd = random.randint(100, 399)
    return f"{point_id}: FDB {fdb} - IPD {ipd}"

def make_alarm_name(alarm_type: str) -> str:
    # Keep your original “Sensor X Alarm Sector Y” style,
    # but you can tweak wording by alarm_type if desired.
    sensor_letter = chr(ord('A') + random.randint(0, 25))
    sector = random.randint(1, 60)

    if alarm_type == "Sensor_Fault":
        return f"Sensor {sensor_letter}  Fault  Sector {sector}"
    if alarm_type == "Tamper":
        return f"Sensor {sensor_letter}  Tamper  Sector {sector}"
    return f"Sensor {sensor_letter}  Alarm  Sector {sector}"

def make_station() -> str:
    return f"Operator Station {random.randint(1, 8)}"

def make_comment(prefix: str) -> str:
    who = random.choice(OPERATOR_POOL)
    return f"{prefix}  //{who}//"

def weighted_choice_tag():
    tags = list(TAG_SCENARIOS.keys())
    weights = [TAG_SCENARIOS[t]["weight"] for t in tags]
    return random.choices(tags, weights=weights, k=1)[0]

def enforce_faults_usually_valid(true_tag: str, alarm_type: str, rare_fault_false_rate: float) -> str:
    """
    Your domain rule: Sensor_Fault and Tamper are almost always Valid.
    We enforce that here at the truth-level, not the observed Tag-level.
    """
    if alarm_type in {"Sensor_Fault", "Tamper"}:
        # with very high probability, truth is Valid
        if random.random() >= rare_fault_false_rate:
            return "Valid"
        # rare case allowed: keep whatever tag scenario selected (or force False/Nuisance)
        # Here, we bias rare cases toward False rather than Nuisance.
        return random.choice(["False", "Nuisance"])
    return true_tag

def sample_observed_tag(true_tag: str, noise_rate: float):
    """
    noise_rate = probability the operator chose the wrong tag.
    Correct Tag field marks whether observed Tag matches true_tag.
    """
    if random.random() < noise_rate:
        wrong = random.choice(CONFUSIONS.get(true_tag, TAGS))
        return wrong, "Invalid"
    return true_tag, "Valid"

def next_alarm_time(current: datetime, in_burst: bool) -> datetime:
    if in_burst:
        return current + timedelta(milliseconds=random.randint(200, 4000))
    return current + timedelta(seconds=random.randint(30, 600))

def generate_rows(
    n: int,
    noise_rate: float,
    start_dt: datetime,
    rare_fault_false_rate: float = 0.005,  # 0.5% of faults/tampers are truly false/nuisance
):
    dt = start_dt
    point_id = 100

    # burst state
    in_burst = False
    burst_remaining = 0
    burst_station = None

    for _ in range(n):
        if not in_burst and random.random() < 0.15:
            in_burst = True
            burst_remaining = random.randint(10, 80)
            burst_station = make_station()

        dt = next_alarm_time(dt, in_burst)

        station = burst_station if in_burst else make_station()
        if in_burst:
            burst_remaining -= 1
            if burst_remaining <= 0:
                in_burst = False

        # --- Ground truth by construction ---
        true_tag = weighted_choice_tag()
        scenario = TAG_SCENARIOS[true_tag]

        alarm_type = random.choice(scenario["alarm_types"])
        mode = random.choice(scenario["modes"])
        comment_prefix = random.choice(scenario["comment_prefix"])

        # enforce your domain constraint (fault/tamper almost always Valid)
        true_tag = enforce_faults_usually_valid(true_tag, alarm_type, rare_fault_false_rate)

        # If enforcement changed truth, switch to that scenario for consistency (comment/text matches truth)
        scenario = TAG_SCENARIOS[true_tag]
        # keep alarm_type if it's fault/tamper; otherwise resample to match truth scenario
        if alarm_type not in scenario["alarm_types"]:
            alarm_type = random.choice(scenario["alarm_types"])
        if mode not in scenario["modes"]:
            mode = random.choice(scenario["modes"])
        comment_prefix = random.choice(scenario["comment_prefix"])

        # --- Generate remaining fields ---
        point_id += random.randint(0, 3)
        alarm_ts = dt

        lag_ms = random.randint(200, 4000) if in_burst else random.randint(200, 12000)
        assess_ts = alarm_ts + timedelta(milliseconds=lag_ms)

        observed_tag, correct = sample_observed_tag(true_tag, noise_rate)

        row = {
            "Alarm timestamp": fmt_ts(alarm_ts),
            "Alarm type": alarm_type,
            "Alarm Point number": make_point_number(point_id),
            "Alarm name": make_alarm_name(alarm_type),
            "Monitor number": random.randint(1, 20),
            "Alarm direction": random.choice([1, 2, 3]),
            "Mode": mode,
            "Acknowledge by": station,
            "Tag": observed_tag,
            "Full comment": make_comment(comment_prefix),
            "Assessment timestamp": fmt_ts(assess_ts),
            "Correct Tag": correct,
        }
        yield row

def write_csv(path: str, n: int, noise_rate: float, rare_fault_false_rate: float = 0.005):
    start_dt = datetime(2025, 12, 5, 10, 15, 42, 198000)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in generate_rows(
            n=n,
            noise_rate=noise_rate,
            start_dt=start_dt,
            rare_fault_false_rate=rare_fault_false_rate,
        ):
            w.writerow(row)

if __name__ == "__main__":
    # Example: 1M rows, 5% operator mis-tags, 0.5% of faults/tampers truly false-ish
    write_csv("synthetic_alarms_100K.csv", n=100_000, noise_rate=0.05, rare_fault_false_rate=0.005)
