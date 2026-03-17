from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd


# ============================================================
# CONFIG / LOOKUPS
# ============================================================

SEED = 42
random.seed(SEED)

ALARM_TYPES = ["Intrusion", "Tamper", "Sensor_Fault"]
MODES = ["Armed", "Secured"]
TAGS = ["Valid", "Nuisance", "FALSE"]

OPERATOR_NAMES = [
    "Tyler Douglas",
    "Christopher Henderson",
    "Shawn Schmidt",
    "Mark Clay",
    "Allison Perez",
    "Ashley Patel",
    "Susan Stevenson",
    "Jeffrey Prince",
    "Brandon Cole",
    "Amanda Brooks",
]

OPERATOR_STATIONS = [f"Operator Station {i}" for i in range(1, 8)]

SENSOR_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Sensor A ... Sensor Z

WEATHER_COMMENTS = ["Storm", "Wind", "Rain", "Weather", "Heavy wind"]
ANIMAL_COMMENTS = ["Animal", "Wildlife", "Possible animal", "Deer near fence"]
INTRUSION_COMMENTS = [
    "Possible breach",
    "Verified",
    "Camera confirmed",
    "Dispatch",
    "Police notified",
    "Unauthorized access",
]
MAINTENANCE_COMMENTS = [
    "Maintenance",
    "Technician onsite",
    "Service activity",
    "Work order",
    "Dispatch",
]
FAULT_COMMENTS = [
    "Sensor fault",
    "Device trouble",
    "Service required",
    "Hardware issue",
    "Dispatch",
]

# Simple sector adjacency graph.
# You can expand this to 50+ sectors easily.
ADJACENCY: Dict[int, List[int]] = {
    4: [3, 5],
    5: [4, 6],
    6: [5, 7],
    18: [17, 19],
    19: [18, 20],
    20: [19, 21],
    29: [28, 30],
    30: [29, 31],
    31: [30, 32],
    36: [35, 37],
    37: [36, 38],
    38: [37, 39],
    40: [39, 41],
    41: [40, 42],
    42: [41, 43],
    48: [47, 49],
    49: [48, 50],
    50: [49, 51],
    56: [55, 57],
    57: [56, 58],
    58: [57, 59],
}

VALID_START_SECTORS = list(ADJACENCY.keys())

SCENARIO_WEIGHTS = {
    "intruder_fence_to_camera": 0.15,
    "multi_sector_intruder": 0.12,
    "gate_tamper_then_intrusion": 0.10,
    "ambiguous_intrusion": 0.08,
    "storm_cluster": 0.18,
    "animal_path": 0.14,
    "authorized_maintenance": 0.11,
    "sensor_fault_burst": 0.12,
}


# ============================================================
# DATA MODELS
# ============================================================

@dataclass
class Step:
    step_order: int
    relative_time_sec: float
    device_kind: str
    alarm_type: str
    sector: int
    sensor_label: str
    fdb: int
    ipd: int
    monitor_number: int
    alarm_direction: int
    mode: str
    observation: str


@dataclass
class Episode:
    episode_id: str
    scenario_type: str
    ground_truth_event_class: str
    should_be_tag: str
    assigned_tag: str
    is_mislabeled: bool
    operator_station: str
    operator_name: str
    shift: str
    weather: str
    episode_start: datetime
    steps: List[Step]
    comment_texts: List[str]


# ============================================================
# HELPERS
# ============================================================

def weighted_choice(weight_map: Dict[str, float]) -> str:
    names = list(weight_map.keys())
    weights = list(weight_map.values())
    return random.choices(names, weights=weights, k=1)[0]


def random_sensor_label() -> str:
    return f"Sensor {random.choice(SENSOR_LABELS)}"


def random_fdb() -> int:
    return random.randint(1, 5)


def random_ipd() -> int:
    return random.randint(100, 399)


def random_monitor() -> int:
    return random.randint(1, 17)


def random_alarm_direction() -> int:
    # You can tune this distribution later if needed
    return random.choice([1, 1, 1, 3, 3])


def random_mode(default: str = "Armed") -> str:
    return default if random.random() < 0.85 else random.choice(MODES)


def pick_operator() -> tuple[str, str]:
    return random.choice(OPERATOR_STATIONS), random.choice(OPERATOR_NAMES)


def pick_shift(dt: datetime) -> str:
    hour = dt.hour
    if 7 <= hour < 15:
        return "day"
    if 15 <= hour < 23:
        return "evening"
    return "night"


def format_timestamp(dt: datetime) -> str:
    # Matches example format like 12/05/2025 10:23:56.198 AM
    return dt.strftime("%m/%d/%Y %I:%M:%S.") + f"{dt.microsecond // 1000:03d} " + dt.strftime("%p")


def build_alarm_name(sensor_label: str, alarm_type: str, sector: int) -> str:
    if alarm_type == "Intrusion":
        mid = "Alarm"
    elif alarm_type == "Tamper":
        mid = "Tamper"
    elif alarm_type == "Sensor_Fault":
        mid = "Fault"
    else:
        mid = alarm_type
    return f"{sensor_label}  {mid}  Sector {sector}"


def sample_path(start_sector: int, length: int) -> List[int]:
    """
    Random walk over adjacent sectors.
    """
    path = [start_sector]
    current = start_sector

    for _ in range(length - 1):
        neighbors = ADJACENCY.get(current, [])
        if not neighbors:
            break
        nxt = random.choice(neighbors)
        path.append(nxt)
        current = nxt
    return path


def maybe_mislabel(scenario_type: str, should_be_tag: str) -> tuple[str, bool]:
    """
    Structured mislabel logic.
    Tweak these rates later based on what you want for your experiments.
    """
    mislabel_probs = {
        "intruder_fence_to_camera": 0.05,
        "multi_sector_intruder": 0.03,
        "gate_tamper_then_intrusion": 0.08,
        "ambiguous_intrusion": 0.22,
        "storm_cluster": 0.08,
        "animal_path": 0.15,
        "authorized_maintenance": 0.10,
        "sensor_fault_burst": 0.12,
    }

    p = mislabel_probs.get(scenario_type, 0.05)
    if random.random() > p:
        return should_be_tag, False

    # Some human-like mistakes
    if should_be_tag == "Valid":
        return random.choice(["Nuisance", "FALSE"]), True
    else:
        return random.choice(["Valid", "FALSE"] if should_be_tag == "Nuisance" else ["Valid", "Nuisance"]), True


def jitter_seconds(base: float, low: float = -1.5, high: float = 1.5) -> float:
    return max(0, base + random.uniform(low, high))


def assessment_delay_sec(tag: str, scenario_type: str) -> float:
    if scenario_type in {"storm_cluster", "animal_path"}:
        return random.uniform(2, 8)
    if scenario_type in {"intruder_fence_to_camera", "multi_sector_intruder", "gate_tamper_then_intrusion", "ambiguous_intrusion"}:
        return random.uniform(4, 18)
    return random.uniform(2, 12)


def next_alarm_point_id(start_id: int) -> int:
    return start_id


# ============================================================
# SCENARIO TEMPLATE FUNCTIONS
# ============================================================

def scenario_intruder_fence_to_camera(ep_id: str, start_dt: datetime) -> Episode:
    operator_station, operator_name = pick_operator()
    start_sector = random.choice([29, 30, 31, 18, 19, 20, 40, 41, 42])

    path = sample_path(start_sector, length=random.randint(2, 3))

    steps = []
    rel = 0.0
    for i, sector in enumerate(path, start=1):
        steps.append(
            Step(
                step_order=i,
                relative_time_sec=rel,
                device_kind="perimeter_sensor" if i == 1 else "camera_zone" if i == 2 else "inner_sensor",
                alarm_type="Intrusion",
                sector=sector,
                sensor_label=random_sensor_label(),
                fdb=random_fdb(),
                ipd=random_ipd(),
                monitor_number=random_monitor(),
                alarm_direction=random_alarm_direction(),
                mode="Armed",
                observation="intruder movement"
            )
        )
        rel += random.uniform(5, 12)

    should_be_tag = "Valid"
    assigned_tag, is_mislabeled = maybe_mislabel("intruder_fence_to_camera", should_be_tag)

    comments = ["Possible breach", "Camera confirmed", "Police notified"][:len(steps)]

    return Episode(
        episode_id=ep_id,
        scenario_type="intruder_fence_to_camera",
        ground_truth_event_class="real_intrusion",
        should_be_tag=should_be_tag,
        assigned_tag=assigned_tag,
        is_mislabeled=is_mislabeled,
        operator_station=operator_station,
        operator_name=operator_name,
        shift=pick_shift(start_dt),
        weather="clear",
        episode_start=start_dt,
        steps=steps,
        comment_texts=comments,
    )


def scenario_multi_sector_intruder(ep_id: str, start_dt: datetime) -> Episode:
    operator_station, operator_name = pick_operator()
    start_sector = random.choice([4, 5, 6, 48, 49, 50, 56, 57, 58])

    path = sample_path(start_sector, length=random.randint(3, 5))

    steps = []
    rel = 0.0
    for i, sector in enumerate(path, start=1):
        steps.append(
            Step(
                step_order=i,
                relative_time_sec=rel,
                device_kind="perimeter_sensor" if i == 1 else "inner_sensor",
                alarm_type="Intrusion",
                sector=sector,
                sensor_label=random_sensor_label(),
                fdb=random_fdb(),
                ipd=random_ipd(),
                monitor_number=random_monitor(),
                alarm_direction=random_alarm_direction(),
                mode="Armed",
                observation="continued movement inward"
            )
        )
        rel += random.uniform(4, 15)

    should_be_tag = "Valid"
    assigned_tag, is_mislabeled = maybe_mislabel("multi_sector_intruder", should_be_tag)

    comments_pool = ["Possible breach", "Verified", "Camera confirmed", "Dispatch", "Police notified"]
    comments = comments_pool[:len(steps)]

    return Episode(
        episode_id=ep_id,
        scenario_type="multi_sector_intruder",
        ground_truth_event_class="real_intrusion",
        should_be_tag=should_be_tag,
        assigned_tag=assigned_tag,
        is_mislabeled=is_mislabeled,
        operator_station=operator_station,
        operator_name=operator_name,
        shift=pick_shift(start_dt),
        weather="clear",
        episode_start=start_dt,
        steps=steps,
        comment_texts=comments,
    )


def scenario_gate_tamper_then_intrusion(ep_id: str, start_dt: datetime) -> Episode:
    operator_station, operator_name = pick_operator()
    start_sector = random.choice([18, 29, 40, 56])
    next_sector = random.choice(ADJACENCY[start_sector])

    steps = [
        Step(
            step_order=1,
            relative_time_sec=0.0,
            device_kind="gate_panel",
            alarm_type="Tamper",
            sector=start_sector,
            sensor_label=random_sensor_label(),
            fdb=random_fdb(),
            ipd=random_ipd(),
            monitor_number=random_monitor(),
            alarm_direction=1,
            mode="Armed",
            observation="possible gate tamper"
        ),
        Step(
            step_order=2,
            relative_time_sec=random.uniform(3, 10),
            device_kind="perimeter_sensor",
            alarm_type="Intrusion",
            sector=next_sector,
            sensor_label=random_sensor_label(),
            fdb=random_fdb(),
            ipd=random_ipd(),
            monitor_number=random_monitor(),
            alarm_direction=random_alarm_direction(),
            mode="Armed",
            observation="intrusion after tamper"
        )
    ]

    if random.random() < 0.7:
        third_sector = random.choice(ADJACENCY.get(next_sector, [next_sector]))
        steps.append(
            Step(
                step_order=3,
                relative_time_sec=steps[-1].relative_time_sec + random.uniform(4, 10),
                device_kind="camera_zone",
                alarm_type="Intrusion",
                sector=third_sector,
                sensor_label=random_sensor_label(),
                fdb=random_fdb(),
                ipd=random_ipd(),
                monitor_number=random_monitor(),
                alarm_direction=3,
                mode="Armed",
                observation="camera confirmation"
            )
        )

    should_be_tag = "Valid"
    assigned_tag, is_mislabeled = maybe_mislabel("gate_tamper_then_intrusion", should_be_tag)
    comments = ["Verified", "Unauthorized access", "Dispatch"][:len(steps)]

    return Episode(
        episode_id=ep_id,
        scenario_type="gate_tamper_then_intrusion",
        ground_truth_event_class="real_intrusion",
        should_be_tag=should_be_tag,
        assigned_tag=assigned_tag,
        is_mislabeled=is_mislabeled,
        operator_station=operator_station,
        operator_name=operator_name,
        shift=pick_shift(start_dt),
        weather="clear",
        episode_start=start_dt,
        steps=steps,
        comment_texts=comments,
    )


def scenario_ambiguous_intrusion(ep_id: str, start_dt: datetime) -> Episode:
    operator_station, operator_name = pick_operator()
    start_sector = random.choice(VALID_START_SECTORS)
    path = sample_path(start_sector, length=random.randint(2, 3))

    steps = []
    rel = 0.0
    for i, sector in enumerate(path, start=1):
        steps.append(
            Step(
                step_order=i,
                relative_time_sec=rel,
                device_kind="weak_signal_sensor",
                alarm_type="Intrusion",
                sector=sector,
                sensor_label=random_sensor_label(),
                fdb=random_fdb(),
                ipd=random_ipd(),
                monitor_number=random_monitor(),
                alarm_direction=random.choice([1, 3]),
                mode="Armed",
                observation="partial evidence"
            )
        )
        rel += random.uniform(8, 20)

    should_be_tag = "Valid"
    assigned_tag, is_mislabeled = maybe_mislabel("ambiguous_intrusion", should_be_tag)
    comments = random.sample(["Possible breach", "Checking", "Verified", "Dispatch"], k=len(steps))

    return Episode(
        episode_id=ep_id,
        scenario_type="ambiguous_intrusion",
        ground_truth_event_class="real_intrusion",
        should_be_tag=should_be_tag,
        assigned_tag=assigned_tag,
        is_mislabeled=is_mislabeled,
        operator_station=operator_station,
        operator_name=operator_name,
        shift=pick_shift(start_dt),
        weather=random.choice(["clear", "low_visibility", "fog"]),
        episode_start=start_dt,
        steps=steps,
        comment_texts=comments,
    )


def scenario_storm_cluster(ep_id: str, start_dt: datetime) -> Episode:
    operator_station, operator_name = pick_operator()
    start_sector = random.choice(VALID_START_SECTORS)
    path = sample_path(start_sector, length=random.randint(2, 4))

    steps = []
    rel = 0.0
    for i, sector in enumerate(path, start=1):
        steps.append(
            Step(
                step_order=i,
                relative_time_sec=rel,
                device_kind="perimeter_sensor",
                alarm_type="Intrusion",
                sector=sector,
                sensor_label=random_sensor_label(),
                fdb=random_fdb(),
                ipd=random_ipd(),
                monitor_number=random_monitor(),
                alarm_direction=random_alarm_direction(),
                mode="Armed",
                observation="weather nuisance"
            )
        )
        rel += random.uniform(1, 6)

    should_be_tag = "Nuisance"
    assigned_tag, is_mislabeled = maybe_mislabel("storm_cluster", should_be_tag)
    comments = [random.choice(WEATHER_COMMENTS) for _ in steps]

    return Episode(
        episode_id=ep_id,
        scenario_type="storm_cluster",
        ground_truth_event_class="weather_nuisance",
        should_be_tag=should_be_tag,
        assigned_tag=assigned_tag,
        is_mislabeled=is_mislabeled,
        operator_station=operator_station,
        operator_name=operator_name,
        shift=pick_shift(start_dt),
        weather=random.choice(["storm", "rain", "windy"]),
        episode_start=start_dt,
        steps=steps,
        comment_texts=comments,
    )


def scenario_animal_path(ep_id: str, start_dt: datetime) -> Episode:
    operator_station, operator_name = pick_operator()
    start_sector = random.choice(VALID_START_SECTORS)
    path = sample_path(start_sector, length=random.randint(2, 3))

    steps = []
    rel = 0.0
    for i, sector in enumerate(path, start=1):
        steps.append(
            Step(
                step_order=i,
                relative_time_sec=rel,
                device_kind="perimeter_sensor" if i == 1 else "camera_zone",
                alarm_type="Intrusion",
                sector=sector,
                sensor_label=random_sensor_label(),
                fdb=random_fdb(),
                ipd=random_ipd(),
                monitor_number=random_monitor(),
                alarm_direction=random_alarm_direction(),
                mode="Armed",
                observation="animal movement"
            )
        )
        rel += random.uniform(2, 8)

    should_be_tag = "Nuisance"
    assigned_tag, is_mislabeled = maybe_mislabel("animal_path", should_be_tag)
    comments = [random.choice(ANIMAL_COMMENTS) for _ in steps]

    return Episode(
        episode_id=ep_id,
        scenario_type="animal_path",
        ground_truth_event_class="animal_activity",
        should_be_tag=should_be_tag,
        assigned_tag=assigned_tag,
        is_mislabeled=is_mislabeled,
        operator_station=operator_station,
        operator_name=operator_name,
        shift=pick_shift(start_dt),
        weather="clear",
        episode_start=start_dt,
        steps=steps,
        comment_texts=comments,
    )


def scenario_authorized_maintenance(ep_id: str, start_dt: datetime) -> Episode:
    operator_station, operator_name = pick_operator()
    sector = random.choice(VALID_START_SECTORS)

    steps = [
        Step(
            step_order=1,
            relative_time_sec=0.0,
            device_kind="control_panel",
            alarm_type=random.choice(["Tamper", "Sensor_Fault"]),
            sector=sector,
            sensor_label=random_sensor_label(),
            fdb=random_fdb(),
            ipd=random_ipd(),
            monitor_number=random_monitor(),
            alarm_direction=1,
            mode=random.choice(["Secured", "Armed"]),
            observation="authorized work"
        )
    ]

    if random.random() < 0.55:
        steps.append(
            Step(
                step_order=2,
                relative_time_sec=random.uniform(2, 10),
                device_kind="adjacent_device",
                alarm_type=random.choice(["Tamper", "Sensor_Fault"]),
                sector=sector,
                sensor_label=random_sensor_label(),
                fdb=random_fdb(),
                ipd=random_ipd(),
                monitor_number=random_monitor(),
                alarm_direction=1,
                mode=random.choice(["Secured", "Armed"]),
                observation="continued service activity"
            )
        )

    should_be_tag = "Valid"
    assigned_tag, is_mislabeled = maybe_mislabel("authorized_maintenance", should_be_tag)
    comments = [random.choice(MAINTENANCE_COMMENTS) for _ in steps]

    return Episode(
        episode_id=ep_id,
        scenario_type="authorized_maintenance",
        ground_truth_event_class="authorized_maintenance",
        should_be_tag=should_be_tag,
        assigned_tag=assigned_tag,
        is_mislabeled=is_mislabeled,
        operator_station=operator_station,
        operator_name=operator_name,
        shift=pick_shift(start_dt),
        weather="clear",
        episode_start=start_dt,
        steps=steps,
        comment_texts=comments,
    )


def scenario_sensor_fault_burst(ep_id: str, start_dt: datetime) -> Episode:
    operator_station, operator_name = pick_operator()
    sector = random.choice(VALID_START_SECTORS)
    sensor_label = random_sensor_label()
    fdb = random_fdb()
    ipd = random_ipd()
    monitor_number = random_monitor()

    num_steps = random.randint(2, 4)
    steps = []
    rel = 0.0
    for i in range(1, num_steps + 1):
        steps.append(
            Step(
                step_order=i,
                relative_time_sec=rel,
                device_kind="faulty_sensor",
                alarm_type="Sensor_Fault",
                sector=sector,
                sensor_label=sensor_label,
                fdb=fdb,
                ipd=ipd,
                monitor_number=monitor_number,
                alarm_direction=random.choice([1, 3]),
                mode=random.choice(["Secured", "Armed"]),
                observation="repeated device degradation"
            )
        )
        rel += random.uniform(10, 60)

    should_be_tag = "Valid"
    assigned_tag, is_mislabeled = maybe_mislabel("sensor_fault_burst", should_be_tag)
    comments = [random.choice(FAULT_COMMENTS) for _ in steps]

    return Episode(
        episode_id=ep_id,
        scenario_type="sensor_fault_burst",
        ground_truth_event_class="equipment_fault",
        should_be_tag=should_be_tag,
        assigned_tag=assigned_tag,
        is_mislabeled=is_mislabeled,
        operator_station=operator_station,
        operator_name=operator_name,
        shift=pick_shift(start_dt),
        weather="clear",
        episode_start=start_dt,
        steps=steps,
        comment_texts=comments,
    )


SCENARIO_FUNCTIONS = {
    "intruder_fence_to_camera": scenario_intruder_fence_to_camera,
    "multi_sector_intruder": scenario_multi_sector_intruder,
    "gate_tamper_then_intrusion": scenario_gate_tamper_then_intrusion,
    "ambiguous_intrusion": scenario_ambiguous_intrusion,
    "storm_cluster": scenario_storm_cluster,
    "animal_path": scenario_animal_path,
    "authorized_maintenance": scenario_authorized_maintenance,
    "sensor_fault_burst": scenario_sensor_fault_burst,
}


# ============================================================
# EPISODE -> ROWS
# ============================================================

def episode_to_rows(
    episode: Episode,
    starting_alarm_point_id: int,
    include_hidden_columns: bool = True
) -> List[Dict]:
    rows = []
    alarm_point_id = starting_alarm_point_id

    for idx, step in enumerate(episode.steps):
        base_alarm_dt = episode.episode_start + timedelta(seconds=step.relative_time_sec)
        alarm_dt = base_alarm_dt + timedelta(seconds=random.uniform(-1.5, 1.5))
        assess_dt = alarm_dt + timedelta(seconds=assessment_delay_sec(episode.assigned_tag, episode.scenario_type))

        comment_text = episode.comment_texts[min(idx, len(episode.comment_texts) - 1)]
        full_comment = f"{comment_text}  //{episode.operator_name}//"

        row = {
            "Alarm timestamp": format_timestamp(alarm_dt),
            "Alarm type": step.alarm_type,
            "Alarm Point number": f"{alarm_point_id}: FDB {step.fdb} - IPD {step.ipd}",
            "Alarm name": build_alarm_name(step.sensor_label, step.alarm_type, step.sector),
            "Monitor number": step.monitor_number,
            "Alarm direction": step.alarm_direction,
            "Mode": step.mode,
            "Acknowledge by": episode.operator_station,
            "Tag": episode.assigned_tag,
            "Full comment": full_comment,
            "Assessment timestamp": format_timestamp(assess_dt),
        }

        if include_hidden_columns:
            row.update({
                "episode_id": episode.episode_id,
                "scenario_type": episode.scenario_type,
                "ground_truth_event_class": episode.ground_truth_event_class,
                "should_be_tag": episode.should_be_tag,
                "is_mislabeled": episode.is_mislabeled,
                "weather": episode.weather,
                "shift": episode.shift,
                "step_order": step.step_order,
                "device_kind": step.device_kind,
                "sector": step.sector,
                "observation": step.observation,
            })

        rows.append(row)
        alarm_point_id += 1

    return rows


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_episode(ep_num: int, start_dt: datetime) -> Episode:
    scenario_name = weighted_choice(SCENARIO_WEIGHTS)
    ep_id = f"EP-{start_dt.strftime('%Y%m%d')}-{ep_num:06d}"
    return SCENARIO_FUNCTIONS[scenario_name](ep_id, start_dt)


def generate_dataset(
    num_episodes: int = 1000,
    start_datetime: str = "2025-12-05 00:00:00",
    include_hidden_columns: bool = True,
) -> pd.DataFrame:
    """
    Generate a flat log dataset from episode-based scenarios.
    """
    start_dt = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
    all_rows = []
    alarm_point_counter = 100

    current_dt = start_dt

    for ep_num in range(1, num_episodes + 1):
        # Move time forward between episodes
        current_dt += timedelta(seconds=random.uniform(15, 300))

        episode = generate_episode(ep_num, current_dt)
        rows = episode_to_rows(
            episode=episode,
            starting_alarm_point_id=alarm_point_counter,
            include_hidden_columns=include_hidden_columns,
        )
        all_rows.extend(rows)
        alarm_point_counter += len(rows)

    df = pd.DataFrame(all_rows)
    return df


# ============================================================
# OPTIONAL POST-PROCESSING / VIEWS
# ============================================================

def get_sponsor_style_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns only the columns from your visible schema.
    """
    visible_cols = [
        "Alarm timestamp",
        "Alarm type",
        "Alarm Point number",
        "Alarm name",
        "Monitor number",
        "Alarm direction",
        "Mode",
        "Acknowledge by",
        "Tag",
        "Full comment",
        "Assessment timestamp",
    ]
    return df[visible_cols].copy()


def summarize_dataset(df: pd.DataFrame) -> None:
    print("\nRows:", len(df))
    if "episode_id" in df.columns:
        print("Episodes:", df["episode_id"].nunique())
    print("\nAlarm type counts:")
    print(df["Alarm type"].value_counts(dropna=False))
    print("\nTag counts:")
    print(df["Tag"].value_counts(dropna=False))

    if "scenario_type" in df.columns:
        print("\nScenario counts:")
        print(df.groupby("scenario_type")["episode_id"].nunique().sort_values(ascending=False))

    if "is_mislabeled" in df.columns:
        print("\nMislabeled rows:")
        print(df["is_mislabeled"].value_counts(dropna=False))


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    df = generate_dataset(
        num_episodes=5000,
        start_datetime="2025-12-05 10:00:00",
        include_hidden_columns=True,
    )

    print("\nFULL DATASET SAMPLE:")
    print(df.head(15).to_string(index=False))

    summarize_dataset(df)

    sponsor_df = get_sponsor_style_view(df)
    print("\nSPONSOR-STYLE SAMPLE:")
    print(sponsor_df.head(10).to_string(index=False))

    # Save if needed
    df.to_csv("synthetic_alarm_logs_with_hidden_columns.csv", index=False)
    sponsor_df.to_csv("synthetic_alarm_logs_sponsor_style.csv", index=False)
    # sponsor_df.to_csv("synthetic_alarm_logs_visible_schema.csv", index=False)