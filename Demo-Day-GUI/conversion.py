from __future__ import annotations

import json
from dataclasses import dataclass

CURRENT_TO_PEROXIDE_SLOPE = 0.0913
CURRENT_TO_PEROXIDE_INTERCEPT = 0.163
PEROXIDE_TO_ECO_SLOPE = 1.545
PEROXIDE_TO_ECO_INTERCEPT = -1.03
SYNTHETIC_ECO_MIN = 0.0
SYNTHETIC_ECO_MAX = 25.0

CURRENT_UNIT_TO_UA = {
    "pA": 0.000001,
    "nA": 0.001,
    "uA": 1.0,
    "mA": 1000.0,
    "A": 1000000.0,
}
CURRENT_UNITS = tuple(CURRENT_UNIT_TO_UA.keys())
DEFAULT_CURRENT_UNIT = "nA"

DEFAULT_PROXY_LEVELS = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0]
DEFAULT_PAYLOAD_TEMPLATE = '{\n  "eco_ppm": {{PPM}}\n}\n'


@dataclass(slots=True)
class MappingConfig:
    current_to_peroxide_slope: float = CURRENT_TO_PEROXIDE_SLOPE
    current_to_peroxide_intercept: float = CURRENT_TO_PEROXIDE_INTERCEPT
    peroxide_to_ppm_slope: float = PEROXIDE_TO_ECO_SLOPE
    peroxide_to_ppm_intercept: float = PEROXIDE_TO_ECO_INTERCEPT
    ppm_min: float = SYNTHETIC_ECO_MIN
    ppm_max: float = SYNTHETIC_ECO_MAX
    sensitivity_multiplier: float = 1.0


@dataclass(slots=True)
class ConversionResult:
    input_current_value: float
    input_current_unit: str
    raw_current_uA: float
    adjusted_current_uA: float
    raw_peroxide_equivalent_mM: float
    peroxide_equivalent_mM: float
    raw_eco_ppm: float
    synthetic_eco_ppm: float
    eco_ppm_clipped: bool


@dataclass(slots=True)
class ProxyPoint:
    requested_ppm: float
    plotted_ppm: float
    current_value: float
    current_unit: str
    clipped_to_fit: bool


def format_number(value: float, digits: int = 3) -> str:
    text = f"{value:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def get_unit_factor_to_uA(current_unit: str) -> float:
    try:
        return CURRENT_UNIT_TO_UA[current_unit]
    except KeyError as exc:
        supported = ", ".join(CURRENT_UNITS)
        raise ValueError(f"Unsupported current unit `{current_unit}`. Use one of: {supported}.") from exc


def current_to_uA(current_value: float, current_unit: str) -> float:
    return current_value * get_unit_factor_to_uA(current_unit)


def current_from_uA(current_uA: float, current_unit: str) -> float:
    return current_uA / get_unit_factor_to_uA(current_unit)


def validate_mapping_config(mapping: MappingConfig) -> None:
    if mapping.current_to_peroxide_slope == 0:
        raise ValueError("Current-to-peroxide slope cannot be 0.")
    if mapping.peroxide_to_ppm_slope == 0:
        raise ValueError("Peroxide-to-PPM slope cannot be 0.")
    if mapping.ppm_max <= mapping.ppm_min:
        raise ValueError("PPM max must be greater than PPM min.")
    if mapping.sensitivity_multiplier <= 0:
        raise ValueError("Sensitivity multiplier must be greater than 0.")


def apply_sensitivity_correction(raw_current_uA: float, mapping: MappingConfig) -> float:
    return raw_current_uA * mapping.sensitivity_multiplier


def peroxide_from_current_uA(
    current_uA: float,
    mapping: MappingConfig,
) -> tuple[float, float]:
    raw_peroxide = (
        current_uA - mapping.current_to_peroxide_intercept
    ) / mapping.current_to_peroxide_slope
    peroxide = max(raw_peroxide, 0.0)
    return raw_peroxide, peroxide


def ppm_from_peroxide(peroxide_equivalent_mM: float, mapping: MappingConfig) -> float:
    return (
        mapping.peroxide_to_ppm_slope * peroxide_equivalent_mM
    ) + mapping.peroxide_to_ppm_intercept


def convert_current_to_ppm(
    current_value: float,
    *,
    current_unit: str = DEFAULT_CURRENT_UNIT,
    mapping: MappingConfig | None = None,
) -> ConversionResult:
    active_mapping = mapping or MappingConfig()
    validate_mapping_config(active_mapping)

    raw_current_uA = current_to_uA(current_value, current_unit)
    adjusted_current_uA = apply_sensitivity_correction(raw_current_uA, active_mapping)
    raw_peroxide, peroxide_equivalent_mM = peroxide_from_current_uA(
        adjusted_current_uA,
        active_mapping,
    )
    raw_eco_ppm = ppm_from_peroxide(peroxide_equivalent_mM, active_mapping)
    synthetic_eco_ppm = clamp(raw_eco_ppm, active_mapping.ppm_min, active_mapping.ppm_max)
    eco_ppm_clipped = raw_peroxide != peroxide_equivalent_mM or raw_eco_ppm != synthetic_eco_ppm

    return ConversionResult(
        input_current_value=current_value,
        input_current_unit=current_unit,
        raw_current_uA=raw_current_uA,
        adjusted_current_uA=adjusted_current_uA,
        raw_peroxide_equivalent_mM=raw_peroxide,
        peroxide_equivalent_mM=peroxide_equivalent_mM,
        raw_eco_ppm=raw_eco_ppm,
        synthetic_eco_ppm=synthetic_eco_ppm,
        eco_ppm_clipped=eco_ppm_clipped,
    )


def current_for_visible_ppm(
    ppm: float,
    *,
    current_unit: str = DEFAULT_CURRENT_UNIT,
    mapping: MappingConfig | None = None,
) -> tuple[float, bool]:
    active_mapping = mapping or MappingConfig()
    validate_mapping_config(active_mapping)

    clipped = ppm < active_mapping.ppm_min or ppm > active_mapping.ppm_max
    plotted_ppm = clamp(ppm, active_mapping.ppm_min, active_mapping.ppm_max)
    raw_peroxide_equivalent_mM = (
        plotted_ppm - active_mapping.peroxide_to_ppm_intercept
    ) / active_mapping.peroxide_to_ppm_slope
    peroxide_equivalent_mM = max(raw_peroxide_equivalent_mM, 0.0)
    adjusted_current_uA = (
        active_mapping.current_to_peroxide_intercept
        + (active_mapping.current_to_peroxide_slope * peroxide_equivalent_mM)
    )
    raw_current_uA = adjusted_current_uA / active_mapping.sensitivity_multiplier
    return current_from_uA(raw_current_uA, current_unit), clipped


def build_proxy_points(
    levels: list[float],
    *,
    current_unit: str = DEFAULT_CURRENT_UNIT,
    mapping: MappingConfig | None = None,
) -> list[ProxyPoint]:
    points: list[ProxyPoint] = []
    for level in levels:
        current_value, clipped = current_for_visible_ppm(
            level,
            current_unit=current_unit,
            mapping=mapping,
        )
        active_mapping = mapping or MappingConfig()
        points.append(
            ProxyPoint(
                requested_ppm=level,
                plotted_ppm=clamp(level, active_mapping.ppm_min, active_mapping.ppm_max),
                current_value=current_value,
                current_unit=current_unit,
                clipped_to_fit=clipped,
            )
        )
    return points


def parse_proxy_levels(text: str) -> list[float]:
    levels: list[float] = []
    for part in text.replace(";", ",").split(","):
        stripped = part.strip()
        if not stripped:
            continue
        levels.append(float(stripped))
    if not levels:
        raise ValueError("Enter at least one proxy PPM level.")
    return sorted(dict.fromkeys(levels))


def render_payload_text(template: str, ppm_value: float) -> str:
    if "{{PPM}}" not in template:
        raise ValueError("Payload template must include the {{PPM}} placeholder.")
    rendered = template.replace("{{PPM}}", format_number(ppm_value, digits=6))
    json.loads(rendered)
    return rendered
