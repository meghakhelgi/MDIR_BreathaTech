from __future__ import annotations

import json
import unittest

from conversion import (
    MappingConfig,
    SYNTHETIC_ECO_MAX,
    convert_current_to_ppm,
    current_for_visible_ppm,
    render_payload_text,
)


class ConversionTests(unittest.TestCase):
    def test_zero_current_clips_to_zero_ppm(self) -> None:
        result = convert_current_to_ppm(0.0, current_unit="nA")
        self.assertEqual(result.synthetic_eco_ppm, 0.0)
        self.assertTrue(result.eco_ppm_clipped)

    def test_unit_selection_keeps_equivalent_current_consistent(self) -> None:
        result_na = convert_current_to_ppm(1500.0, current_unit="nA")
        result_ua = convert_current_to_ppm(1.5, current_unit="uA")
        self.assertAlmostEqual(result_na.synthetic_eco_ppm, result_ua.synthetic_eco_ppm, places=6)

    def test_inverse_lookup_hits_requested_ppm_inside_fit_range(self) -> None:
        current_value, clipped = current_for_visible_ppm(10.0, current_unit="uA")
        result = convert_current_to_ppm(current_value, current_unit="uA")
        self.assertFalse(clipped)
        self.assertAlmostEqual(result.synthetic_eco_ppm, 10.0, places=6)

    def test_sensitivity_multiplier_compensates_lower_signal(self) -> None:
        baseline = convert_current_to_ppm(1000.0, current_unit="nA")
        corrected = convert_current_to_ppm(
            1000.0,
            current_unit="nA",
            mapping=MappingConfig(sensitivity_multiplier=2.0),
        )
        expected = convert_current_to_ppm(2000.0, current_unit="nA")
        self.assertGreater(corrected.synthetic_eco_ppm, baseline.synthetic_eco_ppm)
        self.assertAlmostEqual(corrected.synthetic_eco_ppm, expected.synthetic_eco_ppm, places=6)

    def test_high_current_clips_to_fit_max(self) -> None:
        result = convert_current_to_ppm(5000.0, current_unit="nA")
        self.assertEqual(result.synthetic_eco_ppm, SYNTHETIC_ECO_MAX)
        self.assertTrue(result.eco_ppm_clipped)

    def test_payload_template_injects_numeric_ppm(self) -> None:
        rendered = render_payload_text('{"eco_ppm": {{PPM}}}', 12.3456)
        payload = json.loads(rendered)
        self.assertAlmostEqual(payload["eco_ppm"], 12.3456, places=4)


if __name__ == "__main__":
    unittest.main()
