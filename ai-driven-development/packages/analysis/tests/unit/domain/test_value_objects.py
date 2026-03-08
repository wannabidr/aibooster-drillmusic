"""Unit tests for Python domain value objects."""

import pytest


class TestBPMValue:
    """Tests for BPMValue value object."""

    def test_create_valid_bpm(self):
        from src.domain.value_objects.bpm_value import BPMValue

        bpm = BPMValue(128.0)
        assert bpm.value == 128.0

    def test_reject_bpm_below_minimum(self):
        from src.domain.value_objects.bpm_value import BPMValue

        with pytest.raises(ValueError):
            BPMValue(19.0)

    def test_reject_bpm_above_maximum(self):
        from src.domain.value_objects.bpm_value import BPMValue

        with pytest.raises(ValueError):
            BPMValue(301.0)

    def test_bpm_equality(self):
        from src.domain.value_objects.bpm_value import BPMValue

        assert BPMValue(128.0) == BPMValue(128.0)
        assert BPMValue(128.0) != BPMValue(130.0)

    def test_half_time(self):
        from src.domain.value_objects.bpm_value import BPMValue

        bpm = BPMValue(140.0)
        assert bpm.half_time().value == 70.0

    def test_double_time(self):
        from src.domain.value_objects.bpm_value import BPMValue

        bpm = BPMValue(65.0)
        assert bpm.double_time().value == 130.0

    def test_immutability(self):
        from src.domain.value_objects.bpm_value import BPMValue

        bpm = BPMValue(128.0)
        with pytest.raises(AttributeError):
            bpm.value = 130.0


class TestKeySignature:
    """Tests for KeySignature value object."""

    def test_create_from_standard_notation(self):
        from src.domain.value_objects.key_signature import KeySignature

        key = KeySignature("Am")
        assert key.root == "A"
        assert key.mode == "minor"

    def test_create_from_major_key(self):
        from src.domain.value_objects.key_signature import KeySignature

        key = KeySignature("C")
        assert key.root == "C"
        assert key.mode == "major"

    def test_create_from_camelot_notation(self):
        from src.domain.value_objects.key_signature import KeySignature

        key = KeySignature.from_camelot("8A")
        assert key.root == "A"
        assert key.mode == "minor"

    def test_to_camelot(self):
        from src.domain.value_objects.key_signature import KeySignature

        key = KeySignature("Am")
        assert key.to_camelot() == "8A"

    def test_equality(self):
        from src.domain.value_objects.key_signature import KeySignature

        assert KeySignature("Am") == KeySignature("Am")
        assert KeySignature("Am") != KeySignature("Cm")

    def test_reject_invalid_key(self):
        from src.domain.value_objects.key_signature import KeySignature

        with pytest.raises(ValueError):
            KeySignature("X#")

    def test_immutability(self):
        from src.domain.value_objects.key_signature import KeySignature

        key = KeySignature("Am")
        with pytest.raises(AttributeError):
            key.root = "B"


class TestEnergyProfile:
    """Tests for EnergyProfile value object."""

    def test_create_valid_energy(self):
        from src.domain.value_objects.energy_profile import EnergyProfile

        profile = EnergyProfile(overall=75.0)
        assert profile.overall == 75.0

    def test_reject_energy_below_zero(self):
        from src.domain.value_objects.energy_profile import EnergyProfile

        with pytest.raises(ValueError):
            EnergyProfile(overall=-1.0)

    def test_reject_energy_above_100(self):
        from src.domain.value_objects.energy_profile import EnergyProfile

        with pytest.raises(ValueError):
            EnergyProfile(overall=101.0)

    def test_with_segments(self):
        from src.domain.value_objects.energy_profile import EnergyProfile

        segments = [
            {"start_ms": 0, "end_ms": 30000, "level": 60.0},
            {"start_ms": 30000, "end_ms": 60000, "level": 80.0},
        ]
        profile = EnergyProfile(overall=70.0, segments=segments)
        assert len(profile.segments) == 2

    def test_trajectory(self):
        from src.domain.value_objects.energy_profile import EnergyProfile

        profile = EnergyProfile(overall=70.0, trajectory="build")
        assert profile.trajectory == "build"

    def test_reject_invalid_trajectory(self):
        from src.domain.value_objects.energy_profile import EnergyProfile

        with pytest.raises(ValueError):
            EnergyProfile(overall=70.0, trajectory="invalid")

    def test_equality(self):
        from src.domain.value_objects.energy_profile import EnergyProfile

        assert EnergyProfile(overall=75.0) == EnergyProfile(overall=75.0)
        assert EnergyProfile(overall=75.0) != EnergyProfile(overall=80.0)

    def test_immutability(self):
        from src.domain.value_objects.energy_profile import EnergyProfile

        profile = EnergyProfile(overall=75.0)
        with pytest.raises(AttributeError):
            profile.overall = 80.0
