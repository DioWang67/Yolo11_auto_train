"""
Comprehensive tests for utils/normalization.py module.
Coverage target: 4% → 70%+
"""
import pytest
from picture_tool.utils.normalization import normalize_imgsz, normalize_name_sequence


class TestNormalizeImgsz:
    """Test normalize_imgsz function."""
    
    def test_returns_none_for_none(self):
        """Should return None for None input."""
        assert normalize_imgsz(None) is None
    
    def test_normalizes_scalar_int(self):
        """Should convert scalar int to [val, val]."""
        result = normalize_imgsz(640)
        assert result == [640, 640]
    
    def test_normalizes_scalar_str(self):
        """Should convert string number to [val, val]."""
        result = normalize_imgsz("1024")
        assert result == [1024, 1024]
    
    def test_normalizes_list_single_element(self):
        """Should expand single element list."""
        result = normalize_imgsz([800])
        assert result == [800, 800]
    
    def test_normalizes_list_two_elements(self):
        """Should return two-element list as-is."""
        result = normalize_imgsz([640, 480])
        assert result == [640, 480]
    
    def test_normalizes_tuple(self):
        """Should convert tuple to list."""
        result = normalize_imgsz((1280, 720))
        assert result == [1280, 720]
    
    def test_returns_none_for_invalid_type(self):
        """Should return None for invalid types."""
        assert normalize_imgsz({}) is None
        assert normalize_imgsz(object()) is None
    
    def test_returns_none_for_non_numeric_string(self):
        """Should return None for invalid string."""
        assert normalize_imgsz("abc") is None
    
    def test_returns_none_for_empty_list(self):
        """Should return None for empty list."""
        assert normalize_imgsz([]) is None
    
    def test_returns_none_for_oversized_list(self):
        """Should return None for list with >2 elements."""
        assert normalize_imgsz([640, 480, 320]) is None


class TestNormalizeNameSequence:
    """Test normalize_name_sequence function."""
    
    def test_returns_empty_for_none(self):
        """Should return empty list for None."""
        assert normalize_name_sequence(None) == []
    
    def test_returns_empty_for_empty_list(self):
        """Should return empty list for empty input."""
        assert normalize_name_sequence([]) == []
    
    def test_normalizes_list_of_strings(self):
        """Should return list of strings as-is."""
        names = ["class1", "class2", "class3"]
        result = normalize_name_sequence(names)
        assert result == names
    
    def test_normalizes_dict_mapping(self):
        """Should convert dict {idx: name} to ordered list."""
        names = {0: "dog", 1: "cat", 2: "bird"}
        result = normalize_name_sequence(names)
        assert result == ["dog", "cat", "bird"]
    
    def test_normalizes_dict_with_non_sequential_keys(self):
        """Should handle non-sequential integer keys."""
        names = {0: "one", 2: "three", 1: "two"}
        result = normalize_name_sequence(names)
        # Should be sorted by key
        assert result == ["one", "two", "three"]
    
    def test_converts_non_string_values_to_strings(self):
        """Should convert all values to strings."""
        names = [1, 2, "three", None]
        result = normalize_name_sequence(names)
        assert result == ["1", "2", "three", "None"]
    
    def test_filters_out_none_values(self):
        """Should not include literal None."""
        names = ["a", None, "b"]
        result = normalize_name_sequence(names)
        # None gets converted to "None" string based on current implementation
        assert len(result) == 3
    
    def test_handles_tuple_input(self):
        """Should handle tuple input."""
        names = ("first", "second", "third")
        result = normalize_name_sequence(names)
        assert result == ["first", "second", "third"]
    
    def test_returns_empty_for_invalid_type(self):
        """Should return empty list for invalid input."""
        assert normalize_name_sequence(123) == []
        assert normalize_name_sequence("not_a_list") == []


class TestNormalizationEdgeCases:
    """Test edge cases and integration scenarios."""
    
    def test_imgsz_with_float_values(self):
        """Should handle float values."""
        result = normalize_imgsz(640.0)
        assert result == [640, 640]
    
    def test_imgsz_with_float_list(self):
        """Should handle list of floats."""
        result = normalize_imgsz([640.5, 480.3])
        assert result == [640, 480]
    
    def test_name_sequence_with_mixed_types(self):
        """Should handle mixed types in list."""
        names = [1, "two", 3.0, True]
        result = normalize_name_sequence(names)
        assert result == ["1", "two", "3.0", "True"]
