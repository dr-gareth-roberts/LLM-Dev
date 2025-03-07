"""
test_simple.py - Simple test to verify pytest functionality.
"""
import pytest

def test_simple():
    """Simple test that always passes."""
    assert True

def test_with_output():
    """Test with print output."""
    print("This is a test output message")
    assert 1 + 1 == 2
