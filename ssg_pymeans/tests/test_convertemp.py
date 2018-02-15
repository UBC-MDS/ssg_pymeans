import pytest
from convertemp.convertemp import fahr_to_kelvin

def test_fahr_to_kelvin():
    assert fahr_to_kelvin(32) == 273.15, '32 fahrenheit should equal 273.15 kelvin'