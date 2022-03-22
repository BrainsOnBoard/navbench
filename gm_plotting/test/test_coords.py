from gm_plotting.coords import *
import pytest

# GPS coords of Sussex campus
TEST_GPS = (50.8677123, -0.0875492)

# Verified with MATLAB. MATLAB gave fewer decimal places.
TEST_MERC = (0.33536059667040163, 0.49975680777777776)

def test_gps_to_merc():
    assert pytest.approx(gps_to_merc(*TEST_GPS)) == TEST_MERC

def test_merc_to_gps():
    assert pytest.approx(merc_to_gps(*TEST_MERC)) == TEST_GPS

def test_utm_to_merc():
    utm_coords = utm.from_latlon(*TEST_GPS)

    # Check that the equivalent expression with UTM coords is approx the same
    assert pytest.approx(utm_to_merc(*utm_coords)) == TEST_MERC
