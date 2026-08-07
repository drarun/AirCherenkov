import aircherenkov


def test_public_package_api_is_importable():
    assert aircherenkov.__version__ == '0.2.0'
    assert aircherenkov.ShowerSimulation is not None
    assert aircherenkov.TelescopeArray is not None
