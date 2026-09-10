from processor import final_price


def test_no_tier():
    assert final_price(100.0, "bronze") == 100.0


def test_silver():
    assert final_price(100.0, "silver") == 90.0


def test_gold():
    assert final_price(50.0, "gold") == 40.0
