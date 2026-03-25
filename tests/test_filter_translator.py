import pytest

from vink.exceptions import FilterError
from vink.filter_parser import FilterToSql


@pytest.fixture(scope="module")
def translator():
    return FilterToSql()


@pytest.mark.parametrize("filter_expr,expected_keys,expected_param", [
    ("content == 'python'", ("content_fts5", "="), "python"),
    ("category == 'tech'", ("category", "="), "tech"),
    ("price >= 10", ("price", ">="), 10),
    ("rating > 4.5", ("rating", ">"), 4.5),
    ("in_stock == True", ("in_stock", "="), 1),
    ("active == False", ("active", "="), 0),
    ("value == 1e3", ("value", "="), 1000.0),
    ("rate == .5", ("rate", "="), 0.5),
    ("temp < -10", ("temp", "<"), -10),
    ("категория == 'tech'", ("категория", "="), "tech"),
])
def test_valid_filters(translator, filter_expr, expected_keys, expected_param):
    clause, params = translator.translate([filter_expr])
    assert all(key in clause for key in expected_keys), f"Expected {expected_keys} in clause, got: {clause}"
    assert params == [expected_param], f"Expected params {[expected_param]}, got: {params}"


def test_multiple_filters(translator):
    filters = ["category == 'tech'", "price >= 10"]
    clause, params = translator.translate(filters)
    assert "category" in clause and "= ?" in clause, f"Missing 'category' and '= ?' in clause: {clause}"
    assert "price" in clause and ">= ?" in clause, f"Missing 'price' and '>= ?' in clause: {clause}"
    assert params == ["tech", 10], f"Expected ['tech', 10], got: {params}"


def test_empty_filters(translator):
    clause, params = translator.translate([])
    assert clause == "", f"Expected empty clause, got: {clause}"
    assert params == [], f"Expected empty params, got: {params}"


@pytest.mark.parametrize("invalid_filter", [
    "category 'tech'",
    "category == 'tech' extra",
    "== 'tech'",
    "category ==",
])
def test_invalid_filters(translator, invalid_filter):
    with pytest.raises(FilterError):
        translator.translate([invalid_filter])
