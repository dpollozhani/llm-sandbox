from data_analyst.utils.dataframe import to_preview_records


def test_to_preview_records_wraps_a_single_dict_as_one_row():
    assert to_preview_records({"mainaccount": "654000"}) == [{"mainaccount": "654000"}]


def test_to_preview_records_caps_a_list_of_dicts():
    rows = [{"i": i} for i in range(10)]
    assert to_preview_records(rows, limit=5) == rows[:5]


def test_to_preview_records_is_empty_for_a_scalar_or_string():
    assert to_preview_records(42) == []
    assert to_preview_records("654000") == []


def test_to_preview_records_is_empty_for_none():
    assert to_preview_records(None) == []


def test_to_preview_records_is_empty_for_a_list_of_non_dicts():
    """Only a list already row-shaped (every item a dict) counts - a list
    of scalars has no meaningful "row" to make of it."""
    assert to_preview_records([1, 2, 3]) == []
