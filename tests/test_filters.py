from components.filters import create_filter_bar


def test_create_filter_bar():
    categories = ["Electronics", "Clothing", "Home"]
    regions = ["North America", "Europe"]
    bar = create_filter_bar(categories, regions)
    assert bar is not None
