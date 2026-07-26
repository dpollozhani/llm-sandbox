from data_analyst.config.settings import Glossary, GlossaryEntry, get_glossary


def test_glossary_loads_terms_from_yaml(tmp_path):
    path = tmp_path / "glossary.yaml"
    path.write_text(
        """
terms:
  - term: BRIC
    definition: An item attribute (a global item code), not the BRICS country grouping.
"""
    )
    get_glossary.cache_clear()

    glossary = get_glossary(path)

    assert glossary.terms == [
        GlossaryEntry(term="BRIC", definition="An item attribute (a global item code), not the BRICS country grouping.")
    ]


def test_glossary_missing_file_returns_empty_not_an_error(tmp_path):
    get_glossary.cache_clear()

    glossary = get_glossary(tmp_path / "does-not-exist.yaml")

    assert glossary.terms == []


def test_glossary_empty_file_returns_empty(tmp_path):
    path = tmp_path / "glossary.yaml"
    path.write_text("")
    get_glossary.cache_clear()

    glossary = get_glossary(path)

    assert glossary.terms == []


def test_glossary_render_is_one_line_per_term():
    glossary = Glossary(
        terms=[
            GlossaryEntry(term="BRIC", definition="An item attribute, a global item code."),
            GlossaryEntry(term="SKU", definition="Stock keeping unit."),
        ]
    )

    assert glossary.render() == "- BRIC: An item attribute, a global item code.\n- SKU: Stock keeping unit."
