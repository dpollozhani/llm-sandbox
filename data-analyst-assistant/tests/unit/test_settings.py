from data_analyst.config.settings import Glossary, GlossaryEntry, PowerBiCatalog, SemanticModelConfig, get_glossary

_CATALOG = PowerBiCatalog(
    semantic_models=[
        SemanticModelConfig(model_name="Sales Analytics", dataset_id="ds-sales"),
        SemanticModelConfig(model_name="HQ financial costs", dataset_id="ds-hq"),
    ]
)


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


def test_catalog_subset_matches_by_model_name():
    subset = _CATALOG.subset(["Sales Analytics"])
    assert [m.model_name for m in subset.semantic_models] == ["Sales Analytics"]


def test_catalog_subset_matches_by_dataset_id():
    subset = _CATALOG.subset(["ds-hq"])
    assert [m.model_name for m in subset.semantic_models] == ["HQ financial costs"]


def test_catalog_subset_drops_unmatched_identifiers():
    subset = _CATALOG.subset(["Sales Analytics", "Nonexistent Model"])
    assert [m.model_name for m in subset.semantic_models] == ["Sales Analytics"]
