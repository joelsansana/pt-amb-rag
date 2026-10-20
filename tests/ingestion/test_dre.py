"""Parser tests for the DRE screenservice payloads.

The fixture is a trimmed, verbatim slice of the DRE's own
`TextoFormatado` HTML for Decreto-Lei n.º 150/2015 (Seveso III). It is
committed on purpose: the DRE's markup is the contract this parser
depends on, and a silent change in that markup must fail a test rather
than quietly degrade the corpus.
"""

from pathlib import Path

from pt_amb_rag.ingestion.dre import (
    parse_formatted_text,
    units_from_document,
)

FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "dre"
    / "dl_150_2015_snippet.html"
)


def _fixture_text() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def test_empty_input_yields_no_units():
    assert parse_formatted_text("") == []
    assert parse_formatted_text("   \n  ") == []


def test_preamble_is_recognised():
    units = parse_formatted_text(_fixture_text())

    assert units
    assert units[0].part == "preamble"
    assert "acidentes de grande dimensão" in units[0].text


def test_articles_are_split_and_numbered():
    units = parse_formatted_text(_fixture_text())

    articles = [u for u in units if u.part == "article"]
    assert [a.article_num for a in articles][:3] == [1, 2, 3]

    first = articles[0]
    assert first.title == "Objeto"
    assert first.chapter == "CAPÍTULO I"
    assert first.text.startswith("O presente decreto-lei")


def test_article_rubric_is_the_title():
    units = parse_formatted_text(_fixture_text())

    articles = {a.article_num: a for a in units if a.part == "article"}
    assert articles[2].title == "Âmbito de aplicação"
    assert articles[3].title == "Definições"


def test_numbered_paragraphs_are_recorded():
    units = parse_formatted_text(_fixture_text())

    articles = {a.article_num: a for a in units if a.part == "article"}
    # Article 2 opens with "1 - ..." and its second numbered paragraph
    # is "2 - ...".
    assert articles[2].numero == 2
    assert "1 - " in articles[2].text


def test_alinea_is_recorded():
    units = parse_formatted_text(_fixture_text())

    articles = {a.article_num: a for a in units if a.part == "article"}
    # Article 3 is a definitions article: "a) «Acidente grave»..."
    assert articles[3].alinea is not None
    assert "a)" in articles[3].text


def test_annex_is_recognised():
    units = parse_formatted_text(_fixture_text())

    annexes = [u for u in units if u.part == "annex"]
    assert annexes
    assert annexes[0].annex_id == "I"
    assert annexes[0].text.strip()


def test_every_unit_carries_text():
    for unit in parse_formatted_text(_fixture_text()):
        assert unit.text.strip()


def test_units_from_document_maps_the_chunker_schema():
    from pt_amb_rag.ingestion.dre import DreDocument

    document = DreDocument(
        id="69951097",
        title="Decreto-Lei n.º 150/2015",
        number="150/2015",
        summary="",
        text="",
        formatted_text=_fixture_text(),
        published="2015-08-05",
        emitter="MAOTE",
        series="I",
        pages="5378 - 5401",
        eli="https://data.dre.pt/eli/dec-lei/150/2015/08/05/p/dre/pt/html",
        pdf_url="https://files.diariodarepublica.pt/1s/2015/08/15100/0537805401.pdf",
        diploma_type="Decreto-Lei",
        diploma_type_slug="dec-lei",
    )

    units = units_from_document(
        document, corpus_id="dl_150_2015", effective_date="2015-08-05"
    )

    assert units
    for unit in units:
        assert unit["regulation"] == "dl_150_2015"
        assert unit["diploma"] == "Decreto-Lei n.º 150/2015"
        assert unit["diploma_number"] == "150/2015"
        assert unit["published"] == "2015-08-05"
        assert unit["lang"] == "PT"
        assert unit["eli"].startswith("https://data.dre.pt/eli/")
        assert unit["part"] in {
            "preamble",
            "article",
            "annex",
            "chapter",
            "section",
        }
