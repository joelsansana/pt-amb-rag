from pt_amb_rag.ingestion.chunker import chunk_records


def _base_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "regulation": "dl_150_2015",
        "part": "article",
        "article_num": 8,
        "numero": None,
        "alinea": None,
        "annex_id": None,
        "chapter": "CAPÍTULO I",
        "diploma": "Decreto-Lei n.º 150/2015",
        "diploma_type": "Decreto-Lei",
        "diploma_number": "150/2015",
        "published": "2015-08-05",
        "effective_date": "2015-08-05",
        "eli": "https://data.dre.pt/eli/dec-lei/150/2015/08/05/p/dre/pt/html",
        "lang": "PT",
        "title": "Deveres gerais do operador",
        "text": "1 - O operador é responsável por tomar as medidas "
        "necessárias para evitar a ocorrência de acidentes graves.",
    }
    record.update(overrides)
    return record


def test_anexo_is_split_without_dropping_rows() -> None:
    rows = [
        "Indústrias do sector da energia:",
        "Queima de combustíveis em instalações com uma potência térmica "
        "nominal total igual ou superior a 50 MW;",
        "Refinação de petróleo e de gás;",
        "Produção de coque;",
    ]

    record = _base_record(
        part="annex",
        article_num=None,
        annex_id="I",
        chapter=None,
        title="Limiares",
        text="\n\n".join(row * 30 for row in rows),
    )

    chunks = chunk_records([record])

    assert len(chunks) > 1
    assert all(chunk["part"] == "annex" for chunk in chunks)
    assert all(chunk["annex_id"] == "I" for chunk in chunks)

    combined_text = "\n\n".join(chunk["text"] for chunk in chunks)

    for row in rows:
        expanded_row = row * 30
        assert expanded_row in combined_text


def test_article_gets_pt_contextual_header() -> None:
    record = _base_record()

    chunks = chunk_records([record])

    assert len(chunks) == 1
    chunk = chunks[0]

    assert chunk["part"] == "article"
    assert chunk["article_num"] == 8
    assert chunk["header"].startswith(
        "Artigo 8.º — Decreto-Lei n.º 150/2015, de 5 de agosto"
    )
    assert "CAPÍTULO I" in chunk["header"]
    assert "Deveres gerais do operador" in chunk["header"]
    assert chunk["text"].startswith(chunk["header"])
    assert record["text"] in chunk["text"]


def test_numbered_paragraph_is_locatable_in_the_header() -> None:
    record = _base_record(numero=2, alinea="b")

    chunks = chunk_records([record])

    header = chunks[0]["header"]
    assert "Artigo 8.º, n.º 2, al. b)" in header


def test_annex_header_names_the_annex() -> None:
    record = _base_record(
        part="annex",
        article_num=None,
        annex_id="I",
        chapter=None,
        title="Limiares",
        text="Os limiares estabelecidos neste anexo.",
    )

    chunks = chunk_records([record])

    assert chunks[0]["header"].startswith(
        "Anexo I — Decreto-Lei n.º 150/2015, de 5 de agosto"
    )


def test_preamble_header_is_labelled_in_portuguese() -> None:
    record = _base_record(
        part="preamble", article_num=None, chapter=None, title=""
    )

    chunks = chunk_records([record])

    assert chunks[0]["header"].startswith(
        "Preâmbulo — Decreto-Lei n.º 150/2015"
    )


def test_cross_reference_is_preserved_verbatim() -> None:
    cross_reference = (
        "Nos termos do artigo 12.º do Decreto-Lei n.º 127/2013, o operador "
        "assegura a conformidade."
    )

    record = _base_record(text=cross_reference)

    chunks = chunk_records([record])

    assert len(chunks) == 1
    assert cross_reference in chunks[0]["text"]


def test_empty_subdivision_is_skipped_without_crashing() -> None:
    empty_record = _base_record(text="", title="")
    valid_record = _base_record(
        text="Este é o conteúdo válido de um artigo.",
    )

    chunks = chunk_records([empty_record, valid_record])

    assert len(chunks) == 1
    assert chunks[0]["text"].endswith("Este é o conteúdo válido de um artigo.")


def test_chunks_carry_the_pt_payload_fields() -> None:
    chunks = chunk_records([_base_record()])

    chunk = chunks[0]
    for field in (
        "regulation",
        "part",
        "article_num",
        "numero",
        "alinea",
        "annex_id",
        "chapter",
        "diploma",
        "diploma_type",
        "diploma_number",
        "published",
        "effective_date",
        "eli",
    ):
        assert field in chunk
