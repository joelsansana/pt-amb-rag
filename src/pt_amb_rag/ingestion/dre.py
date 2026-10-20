"""Client and parser for Portugal's Diário da República (DRE).

The DRE website is a JavaScript single-page app: `diariodarepublica.pt`
and `data.dre.pt` return the same 2 KB OutSystems shell for every URL,
so there is no server-rendered document HTML to scrape. The data the
site renders is, however, available from its OutSystems screenservice
API — this module speaks that API directly over HTTP, with no browser
and no HTML scraping.

The contract (verified against the production DRE app):

1. ``GET /dr/moduleservices/moduleversioninfo`` returns the current
   module version token.
2. ``GET /dr/scripts/dr.<Module>.<Screen>.mvc.js`` contains the
   per-action ``apiVersion`` token and the screenservice URL. Reading it
   from the deployed bundle (rather than hardcoding it) means a DRE
   deploy does not break the corpus pipeline.
3. The first ``POST`` to a screenservice URL answers ``301`` and sets
   the ``nr2Users`` cookie, which carries the ``crf`` CSRF token.
4. The second ``POST``, replaying the same body with
   ``x-csrftoken: <crf>``, answers ``200`` with the screen data.

What comes back is already structured: ``DetalheConteudo.Texto`` is the
plain text of the diploma and ``DetalheConteudo.TextoFormatado`` is the
same document as HTML whose ``<p>`` elements carry DRE's own structural
classes (``paragraph-center`` for headings, ``paragraph-bold-center-14px``
for an article's rubric, ``paragraph-normal-text`` for body text).

The parser turns that HTML into the unit schema the rest of the
pipeline expects, following Portuguese legal citation grammar
(``Artigo 12.º``, ``n.º 2``, ``alínea b)``, ``ANEXO I``).

Two details of the transport are load-bearing:

* The screenservice URL separates the module and the screen with a
  slash (``/screenservices/dr/<Module>/<Screen>/<Action>``). With a dot
  instead — the form used in the screen's script filename — the edge
  does not route the request to the screenservice at all and answers
  with a bare ``301`` that carries no session cookie.
* Redirects are never followed: the ``301`` points at ``/dr/error``,
  and following it would silently return the SPA shell instead of the
  JSON body.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from selectolax.parser import HTMLParser

logger = logging.getLogger(__name__)

BASE_URL = "https://diariodarepublica.pt"
APP_ROOT = "/dr"

# Screen that serves a diploma's full text. The module and the screen
# are two separate path segments in the screenservice URL.
SCREEN_MODULE = "Legislacao_Conteudos"
SCREEN_NAME = "Conteudo_Detalhe"
CONTENT_ACTION = "DataActionGetAllConteudoDetalheData"

# The screenservice contract changes when DRE deploys; these are the
# values it used at the time this module was written. `DreClient` reads
# the live values from the deployed bundle and only falls back to these.
FALLBACK_MODULE_VERSION = "CoZj_RUMK1LcmLm1_TcAiw"
FALLBACK_API_VERSION = "B3l5BYZxaLmWUxGR3tmKGw"

REQUEST_TIMEOUT_S = 60.0
USER_AGENT = (
    "pt-amb-rag/0.1 (+corpus ingestion; contact: repository maintainer)"
)

# DRE's own structural classes in `TextoFormatado`.
CLASS_TITLE = "paragraph-title-bold-center-18px"
CLASS_HEADING = "paragraph-center"
CLASS_RUBRIC = "paragraph-bold-center-14px"
CLASS_BODY = "paragraph-normal-text"

# Portuguese legal citation grammar.
_ARTICLE_RE = re.compile(r"^Artigo\s+(?:(\d+)\.?º|único)\b", re.IGNORECASE)
_NUMBER_RE = re.compile("^(\\d+)\\s*[-\\u2013\\u2014]\\s*")
_ALINEA_RE = re.compile(r"^([a-záàâãéêíóôõúç]{1,3})\)\s*")
_CHAPTER_RE = re.compile(r"^(T[ÍI]TULO|CAP[ÍI]TULO|SEC[ÇC][ÃA]O)\s+(.+)$")
_ANNEX_RE = re.compile(r"^ANEXO\s+(.+)$", re.IGNORECASE)


class DreError(Exception):
    """Raised when the DRE screenservice cannot be reached or understood."""


@dataclass(frozen=True)
class DreDocument:
    """Metadata and text for one DRE document."""

    id: str
    title: str
    number: str
    summary: str
    text: str
    formatted_text: str
    published: str
    emitter: str
    series: str
    pages: str
    eli: str
    pdf_url: str
    diploma_type: str
    diploma_type_slug: str

    @property
    def content_hash(self) -> str:
        import hashlib

        payload = f"{self.number}|{self.published}|{self.text}"
        return hashlib.sha256(payload.encode()).hexdigest()


def _csrf_from_jar(jar: Any) -> str | None:
    """Read the `crf` CSRF token out of an `nr2Users` cookie."""
    for cookie in jar:
        if cookie.name != "nr2Users":
            continue
        # The DRE sends the whole cookie value URL-encoded, so
        # `crf%3Dabc%3Buid%3D0` has to be decoded before it can be
        # split into its parts.
        decoded = urllib.parse.unquote(cookie.value)
        for part in decoded.split(";"):
            key, _, value = part.strip().partition("=")
            if key == "crf":
                return value
    return None


def _encode_json(body: dict[str, Any]) -> bytes:
    import json

    return json.dumps(body, ensure_ascii=False).encode("utf-8")


class DreClient:
    """Minimal HTTP client for the DRE screenservice API.

    Holds the session cookies and the version tokens; every call to
    `fetch_document` is one screenservice round trip after the session
    has been established.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        *,
        timeout: float = REQUEST_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Language": "pt-PT",
            },
        )
        self._module_version: str | None = None
        self._api_version: str | None = None

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> DreClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- version tokens -------------------------------------------------

    def module_version(self, *, refresh: bool = False) -> str:
        """Current module version token, cached per client."""
        if self._module_version and not refresh:
            return self._module_version

        url = f"{self.base_url}{APP_ROOT}/moduleservices/moduleversioninfo"
        response = self.client.get(url, params={"t": "0"})
        response.raise_for_status()
        token = response.json().get("versionToken")
        if not token:
            raise DreError("moduleversioninfo returned no versionToken")
        self._module_version = str(token)
        return self._module_version

    def api_version(self, *, refresh: bool = False) -> str:
        """Per-action API version token, read from the deployed bundle.

        The token is baked into the screen's `.mvc.js`. Reading it live
        keeps the client working across DRE deploys; the constant in
        this module is only a fallback for when the bundle cannot be
        fetched.
        """
        if self._api_version and not refresh:
            return self._api_version

        url = (
            f"{self.base_url}{APP_ROOT}/scripts/"
            f"dr.{SCREEN_MODULE}.{SCREEN_NAME}.mvc.js"
        )
        try:
            response = self.client.get(
                url, headers={"Accept": "text/javascript,*/*"}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "could not read the screen bundle (%s); falling back to the "
                "pinned apiVersion",
                exc,
            )
            self._api_version = FALLBACK_API_VERSION
            return self._api_version

        pattern = (
            r'callDataAction\("'
            + re.escape(CONTENT_ACTION)
            + r'",\s*"[^"]+",\s*"[^"]+",\s*"([^"]+)"'
        )
        match = re.search(pattern, response.text)
        if not match:
            logger.warning(
                "screen bundle did not expose an apiVersion for %s; "
                "falling back to the pinned one",
                CONTENT_ACTION,
            )
            self._api_version = FALLBACK_API_VERSION
            return self._api_version

        version = match.group(1)
        self._api_version = version
        return version

    # -- requests -------------------------------------------------------

    def _screen_payload(
        self, *, tipo: str, key: str, parte_id: str = ""
    ) -> dict[str, Any]:
        """The screen-data payload the DRE SPA sends for a document."""
        return {
            "versionInfo": {
                "moduleVersion": self.module_version(),
                "apiVersion": self.api_version(),
            },
            "viewName": f"{SCREEN_MODULE}.{SCREEN_NAME}",
            "screenData": {
                "variables": {
                    "Tipo": tipo,
                    "Key": key,
                    "ParteId": parte_id,
                    "KeyAux": "",
                    "ParteIdAux": "0",
                    "ELI_HTML": "",
                    "TextoFormatadoAux": "",
                    "EmissorDesignacao": "",
                    "CanShowAllData": False,
                    "IndexPT": 0,
                    "ShowDiplomaAtoOriginalTexto": True,
                    "IsShowConteudoRelacionado": True,
                    "IsRefinarPesquisa": True,
                    "ShowScrollButtons": True,
                    "IsPrint": False,
                    "IsLoadingPDFAtoOriginal": False,
                    "IsCopyTextLoading": False,
                    "HasJurisprudenciaAssociadaVar": False,
                    "IsPageTracked": False,
                    "ShowResumoPT": False,
                    "ShowResumoEN": False,
                    "ShowDiplomaFragmentacaoIndice": False,
                    "ShowDiplomaFragmentacaoTextoCompleto": False,
                    "ShowDiplomaFragmentacaoMenuCabecalho": False,
                    "Pesquisa": "",
                    "Comes": "",
                    "FragmentoVersaoIdAux": 0,
                }
            },
            "clientVariables": {},
        }

    def _post(self, action: str, body: dict[str, Any]) -> dict[str, Any]:
        url = (
            f"{self.base_url}{APP_ROOT}/screenservices/dr/"
            f"{SCREEN_MODULE}/{SCREEN_NAME}/{action}"
        )
        payload = _encode_json(body)

        # The DRE's session handshake is: a POST without a session is
        # answered with 301 and a `nr2Users` cookie carrying the CSRF
        # token; the same POST replayed with that token in the
        # `x-csrftoken` header succeeds.
        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json",
        }

        response = self.client.post(
            url,
            params={"_": uuid.uuid4().hex},
            content=payload,
            headers=headers,
        )

        if response.status_code in (301, 302, 303):
            token = _csrf_from_jar(response.cookies.jar)
            if not token:
                raise DreError(
                    "the DRE did not issue a session cookie; the "
                    f"screenservice URL may be wrong ({url})"
                )
            headers["x-csrftoken"] = token
            response = self.client.post(
                url,
                params={"_": uuid.uuid4().hex},
                content=payload,
                headers=headers,
            )

        if response.status_code != 200:
            raise DreError(
                f"DRE screenservice returned {response.status_code} "
                f"for action {action}"
            )

        try:
            payload_json = response.json()
        except ValueError as exc:
            raise DreError(
                f"DRE screenservice returned a non-JSON body for {action}"
            ) from exc

        data = payload_json.get("data")
        if not data:
            raise DreError(
                f"DRE screenservice returned no data for action {action}"
            )
        return data

    def fetch_document(self, tipo: str, key: str) -> DreDocument:
        """Fetch one diploma by its DRE type slug and content key.

        `tipo` is the URL segment used by `/dr/detalhe/<tipo>/<key>`
        (e.g. `decreto-lei`); `key` is the trailing identifier of that
        same URL (e.g. `150-2015-69951097`).
        """
        body = self._screen_payload(tipo=tipo, key=key)
        data = self._post(CONTENT_ACTION, body)
        detail = data.get("DetalheConteudo") or {}
        if not detail:
            raise DreError(f"DRE returned no content for {tipo}/{key}")
        return _document_from_detail(detail)


def _document_from_detail(detail: dict[str, Any]) -> DreDocument:
    return DreDocument(
        id=str(detail.get("Id") or ""),
        title=str(detail.get("Titulo") or "").strip(),
        number=str(detail.get("Numero") or ""),
        summary=str(detail.get("Sumario") or ""),
        text=str(detail.get("Texto") or ""),
        formatted_text=str(detail.get("TextoFormatado") or ""),
        published=str(detail.get("DataPublicacao") or ""),
        emitter=str(detail.get("Emissor") or ""),
        series=str(detail.get("Serie") or ""),
        pages=str(detail.get("Pagina") or ""),
        eli=str(detail.get("ELI") or ""),
        pdf_url=str(detail.get("URL_PDF") or ""),
        diploma_type=str(detail.get("TipoDiploma") or ""),
        diploma_type_slug=str(detail.get("TipoDiplomaAcronimo") or ""),
    )


# -- parsing ------------------------------------------------------------


@dataclass
class StructuralUnit:
    """One structural element of a diploma (article, chapter, annex)."""

    part: str
    text: str
    article_num: int | None = None
    numero: int | None = None
    alinea: str | None = None
    chapter: str | None = None
    annex_id: str | None = None
    title: str = ""


def _class_of(node: Any) -> str:
    classes = (node.attributes.get("class") or "").split()
    for name in (CLASS_TITLE, CLASS_HEADING, CLASS_RUBRIC, CLASS_BODY):
        if name in classes:
            return name
    return CLASS_BODY


def _node_text(node: Any) -> str:
    return " ".join(node.text().split())


def parse_formatted_text(formatted_text: str) -> list[StructuralUnit]:
    """Turn DRE's `TextoFormatado` HTML into structural units.

    DRE's markup is regular: a `<p class="paragraph-center">` is a
    structural heading (`Artigo 12.º`, `CAPÍTULO II`, `ANEXO I`), the
    `<p class="paragraph-bold-center-14px">` right after it is that
    heading's rubric, and `paragraph-normal-text` is body text.
    """
    if not formatted_text.strip():
        return []

    document = HTMLParser(formatted_text)
    units: list[StructuralUnit] = []
    current: StructuralUnit | None = None
    pending_title = ""
    chapter: str | None = None
    annex_id: str | None = None

    def flush() -> None:
        nonlocal current
        if current is not None and current.text.strip():
            units.append(current)
        current = None

    for node in document.css("p"):
        kind = _class_of(node)
        text = _node_text(node)
        if not text:
            continue

        if kind == CLASS_TITLE:
            pending_title = text
            continue

        if kind == CLASS_HEADING:
            article_match = _ARTICLE_RE.match(text)
            chapter_match = _CHAPTER_RE.match(text)
            annex_match = _ANNEX_RE.match(text)

            if article_match:
                flush()
                raw_num = article_match.group(1)
                current = StructuralUnit(
                    part="article",
                    text="",
                    article_num=int(raw_num) if raw_num else None,
                    title=pending_title,
                    chapter=chapter,
                    annex_id=annex_id,
                )
                pending_title = ""
            elif annex_match:
                flush()
                annex_id = annex_match.group(1).strip()
                current = StructuralUnit(
                    part="annex",
                    text="",
                    annex_id=annex_id,
                    chapter=chapter,
                )
                pending_title = ""
            elif chapter_match:
                flush()
                chapter = text.strip()
                current = StructuralUnit(
                    part="chapter", text="", chapter=chapter
                )
                pending_title = ""
            else:
                flush()
                current = StructuralUnit(part="section", text="")
                pending_title = ""
            continue

        if kind == CLASS_RUBRIC:
            if current is not None and not current.title:
                current.title = text
            continue

        # Body text.
        if current is None:
            current = StructuralUnit(part="preamble", text="")

        number_match = _NUMBER_RE.match(text)
        alinea_match = _ALINEA_RE.match(text)

        if number_match and current.part == "article":
            current.numero = int(number_match.group(1))
        elif alinea_match and current.part == "article":
            current.alinea = alinea_match.group(1)

        current.text = f"{current.text}\n\n{text}" if current.text else text

    flush()
    return units


def units_from_document(
    document: DreDocument, *, corpus_id: str, effective_date: str
) -> list[dict[str, Any]]:
    """Build the unit records the chunker consumes.

    The schema mirrors the upstream `eur_lex` units so that the
    chunker, the vector store, and the golden-set machinery survive the
    fork: `regulation` carries the corpus id, `article_num` the article
    number, `part` the structural kind.
    """
    units: list[dict[str, Any]] = []
    for structural in parse_formatted_text(document.formatted_text):
        text = structural.text.strip()
        if not text:
            continue
        units.append(
            {
                "regulation": corpus_id,
                "part": structural.part,
                "article_num": structural.article_num,
                "numero": structural.numero,
                "alinea": structural.alinea,
                "annex_id": structural.annex_id,
                "chapter": structural.chapter,
                "diploma": document.title,
                "diploma_type": document.diploma_type,
                "diploma_number": document.number,
                "published": document.published,
                "effective_date": effective_date,
                "eli": document.eli,
                "lang": "PT",
                "title": structural.title,
                "text": text,
            }
        )
    return units


def save_document(
    document: DreDocument, path: Path, *, corpus_id: str, effective_date: str
) -> Path:
    """Write a fetched diploma to disk as newline-delimited units."""
    import json

    units = units_from_document(
        document, corpus_id=corpus_id, effective_date=effective_date
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(u, ensure_ascii=False) for u in units) + "\n",
        encoding="utf-8",
    )
    return path
