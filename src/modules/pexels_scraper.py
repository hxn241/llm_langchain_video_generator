"""
pexels_scraper.py
=================

Abre cada página de Pexels (vídeo o foto), localiza el botón de descarga con un
XPath (por defecto el span "Free download") y guarda el fichero.

Usa Playwright porque Pexels renderiza con JS y está detrás de Cloudflare:
`requests` + `lxml` suele recibir un 403 o un HTML sin el botón.

Instalación:
    pip install playwright
    playwright install chromium

Uso desde CLI:
    python pexels_scraper.py media_escenas.json -o descargas
    python pexels_scraper.py https://www.pexels.com/video/...-6079428/ -o descargas
    python pexels_scraper.py urls.txt --headed        # ventana visible (útil si Cloudflare bloquea)

Uso como módulo:
    from pexels_scraper import load_jobs, download_all
    results = download_all(load_jobs("media_escenas.json"), "descargas")
"""
from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PWTimeout

DEFAULT_XPATH = "//main//span[(contains(text(),'Free'))]"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

log = logging.getLogger("pexels_scraper")


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    url: str
    name: str | None = None  # prefijo del fichero (p. ej. el id de la escena)


@dataclass
class Result:
    url: str
    path: Path | None = None
    error: str | None = None
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return self.path is not None


# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #
def load_jobs(source: str | Path | Iterable[str]) -> list[Job]:
    """
    Acepta:
      - una URL suelta
      - un .txt con una URL por línea
      - un .json: lista de URLs, o lista de objetos con `page_url`
        (o `media.page_url`) y opcionalmente `id`
      - un iterable de URLs
    """
    if not isinstance(source, (str, Path)):
        return [Job(u) for u in source]

    s = str(source)
    if s.startswith("http"):
        return [Job(s)]

    path = Path(s)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        jobs: list[Job] = []
        for item in data:
            if isinstance(item, str):
                jobs.append(Job(item))
                continue
            url = item.get("page_url") or (item.get("media") or {}).get("page_url")
            if not url:
                log.warning("Entrada sin page_url, se ignora: %s", item)
                continue
            name = f"{int(item['id']):02d}" if str(item.get("id", "")).isdigit() else item.get("id")
            jobs.append(Job(url, name))
        return jobs

    lines = path.read_text(encoding="utf-8").splitlines()
    return [Job(l.strip()) for l in lines if l.strip() and not l.startswith("#")]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _safe(name: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", name).strip("_") or "file"


def _target_name(job: Job, filename: str) -> str:
    filename = _safe(filename)
    return f"{job.name}_{filename}" if job.name else filename


def _already_downloaded(job: Job, out_dir: Path) -> Path | None:
    if not job.name:
        return None
    hits = [p for p in out_dir.glob(f"{job.name}_*") if p.stat().st_size > 0]
    return hits[0] if hits else None


def _dismiss_overlays(page: Page) -> None:
    """Cierra banners/modales que puedan tapar el botón (rechazando cookies opcionales)."""
    for label in ("Reject", "Rechazar", "Decline", "Close", "Cerrar"):
        btn = page.get_by_role("button", name=re.compile(label, re.I)).first
        try:
            if btn.is_visible(timeout=500):
                btn.click(timeout=1000)
        except Exception:
            pass
    page.keyboard.press("Escape")


def _filename_from_response(url: str, headers: dict[str, str]) -> str:
    cd = headers.get("content-disposition", "")
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", cd, re.I)
    if m:
        return unquote(m.group(1))
    name = Path(urlparse(url).path).name or "download"
    if not Path(name).suffix:
        ext = mimetypes.guess_extension(headers.get("content-type", "").split(";")[0].strip())
        name += ext or ""
    return name


def _fetch_href(page: Page, href: str, job: Job, out_dir: Path, timeout_ms: int) -> Path:
    """Plan B: descarga el href del <a> padre reutilizando las cookies del navegador."""
    resp = page.context.request.get(href, timeout=timeout_ms, max_redirects=10)
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status} al descargar {href}")
    dest = out_dir / _target_name(job, _filename_from_response(resp.url, resp.headers))
    dest.write_bytes(resp.body())
    return dest


# --------------------------------------------------------------------------- #
# Núcleo
# --------------------------------------------------------------------------- #
def download_one(
    page: Page,
    job: Job,
    out_dir: Path,
    xpath: str = DEFAULT_XPATH,
    timeout_ms: int = 60_000,
) -> Path:
    page.goto(job.url, wait_until="domcontentloaded", timeout=timeout_ms)
    _dismiss_overlays(page)

    target = page.locator(f"xpath={xpath}").first
    target.wait_for(state="visible", timeout=timeout_ms)
    href = target.evaluate("el => { const a = el.closest('a'); return a ? a.href : null; }")

    # Plan A: clic en el botón y capturar el evento de descarga del navegador
    try:
        with page.expect_download(timeout=timeout_ms) as info:
            target.click(timeout=timeout_ms)
        dl = info.value
        if dl.failure():
            raise RuntimeError(dl.failure())
        dest = out_dir / _target_name(job, dl.suggested_filename)
        dl.save_as(dest)
        return dest
    except (PWTimeout, RuntimeError) as exc:
        if not href:
            raise RuntimeError(f"El clic no generó descarga y no hay enlace: {exc}") from exc
        log.info("Sin evento de descarga, probando el enlace directo: %s", href)
        return _fetch_href(page, href, job, out_dir, timeout_ms)


def download_all(
    jobs: Iterable[Job],
    out_dir: str | Path = "descargas",
    xpath: str = DEFAULT_XPATH,
    headless: bool = True,
    timeout_ms: int = 60_000,
    retries: int = 2,
    delay: tuple[float, float] = (1.5, 4.0),
    skip_existing: bool = True,
    channel: str | None = None,
) -> list[Result]:
    """channel: un navegador instalado en el sistema ("msedge", "chrome") en vez del Chromium
    propio de Playwright (que hay que descargar con `playwright install chromium`)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless, channel=channel)
        context = browser.new_context(
            accept_downloads=True,
            user_agent=USER_AGENT,
            locale="en-US",
            viewport={"width": 1440, "height": 900},
        )
        page = context.new_page()

        for i, job in enumerate(jobs, 1):
            if skip_existing and (existing := _already_downloaded(job, out)):
                log.info("[%d] Ya existe, se salta: %s", i, existing.name)
                results.append(Result(job.url, existing, skipped=True))
                continue

            last_err: Exception | None = None
            for attempt in range(1, retries + 2):
                try:
                    path = download_one(page, job, out, xpath, timeout_ms)
                    size_mb = path.stat().st_size / 1_048_576
                    log.info("[%d] OK %s (%.1f MB)", i, path.name, size_mb)
                    results.append(Result(job.url, path))
                    last_err = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    log.warning("[%d] Intento %d fallido (%s): %s", i, attempt, job.url, exc)
                    time.sleep(2 * attempt)

            if last_err is not None:
                results.append(Result(job.url, error=str(last_err)))

            time.sleep(random.uniform(*delay))  # no machacar el servidor

        context.close()
        browser.close()

    ok = sum(r.ok for r in results)
    log.info("Terminado: %d/%d descargados en %s", ok, len(results), out.resolve())
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Descarga vídeos/imágenes de Pexels vía XPath.")
    p.add_argument("source", help="URL, .txt con URLs o .json (lista o media_escenas.json)")
    p.add_argument("-o", "--out", default="descargas", help="Carpeta de salida")
    p.add_argument("--xpath", default=DEFAULT_XPATH, help="XPath del botón de descarga")
    p.add_argument("--headed", action="store_true", help="Mostrar el navegador")
    p.add_argument("--channel", default=None,
                   help='Navegador instalado: "msedge" o "chrome" (por defecto, el Chromium de Playwright)')
    p.add_argument("--timeout", type=int, default=60, help="Timeout por acción (s)")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--no-skip", action="store_true", help="Volver a descargar aunque exista")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    results = download_all(
        load_jobs(args.source),
        out_dir=args.out,
        xpath=args.xpath,
        headless=not args.headed,
        timeout_ms=args.timeout * 1000,
        retries=args.retries,
        skip_existing=not args.no_skip,
        channel=args.channel,
    )
    failed = [r for r in results if not r.ok]
    for r in failed:
        print(f"FALLO  {r.url}\n       {r.error}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()