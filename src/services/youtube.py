"""A ready folder (services/publish.py) -> YouTube, with the YouTube Data API v3
(scripts/upload_youtube.py). Outside the pipeline: it only reads the folder.

    video.mp4 + metadata.json + descripcion.md  -> videos.insert (title, description, tags,
                                                   category, language, privacy / publish_at,
                                                   made for kids, AI content, licence)
    miniatura.jpg                               -> thumbnails.set (needs a phone-verified channel)
    subtitulos.srt                              -> captions.insert
Then the folder moves to youtube.uploaded_dir with youtube.json (the video id and link), so it
is never uploaded twice.

Setup once (config.yaml: youtube.client_secret, youtube.token):
    1. console.cloud.google.com: a project, enable "YouTube Data API v3" (and "YouTube Analytics
       API" for scripts/retention.py).
    2. OAuth consent screen: External, add your Google account as a test user.
    3. Credentials > OAuth client ID > Desktop app > download the JSON to youtube.client_secret.
    4. The first upload opens the browser to choose your channel; the token is saved for later.
Until Google audits the API project, its uploads stay PRIVATE (publish them in YouTube Studio).
Quota: 10,000 units/day; an upload ~1,600, a thumbnail ~50, subtitles ~400.

Needs: pip install google-api-python-client google-auth-oauthlib (optional extra "youtube").
"""
import json
import shutil
from pathlib import Path

from llmkit import config
from services.settings import load_settings

SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube.force-ssl",       # force-ssl: subtitles
          "https://www.googleapis.com/auth/yt-analytics.readonly"]  # retention (services/retention.py)
TAGS_MAX_CHARS = 450              # YouTube's limit is 500 (it counts quotes around tags with spaces)
CHUNK = 8 * 1024 * 1024


def _options():
    return load_settings().get("youtube", {})


def client(api="youtube", version="v3"):
    """An authorized YouTube API client (the first time it opens the browser to choose the channel).
    client("youtubeAnalytics", "v2"): the Analytics API (enable it in the same Cloud project)."""
    try:
        from google.auth.exceptions import RefreshError
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        raise SystemExit("Faltan las librerías de Google: uv pip install google-api-python-client google-auth-oauthlib")
    options = _options()
    secret = config.BASE / options.get("client_secret", "data/youtube/client_secret.json")
    token = config.BASE / options.get("token", "data/youtube/token.json")
    creds = Credentials.from_authorized_user_file(str(token), SCOPES) if token.exists() else None
    if creds and not set(SCOPES) <= set(json.loads(token.read_text(encoding="utf-8")).get("scopes") or []):
        print("  · la autorización guardada no incluye todos los permisos: se abre el navegador para autorizar de nuevo")
        creds = None                      # a token from before a scope was added: it would answer 403
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:          # a consent screen in "Testing": Google revokes it after 7 days
                print("  · la autorización de YouTube ha caducado: se abre el navegador para autorizar de nuevo")
                creds = None
        if not creds or not creds.valid:
            if not secret.exists():
                raise SystemExit(f"Falta la credencial OAuth en {secret} (ver src/services/youtube.py)")
            creds = InstalledAppFlow.from_client_secrets_file(str(secret), SCOPES).run_local_server(port=0)
        token.parent.mkdir(parents=True, exist_ok=True)
        token.write_text(creds.to_json(), encoding="utf-8")
    return build(api, version, credentials=creds)


def limited_tags(tags):
    out, total = [], 0
    for t in tags:
        cost = len(t) + (2 if " " in t else 0) + (1 if out else 0)
        if total + cost > TAGS_MAX_CHARS:
            break
        out.append(t)
        total += cost
    return out


def video_body(meta, description):
    privacy = meta.get("privacy", "private")
    status = {"privacyStatus": "private" if meta.get("publish_at") else privacy,
              "selfDeclaredMadeForKids": bool(meta.get("made_for_kids", False)),
              "containsSyntheticMedia": bool(meta.get("synthetic_media", False)),
              "license": meta.get("license", "youtube"),
              "embeddable": True}
    if meta.get("publish_at"):
        status["publishAt"] = meta["publish_at"]           # ISO 8601; it is private until then
    language = meta.get("default_language", "es")
    return {"snippet": {"title": meta["title"], "description": description, "tags": limited_tags(meta.get("tags", [])),
                        "categoryId": str(meta.get("category_id", "28")), "defaultLanguage": language,
                        "defaultAudioLanguage": language},
            "status": status}


def upload(folder, privacy=None):
    """Uploads the folder's video with all its metadata; returns the video id."""
    folder = Path(folder)
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if privacy:
        meta["privacy"] = privacy
    description = (folder / meta.get("description", "descripcion.md")).read_text(encoding="utf-8").strip()
    youtube = client()                                   # first: it says what is missing (libraries, credential)
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseUpload

    # Every file is opened in a `with`: MediaFileUpload leaves it open, and Windows then refuses
    # to move the folder to uploaded_dir ("Acceso denegado")
    with open(folder / meta.get("video", "video.mp4"), "rb") as f:
        request = youtube.videos().insert(part="snippet,status", body=video_body(meta, description),
                                          media_body=MediaIoBaseUpload(f, mimetype="video/mp4", chunksize=CHUNK,
                                                                       resumable=True))
        response = None
        while response is None:
            progress, response = request.next_chunk()
            if progress:
                print(f"  · subiendo: {progress.progress() * 100:.0f} %")
    video_id = response["id"]
    print(f"  · vídeo subido: https://youtu.be/{video_id} ({response['status']['privacyStatus']})")

    warnings = []
    if meta.get("thumbnail") and (folder / meta["thumbnail"]).exists():
        try:
            with open(folder / meta["thumbnail"], "rb") as f:
                youtube.thumbnails().set(videoId=video_id,
                                         media_body=MediaIoBaseUpload(f, mimetype="image/jpeg")).execute()
            print("  · miniatura puesta")
        except HttpError as e:                                 # e.g. a new channel not verified yet
            warnings.append(f"miniatura: {e}")
    if meta.get("subtitles") and (folder / meta["subtitles"]).exists():
        try:
            with open(folder / meta["subtitles"], "rb") as f:
                youtube.captions().insert(
                    part="snippet",
                    body={"snippet": {"videoId": video_id, "language": meta.get("default_language", "es"),
                                      "name": "Español", "isDraft": False}},
                    media_body=MediaIoBaseUpload(f, mimetype="application/octet-stream"),
                ).execute()
            print("  · subtítulos puestos")
        except HttpError as e:
            warnings.append(f"subtítulos: {e}")
    for w in warnings:
        print(f"  ⚠️  {w}")

    (folder / "youtube.json").write_text(json.dumps(
        {"id": video_id, "url": f"https://youtu.be/{video_id}", "studio": f"https://studio.youtube.com/video/{video_id}/edit",
         "privacy": response["status"]["privacyStatus"], "warnings": warnings}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    done = config.BASE / _options().get("uploaded_dir", "data/subidos")
    done.mkdir(parents=True, exist_ok=True)
    shutil.move(str(folder), str(done / folder.name))
    print(f"  · carpeta movida a {config.readable_path(done / folder.name)}")
    return video_id
