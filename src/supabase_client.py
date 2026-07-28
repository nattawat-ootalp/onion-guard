"""
Minimal Supabase REST client (urllib only — no extra dependency).

Written against PostgREST + Storage HTTP APIs directly rather than the
official `supabase` package because this code has to ship to size-constrained
targets: the Render/Vercel bundle is already near its limit with OpenCV and
scikit-learn, and the Pi deployment deliberately keeps its dependency list
short. Everything used here is in the standard library.

KEY HANDLING
The service_role key bypasses Row Level Security entirely — it can read,
modify and delete anything. It is only ever read from the environment on the
SERVER. Never hand it to a browser; the anon key exists for that, and RLS
restricts anon to SELECT (see supabase/schema.sql).
"""
import json
import mimetypes
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"


def load_env(path=ENV_PATH):
    """Read .env into os.environ without adding a dotenv dependency.

    Existing environment variables win, so a real deployment (Render, Vercel)
    can set them properly and a stray .env file cannot override production
    configuration.
    """
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


class SupabaseError(RuntimeError):
    pass


class SupabaseClient:
    def __init__(self, url=None, key=None, timeout=20):
        load_env()
        self.url = (url or os.environ.get("SUPABASE_URL", "")).rstrip("/")
        self.key = key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
        self.timeout = timeout
        if not self.url or not self.key:
            raise SupabaseError(
                "ไม่พบ SUPABASE_URL หรือคีย์ — ตั้งใน .env หรือ environment variable"
            )

    # ------------------------------------------------------------------ core

    def _request(self, method, path, *, data=None, headers=None, raw=None, query=None):
        url = f"{self.url}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query, doseq=True)

        h = {"apikey": self.key, "Authorization": f"Bearer {self.key}"}
        if data is not None:
            h["Content-Type"] = "application/json"
            body = json.dumps(data).encode()
        else:
            body = raw
        if headers:
            h.update(headers)

        req = urllib.request.Request(url, data=body, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
                if not payload:
                    return None
                ctype = resp.headers.get("Content-Type", "")
                return json.loads(payload) if "json" in ctype else payload
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            raise SupabaseError(f"{method} {path} -> HTTP {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise SupabaseError(f"เชื่อมต่อ Supabase ไม่ได้: {e.reason}") from None

    # ------------------------------------------------------------------ table

    def insert(self, table, row, *, return_row=True):
        headers = {"Prefer": "return=representation" if return_row else "return=minimal"}
        out = self._request("POST", f"/rest/v1/{table}", data=[row], headers=headers)
        return (out or [{}])[0] if return_row else None

    def update(self, table, match, patch):
        query = {k: f"eq.{v}" for k, v in match.items()}
        return self._request("PATCH", f"/rest/v1/{table}", data=patch,
                             headers={"Prefer": "return=representation"}, query=query)

    def select(self, table, *, columns="*", order=None, limit=None, filters=None):
        query = {"select": columns}
        if order:
            query["order"] = order
        if limit:
            query["limit"] = limit
        if filters:
            query.update(filters)
        return self._request("GET", f"/rest/v1/{table}", query=query) or []

    def count(self, table, filters=None):
        """Row count via the Content-Range header, so no rows are transferred."""
        query = {"select": "id"}
        if filters:
            query.update(filters)
        url = f"{self.url}/rest/v1/{table}?" + urllib.parse.urlencode(query, doseq=True)
        req = urllib.request.Request(url, headers={
            "apikey": self.key, "Authorization": f"Bearer {self.key}",
            "Prefer": "count=exact", "Range": "0-0",
        })
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                rng = resp.headers.get("Content-Range", "")  # e.g. "0-0/57"
                return int(rng.split("/")[-1]) if "/" in rng else 0
        except urllib.error.HTTPError as e:
            raise SupabaseError(f"count {table} -> HTTP {e.code}") from None

    # ---------------------------------------------------------------- storage

    def upload(self, bucket, path, content, *, content_type=None, upsert=True):
        ctype = content_type or mimetypes.guess_type(path)[0] or "application/octet-stream"
        headers = {"Content-Type": ctype}
        if upsert:
            headers["x-upsert"] = "true"
        self._request("POST", f"/storage/v1/object/{bucket}/{path}",
                      raw=content, headers=headers)
        return path

    def signed_url(self, bucket, path, expires_in=3600):
        """Time-limited URL for a private bucket.

        The bucket is private by default so scan images are not world-readable
        by guessing a path; the server mints a short-lived link instead.
        """
        out = self._request("POST", f"/storage/v1/object/sign/{bucket}/{path}",
                            data={"expiresIn": expires_in})
        signed = (out or {}).get("signedURL", "")
        return f"{self.url}/storage/v1{signed}" if signed else None

    # ------------------------------------------------------------------ misc

    def health(self):
        try:
            self._request("GET", "/rest/v1/")
            return True
        except SupabaseError:
            return False


if __name__ == "__main__":
    import sys

    c = SupabaseClient()
    print(f"URL      : {c.url}")
    print(f"key type : {'service_role' if 'service_role' in (c.key or '') or os.environ.get('SUPABASE_SERVICE_ROLE_KEY') else 'anon'}")
    print(f"reachable: {c.health()}")
    try:
        n = c.count("scans")
        print(f"scans    : {n} แถว")
    except SupabaseError as e:
        print(f"scans    : ยังไม่มีตาราง — รัน supabase/schema.sql ใน SQL Editor ก่อน")
        print(f"           ({e})")
        sys.exit(1)
