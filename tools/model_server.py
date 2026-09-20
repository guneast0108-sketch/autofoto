#!/usr/bin/env python3
"""앱이 기대하는 모델 서버를 맥에서 띄운다. 표준 라이브러리만 쓴다.

    GET /models                           → ["mobilenetv2", ...]  (JSON)
    GET /download-model?model_name=<name> → zip { model.tflite, labels.txt }
    GET /eval/list                        → [{"file": ..., "label_index": 0~999}, ...]   (--eval)
    GET /eval/image?name=<file>           → JPEG 원본 바이트                            (--eval)

모델은 tools/models/<name>/{model.tflite,labels.txt} 에서 읽는다
(tools/fetch_model.py가 만들어 준다).

    python3 tools/fetch_model.py
    python3 tools/model_server.py          # 0.0.0.0:9000
    flutter run --profile --dart-define=MODEL_SERVER=http://<출력된 주소>:9000

폰과 맥이 같은 와이파이에 있어야 한다.

--eval 을 주면 ImageNet 클래스당 1장(tools/data/imagenet_sample_files.txt, 약 51MB)을
tools/.cache/imagenet-sample 에 받아 두고 앱의 「디코더 정확도 평가」 버튼에 내려준다.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import socket
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REQUIRED = ("model.tflite", "labels.txt")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
TOOLS = Path(__file__).resolve().parent
EVAL_LIST = TOOLS / "data" / "imagenet_sample_files.txt"
EVAL_CACHE = TOOLS / ".cache" / "imagenet-sample"
EVAL_BASE_URL = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"


def prepare_eval_set(cache: Path) -> list[dict]:
    """평가 사진을 받아 두고 받은 것만 목록으로 돌려준다 (check_normalization.py와 같은 캐시)."""
    rows = []
    for line in EVAL_LIST.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        i, name = line.split("\t")
        rows.append({"file": name.strip(), "label_index": int(i)})

    cache.mkdir(parents=True, exist_ok=True)

    def fetch(row: dict) -> bool:
        dest = cache / row["file"]
        if dest.exists() and dest.stat().st_size > 0:
            return True
        for _ in range(3):
            try:
                tmp = dest.with_suffix(".part")
                urllib.request.urlretrieve(EVAL_BASE_URL + row["file"], tmp)
                tmp.replace(dest)
                return True
            except OSError:
                continue
        return False

    with ThreadPoolExecutor(16) as ex:
        ok = list(ex.map(fetch, rows))
    missing = [r["file"] for r, good in zip(rows, ok) if not good]
    if missing:
        print(f"평가 사진 {len(missing)}장 받기 실패 (제외): {missing[:3]}")
    return [r for r, good in zip(rows, ok) if good]


def list_models(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and all((p / f).is_file() for f in REQUIRED))


def make_zip(model_dir: Path) -> bytes:
    buf = io.BytesIO()
    # model.tflite는 이미 압축이 안 되는 데이터라 STORED로 둔다 (폰에서 해제가 빠르다).
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        for f in REQUIRED:
            z.write(model_dir / f, arcname=f)
    return buf.getvalue()


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))  # 실제로 보내지 않음. 기본 경로의 인터페이스 주소만 얻는다.
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def make_handler(root: Path, eval_rows: list[dict] | None = None, eval_cache: Path = EVAL_CACHE):
    eval_files = {r["file"] for r in eval_rows or []}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            if url.path == "/models":
                body = json.dumps(list_models(root)).encode()
                return self._send(200, body, "application/json")
            if url.path == "/download-model":
                name = parse_qs(url.query).get("model_name", [""])[0]
                if not NAME_RE.match(name) or name not in list_models(root):
                    return self._send(404, f"unknown model: {name}".encode(), "text/plain; charset=utf-8")
                body = make_zip(root / name)
                return self._send(200, body, "application/zip",
                                  {"Content-Disposition": f'attachment; filename="{name}.zip"'})
            if url.path == "/eval/list" and eval_rows is not None:
                return self._send(200, json.dumps(eval_rows).encode(), "application/json")
            if url.path == "/eval/image" and eval_rows is not None:
                name = parse_qs(url.query).get("name", [""])[0]
                if name not in eval_files:  # 목록에 있는 이름만 — 경로 조작 차단
                    return self._send(404, b"unknown image", "text/plain")
                return self._send(200, (eval_cache / name).read_bytes(), "image/jpeg")
            return self._send(404, b"not found", "text/plain")

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--models", type=Path, default=TOOLS / "models")
    ap.add_argument("--eval", action="store_true", help="디코더 정확도 평가용 ImageNet 표본 1000장도 내려준다")
    a = ap.parse_args()

    models = list_models(a.models)
    if not models:
        raise SystemExit(f"{a.models} 에 모델이 없습니다. 먼저 python3 tools/fetch_model.py")

    eval_rows = None
    if a.eval:
        print("평가 사진 준비 중 (처음엔 1~2분)...")
        eval_rows = prepare_eval_set(EVAL_CACHE)
        print(f"평가 사진: {len(eval_rows)}장")

    print(f"모델: {', '.join(models)}")
    print(f"앱 빌드: flutter run --profile --dart-define=MODEL_SERVER=http://{lan_ip()}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), make_handler(a.models, eval_rows)).serve_forever()


if __name__ == "__main__":
    main()
