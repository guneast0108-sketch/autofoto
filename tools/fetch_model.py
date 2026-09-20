#!/usr/bin/env python3
"""앱이 쓰는 MobileNetV2 TFLite를 받아 모델 서버 폴더 구조로 풀어 둔다.

원래 모델 서버(개발 당시 핫스팟의 다른 노트북)와 모델 파일은 남아 있지 않다.
그래서 출처가 분명한 파일로 고정한다:

    TensorFlow 공식 tflite-support 저장소의 테스트 데이터
    mobilenet_v2_1.0_224.tflite — float32, 입력 [1,224,224,3], 출력 1001 클래스
    메타데이터: NormalizationOptions mean 127.5 / std 127.5  →  입력 [-1, 1]

첫 커밋(b98d580)의 assets/labels.txt가 이 모델의 내장 라벨과 1001줄 모두
일치하므로, 원래 쓰던 것도 같은 계열 모델이었을 가능성이 높다.

표준 라이브러리만 쓴다.

    python3 tools/fetch_model.py            # → tools/models/mobilenetv2/{model.tflite,labels.txt}
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

MODEL_URL = (
    "https://raw.githubusercontent.com/tensorflow/tflite-support/master/"
    "tensorflow_lite_support/cc/test/testdata/task/vision/mobilenet_v2_1.0_224.tflite"
)
# 2026-09-15에 받은 파일. 저장소 쪽 파일이 바뀌면 여기서 멈춰서 알린다.
MODEL_SHA256 = "ff5cb7f9e62c92ebdad971f8a98aa6b3106d82a64587a7787c6a385c9e791339"

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 앱 기본 모델명(home_screen.dart의 'mobilenetv2')과 같아야 첫 실행 자동 다운로드가 된다.
    ap.add_argument("--name", default="mobilenetv2", help="앱 모델 목록에 보일 이름 (폴더명)")
    ap.add_argument("--out", type=Path, default=ROOT / "models")
    a = ap.parse_args()

    dest = a.out / a.name
    dest.mkdir(parents=True, exist_ok=True)
    model = dest / "model.tflite"

    if not model.exists() or sha256(model) != MODEL_SHA256:
        print(f"다운로드: {MODEL_URL}")
        tmp = model.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(model)

    digest = sha256(model)
    if digest != MODEL_SHA256:
        print(f"SHA-256 불일치: {digest}\n원본 파일이 바뀌었습니다. 정규화 판단을 다시 해야 합니다.", file=sys.stderr)
        return 1

    # 메타데이터가 붙은 TFLite는 파일 끝에 zip으로 부속 파일을 싣는다.
    with zipfile.ZipFile(model) as z:
        labels = z.read("labels.txt").decode()
    lines = [l for l in labels.splitlines() if l.strip()]
    (dest / "labels.txt").write_text("\n".join(lines) + "\n")

    print(f"완료: {dest}")
    print(f"  model.tflite  {model.stat().st_size / 1e6:.1f} MB  sha256 {digest[:12]}…")
    print(f"  labels.txt    {len(lines)}개 (0번 = {lines[0]!r})")
    print("  입력 정규화   [-1, 1]  (앱 기본값 InputNormalization.minusOneToOne)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
