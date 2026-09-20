#!/usr/bin/env python3
"""입력 정규화 [0,1] vs [-1,1]을 라벨 있는 사진 1000장으로 비교한다.

왜 필요한가: 정규화가 틀려도 에러가 나지 않는다. 쉬운 사진은 틀린 정규화로도
대개 맞히고, 확률이 오히려 더 높게 나오기도 한다. 그래서 사진 한두 장이 아니라
라벨 있는 여러 장의 정확도로 판단한다.

데이터: ImageNet 클래스당 1장 (github.com/EliSchwartz/imagenet-sample-images).
        파일 목록과 정답 인덱스는 tools/data/imagenet_sample_files.txt.
        주의 — 이 사진들이 학습 세트에 들어 있을 수 있어 **절대 정확도는
        낙관적**이다. 두 정규화의 상대 비교에만 쓴다.

전처리는 앱(lib/services/image_classifier.dart)과 같다:
        원본 전체를 224×224로 리사이즈(크롭 없음) → RGB → 정규화 → NHWC float32.
        리사이즈 보간은 앱이 쓰는 image 패키지 copyResize 기본값(nearest)과
        bilinear 둘 다 잰다.

    pip install ai-edge-litert pillow numpy
    python3 tools/fetch_model.py
    python3 tools/check_normalization.py      # 사진 받는 데 1~2분, 추론 30초 안팎
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # TensorFlow가 이미 있으면 그걸 쓴다
    from tensorflow.lite import Interpreter  # type: ignore

ROOT = Path(__file__).resolve().parent
BASE_URL = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/"
NORMS = {
    "zeroToOne": lambda x: x / 255.0,
    "minusOneToOne": lambda x: (x - 127.5) / 127.5,
}
RESAMPLE = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR}


def load_file_list() -> list[tuple[int, str]]:
    rows = []
    for line in (ROOT / "data" / "imagenet_sample_files.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        i, name = line.split("\t")
        rows.append((int(i), name))
    return rows


def download(files: list[tuple[int, str]], cache: Path) -> list[str]:
    cache.mkdir(parents=True, exist_ok=True)

    def one(name: str) -> str | None:
        dest = cache / name
        if dest.exists() and dest.stat().st_size > 0:
            return None
        for _ in range(3):
            try:
                tmp = dest.with_suffix(".part")
                urllib.request.urlretrieve(BASE_URL + name, tmp)
                tmp.replace(dest)
                return None
            except OSError:
                continue
        return name

    with ThreadPoolExecutor(16) as ex:
        return [m for m in ex.map(one, [n for _, n in files]) if m]


def mcnemar_exact(b: int, c: int) -> float:
    """짝지은 두 분류기의 불일치 (b, c)에 대한 양측 정확 McNemar 검정 p값."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=ROOT / "models" / "mobilenetv2" / "model.tflite")
    ap.add_argument("--labels", type=Path, default=None, help="기본: 모델 폴더의 labels.txt")
    ap.add_argument("--cache", type=Path, default=ROOT / ".cache" / "imagenet-sample")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "normalization.json")
    a = ap.parse_args()

    labels_path = a.labels or a.model.with_name("labels.txt")
    labels = [l.strip() for l in labels_path.read_text().splitlines() if l.strip()]
    it = Interpreter(model_path=str(a.model))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    _, h, w, _ = inp["shape"]
    n_out = int(out["shape"][-1])
    offset = n_out - 1000  # 1001-클래스 모델은 0번이 background
    if offset not in (0, 1) or n_out != len(labels):
        print(f"출력 {n_out}개 / 라벨 {len(labels)}개 — 이 스크립트는 ImageNet 1000/1001 클래스 모델용입니다", file=sys.stderr)
        return 1

    files = load_file_list()
    missing = download(files, a.cache)
    if missing:
        print(f"다운로드 실패 {len(missing)}장 (제외하고 진행): {missing[:5]}", file=sys.stderr)
    files = [(i, n) for i, n in files if n not in missing]

    # hits[resample][norm] = 사진별 top-1 정답 여부, conf = 상위 1개 확률
    hits = {r: {k: [] for k in NORMS} for r in RESAMPLE}
    top5 = {r: {k: 0 for k in NORMS} for r in RESAMPLE}
    conf = {r: {k: [] for k in NORMS} for r in RESAMPLE}

    for cls, name in files:
        gt = cls + offset
        rgb = Image.open(a.cache / name).convert("RGB")
        for rname, resample in RESAMPLE.items():
            px = np.asarray(rgb.resize((w, h), resample), dtype=np.float32)
            for norm, fn in NORMS.items():
                it.set_tensor(inp["index"], fn(px)[None].astype(np.float32))
                it.invoke()
                s = it.get_tensor(out["index"])[0]
                order = np.argsort(-s)
                hits[rname][norm].append(int(order[0] == gt))
                top5[rname][norm] += int(gt in order[:5])
                conf[rname][norm].append(float(s[order[0]]))

    n = len(files)
    report = {"model": str(a.model.name), "images": n, "note": "절대 정확도는 낙관적(학습 세트 포함 가능). 상대 비교만 유효.", "resample": {}}
    print(f"사진 {n}장 · 모델 {a.model.name}\n")
    print("| 리사이즈 | 정규화 | top-1 | top-5 | 상위 1개 확률 중앙값 |")
    print("|---|---|---:|---:|---:|")
    for rname in RESAMPLE:
        z, m = np.array(hits[rname]["zeroToOne"]), np.array(hits[rname]["minusOneToOne"])
        b, c = int(((m == 1) & (z == 0)).sum()), int(((m == 0) & (z == 1)).sum())
        p = mcnemar_exact(b, c)
        # 속도계 버튼처럼 "한 장에서 확률이 높은 쪽"을 고르면 [-1,1]을 고르는 비율
        pick = float(np.mean(np.array(conf[rname]["minusOneToOne"]) > np.array(conf[rname]["zeroToOne"])))
        entry = {"mcnemar": {"only_minusOneToOne_correct": b, "only_zeroToOne_correct": c, "p": p},
                 "single_image_confidence_picks_minusOneToOne": pick}
        for norm in NORMS:
            t1 = float(np.mean(hits[rname][norm]))
            t5 = top5[rname][norm] / n
            med = float(np.median(conf[rname][norm]))
            entry[norm] = {"top1": t1, "top5": t5, "median_top1_confidence": med}
            print(f"| {rname} | {norm} | {t1 * 100:.1f}% | {t5 * 100:.1f}% | {med * 100:.1f}% |")
        report["resample"][rname] = entry

    print()
    for rname, e in report["resample"].items():
        mc = e["mcnemar"]
        print(f"{rname}: [-1,1]만 맞힘 {mc['only_minusOneToOne_correct']}장, [0,1]만 맞힘 "
              f"{mc['only_zeroToOne_correct']}장, 정확 McNemar p = {mc['p']:.1e} · "
              f"한 장 확률로 고르면 [-1,1] 선택 {e['single_image_confidence_picks_minusOneToOne'] * 100:.0f}%")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"\n저장: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
