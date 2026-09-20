# 2026-09-15 · iPhone 17 Pro Max 실측

- 빌드: `flutter run --profile` (속도계 결과 표에 `profile` 표시 확인)
- 모델: `mobilenet_v2_1.0_224.tflite` float32, SHA-256 `ff5cb7f9…` (`tools/fetch_model.py`)
- 정규화: `minusOneToOne` · 전처리: 피커 긴 변 1024px 축소 → `image` 패키지 디코드 → 224×224 면적 평균 (플랫폼 디코더 경로 추가 전)
- 방법: 속도계 버튼 1회 = 워밍업 3회 + 20회 중앙값. 사진마다 1회씩
- 사진 원본은 출처가 분명하지 않은 웹 이미지라 레포에 넣지 않음. 캡처는 결과 창만 잘라 저장

| 캡처 | 사진 |
|---|---|
| [`cup-650x650.png`](cup-650x650.png) | 흰 커피잔, 650×650 JPEG |
| [`cat-1000x667.png`](cat-1000x667.png) | 파란 배경 고양이, 1000×667 WebP |
| [`keyboard-1320x701.png`](keyboard-1320x701.png) | 흰 배경 키보드 제품 사진, 1320×701 JPEG (피커가 1024×544로 축소) |

## 결과 창 원문

| 항목 (ms) | 커피잔 | 고양이 | 키보드 |
|---|---:|---:|---:|
| 디코드 | 17.7 | 24.1 | 19.5 |
| 리사이즈 | 13.0 | 16.5 | 15.6 |
| 텐서 변환 | 1.1 | 1.1 | 1.1 |
| 추론 | 9.5 | 9.8 | 9.6 |
| 후처리 | 0.2 | 0.2 | 0.2 |
| 전처리 합계 | 31.8 | 41.7 | 36.2 |
| 전체 | 41.5 | 51.7 | 46.0 |

| 텐서 변환 비교 (ms) | 커피잔 | 고양이 | 키보드 |
|---|---:|---:|---:|
| 중첩 List | 3.1 | 3.5 | 3.6 |
| Float32List | 1.2 | 1.2 | 2.7 |
| 앱 표시 배수 | 2.6× | 3.0× | 1.3× |
| 이 비교 중 추론 | 10.0 | 10.0 | 10.0 |

| 정규화 비교 (상위 1개) | 커피잔 | 고양이 | 키보드 |
|---|---|---|---|
| zeroToOne | cup 69.9% | Persian cat 74.8% | modem 10.5% |
| minusOneToOne | cup 80.5% | Persian cat 57.4% | computer keyboard 48.9% |

분류 화면 상위 2·3위 (minusOneToOne): 커피잔 coffee mug 4.5%, espresso 3.6% ·
고양이 Egyptian cat 2.2%, lynx 1.6% · 키보드 space bar 17.2%, notebook 7.2%

## PC 재현 (같은 사진, 긴 변 1024px LANCZOS 축소 → 224 PIL BOX)

| | 커피잔 | 고양이 | 키보드 |
|---|---|---|---|
| zeroToOne | cup 68.7% | Persian cat 74.4% | modem 9.5% |
| minusOneToOne | cup 79.0% | Persian cat 61.6% | computer keyboard 54.6% |

상위 1개 6/6 기기와 일치, 확률 차이 0.4~5.7%p. nearest(축소 없음)로 돌리면 `[0,1]` 키보드가
computer keyboard 40.2%로 나와 기기와 불일치 → 측정 빌드가 면적 평균 전처리였다는 확인이기도 함.

---

# 2회차 (23:15) · 디코더 비교

같은 기기·profile 빌드, `ImageDecoder.platform` 경로 추가 후. 캡처:
[`decoders-cup.png`](decoders-cup.png) · [`decoders-cat.png`](decoders-cat.png) · [`decoders-keyboard.png`](decoders-keyboard.png)

| 항목 (ms) | 커피잔 650×650 | 고양이 1000×667 | 키보드 1024×544 |
|---|---:|---:|---:|
| 디코드 (dartImage) | 15.4 | 20.2 | 16.6 |
| 리사이즈 | 11.2 | 14.3 | 13.7 |
| 텐서 변환 | 1.0 | 0.9 | 0.9 |
| 추론 | 8.1 | 7.8 | 8.5 |
| 전체 | 35.8 | 43.4 | 39.8 |
| 중첩 List / Float32List | 2.6 / 1.1 | 3.0 / 1.0 | 3.5 / 2.4 |

| 디코더 비교 | 디코드(+축소) | 리사이즈/픽셀 읽기 | 텐서 | 추론 | 전체 | 상위 1개 |
|---|---:|---:|---:|---:|---:|---|
| 커피잔 dartImage | 17.4 | 12.6 | 1.1 | 9.3 | 40.7 | cup 80.5% |
| 커피잔 platform | 1.6 | 0.5 | 0.1 | 9.0 | 11.3 | cup 72.5% |
| 고양이 dartImage | 23.3 | 16.0 | 1.1 | 9.0 | 49.6 | Persian cat 57.4% |
| 고양이 platform | 1.8 | 0.8 | 0.1 | 8.9 | 11.8 | Persian cat 68.7% |
| 키보드 dartImage | 18.5 | 14.3 | 2.4 | 9.4 | 44.7 | computer keyboard 48.9% |
| 키보드 platform | 2.2 | 0.5 | 0.1 | 9.5 | 12.5 | computer keyboard 39.1% |

앱 표시 배수: 3.6× / 4.2× / 3.6×. 정규화 비교 결과는 1회차와 동일(같은 dartImage 경로).

## platform 축소 필터 추정 시도 (PC, 실패)

같은 사진·[-1,1]에서 상위 1개 확률(%). 기기 platform: 72.5 / 68.7 / 39.1

| PC 필터 | 커피잔 | 고양이 | 키보드 |
|---|---:|---:|---:|
| nearest | 83.7 | 47.4 | 69.6 |
| bilinear (안티앨리어싱 없음) | 83.1 | 46.3 | 67.0 |
| mipmap(box ½) + bilinear | 77.4 | 61.7 | 60.0 |
| PIL bilinear (AA) | 72.9 | 50.2 | 27.0 |
| PIL box | 79.0 | 61.6 | 54.6 |
| PIL bicubic (AA) | 78.9 | 49.8 | 43.0 |
| PIL lanczos | 81.6 | 54.8 | 49.8 |

세 장을 동시에 맞추는 필터가 없음 → 1000장 정확도는 기기에서 직접 잰다 (「디코더 정확도 평가」 버튼).

---

# 3회차 (23:35) · 디코더 정확도, ImageNet 표본 1000장

앱 「디코더 정확도 평가」 버튼, `model_server.py --eval`. 캡처: [`decoder-accuracy-1000.png`](decoder-accuracy-1000.png)

| 디코더 | top-1 | top-5 | 한 장 전체 중앙값 |
|---|---:|---:|---:|
| dartImage | 86.0% (860) | 97.7% | 29.5 ms |
| platform | 86.4% (864) | 98.1% | 11.2 ms |

- 실패 0 / 0. dartImage만 맞힘 27, platform만 맞힘 31, 정확 McNemar p = 0.69. 상위 1개 일치 920/1000
- 짝지은 차이 +0.4%p, 95% CI −1.1 ~ +1.9%p (정규근사, 불일치 58장 기준)
- dartImage 86.0%는 같은 1000장을 PC에서 PIL BOX로 줄여 잰 86.0%와 일치 → 앱 평가 코드 검증
- 결론: 기본 디코더를 `platform`으로 변경
