import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// 모델이 기대하는 입력 정규화 범위.
///
/// MobileNetV2 배포본은 두 종류가 돌아다닌다. Keras `preprocess_input`·TF-Slim
/// 계열은 [-1, 1]을, TF Hub의 일부 TF2 변환본은 [0, 1]을 기대한다. **틀려도 에러가
/// 나지 않고 정확도만 조용히 떨어진다.**
///
/// 이 앱이 쓰는 모델(`tools/fetch_model.py`가 받는 1001-클래스 MobileNetV2)은
/// [-1, 1]로 확정했다. 근거는 세 가지다 (README 「정규화 범위 확인」).
///  1. 모델 메타데이터: NormalizationOptions mean 127.5 / std 127.5
///  2. ImageNet 클래스당 1장(1000장)을 앱과 같은 전처리(nearest 리사이즈)로 분류:
///     top-1 [-1,1] 83.9% vs [0,1] 75.0% (McNemar p ≈ 5e-16)
///  3. 첫 커밋의 labels.txt가 이 모델의 내장 라벨과 1001줄 모두 일치
///
/// 모델을 바꾸면 이 판단도 다시 해야 한다. 한 장의 확률만으로는 판단할 수 없다
/// ([ImageClassifier.compareNormalizations] 참고).
enum InputNormalization {
  /// pixel / 255 → [0, 1]
  zeroToOne,

  /// (pixel - 127.5) / 127.5 → [-1, 1]
  minusOneToOne,
}

/// 사진을 모델 입력 크기의 픽셀로 만드는 방법.
///
/// iPhone 17 Pro Max 실측(README 「측정」)에서 추론은 약 10 ms인데 전처리(디코드·
/// 리사이즈·텐서 변환)가 32~42 ms로 전체의 77~81%였다. 피커에서 1024px로 줄인 뒤인데도
/// 순수 Dart 디코드와 면적 평균 축소가 대부분이라 플랫폼 디코더와 비교한다.
enum ImageDecoder {
  /// `image` 패키지 (순수 Dart). 받은 파일을 전부 디코드한 뒤 면적 평균으로 줄인다.
  dartImage,

  /// Flutter 엔진의 네이티브 코덱 (iOS는 ImageIO 계열). 디코드 단계에서 바로
  /// 모델 크기로 줄여 원본 해상도 버퍼를 만들지 않는다. **기본값.**
  ///
  /// iPhone 17 Pro Max, ImageNet 표본 1000장 (README 「디코더 비교」):
  /// top-1 86.4% vs dartImage 86.0% (짝지은 차이 +0.4%p, 95% CI −1.1~+1.9%p,
  /// McNemar p = 0.69), 한 장 중앙값 11.2 ms vs 29.5 ms.
  platform,
}

/// 한 장을 분류하는 데 걸린 시간 (마이크로초 단위로 재고 밀리초로 보고).
///
/// 온디바이스에서는 추론 시간만 재면 절반만 보는 것이다. 디코드·리사이즈·
/// 텐서 채우기가 추론보다 오래 걸리는 경우가 흔하다.
class ClassificationTimings {
  const ClassificationTimings({
    required this.decodeUs,
    required this.resizeUs,
    required this.tensorFillUs,
    required this.inferenceUs,
    required this.postprocessUs,
  });

  final int decodeUs;
  final int resizeUs;
  final int tensorFillUs;
  final int inferenceUs;
  final int postprocessUs;

  int get preprocessUs => decodeUs + resizeUs + tensorFillUs;
  int get totalUs => preprocessUs + inferenceUs + postprocessUs;

  static String _ms(int us) => (us / 1000).toStringAsFixed(1);

  /// 로그로 바로 붙일 수 있는 한 줄 요약.
  @override
  String toString() =>
      'decode ${_ms(decodeUs)}ms | resize ${_ms(resizeUs)}ms | '
      'tensor ${_ms(tensorFillUs)}ms | infer ${_ms(inferenceUs)}ms | '
      'post ${_ms(postprocessUs)}ms | total ${_ms(totalUs)}ms';

  /// README 「측정」 표에 그대로 붙일 수 있는 마크다운 행.
  String toMarkdownRows() => '| 디코드 | ${_ms(decodeUs)} ms |\n'
      '| 리사이즈 | ${_ms(resizeUs)} ms |\n'
      '| 텐서 변환 | ${_ms(tensorFillUs)} ms |\n'
      '| **추론** | **${_ms(inferenceUs)} ms** |\n'
      '| 후처리 | ${_ms(postprocessUs)} ms |\n'
      '| 전처리 합계 | ${_ms(preprocessUs)} ms |\n'
      '| 전체 | ${_ms(totalUs)} ms |';
}

class Prediction {
  const Prediction({required this.index, required this.label, required this.confidence});

  /// 모델 출력 인덱스. ImageNet 라벨은 이름이 겹치는 클래스가 있어
  /// (crane 새/기계, maillot 두 개) 정답 비교는 이름이 아니라 인덱스로 한다.
  final int index;
  final String label;
  final double confidence;

  Map<String, dynamic> toMap() => {'label': label, 'confidence': confidence};
}

class ImageClassifier {
  Interpreter? _interpreter;
  List<String>? _labels;

  /// 입력 정규화 범위. 모델 카드와 맞지 않으면 정확도가 조용히 떨어진다.
  /// 기본 모델 기준 [-1, 1]이 맞다 ([InputNormalization] 문서의 근거 참고).
  InputNormalization normalization = InputNormalization.minusOneToOne;

  /// true면 [Float32List]에 연속으로 쓰고, false면 예전 중첩 List 경로를 쓴다.
  /// 벤치마크에서 전처리 최적화 전/후를 같은 기기에서 비교하기 위해 남겨 둔다.
  bool useFastPreprocess = true;

  /// 픽셀을 만드는 방법. 기기 1000장 평가에서 정확도 차이 없이 2.6~4.2배 빨라
  /// platform을 기본값으로 둔다. dartImage는 비교 측정용으로 남긴다.
  ImageDecoder decoder = ImageDecoder.platform;

  /// 마지막 [classifyImage] 호출의 구간별 소요 시간.
  ClassificationTimings? lastTimings;

  /// 마지막으로 분류한 사진의 원본 크기. 디코드 시간은 해상도에 좌우되므로
  /// 측정 표에 같이 적는다.
  int? lastSourceWidth;
  int? lastSourceHeight;

  /// 마지막 로드 실패 이유. UI에 그대로 띄워 원인을 바로 보이게 한다.
  String? lastError;

  bool isModelLoaded() => _interpreter != null;

  List<String> get labels => _labels ?? const [];

  Future<void> initializeClassifier(String modelName) async {
    try {
      final appDir = await getApplicationDocumentsDirectory();
      final modelPath = '${appDir.path}/models/$modelName/model.tflite';
      final labelPath = '${appDir.path}/models/$modelName/labels.txt';

      lastError = null;
      final interpreter = await Interpreter.fromFile(File(modelPath));
      interpreter.allocateTensors();

      final labelData = await File(labelPath).readAsString();
      final labels =
          labelData.split('\n').map((l) => l.trim()).where((l) => l.isNotEmpty).toList();

      final outputLength = interpreter.getOutputTensor(0).shape.last;
      if (outputLength != labels.length) {
        // 라벨 수와 출력 차원이 다르면 인덱스가 밀려 엉뚱한 이름이 붙는다.
        // 조용히 넘기면 "그럭저럭 맞는 것처럼" 보이기 때문에 여기서 끊는다.
        interpreter.close();
        throw StateError(
          '라벨 ${labels.length}개 / 모델 출력 $outputLength개 — 개수가 맞지 않습니다',
        );
      }

      _interpreter = interpreter;
      _labels = labels;
    } catch (e) {
      // 호출부(home_screen)는 isModelLoaded()로 성공 여부를 판단한다. 여기서
      // 던지면 UI 흐름이 끊기므로, 이유만 남기고 실패 상태로 둔다.
      _interpreter = null;
      _labels = null;
      lastError = e.toString();
    }
  }

  /// 상위 [topK]개 분류 결과. 구간별 소요 시간은 [lastTimings]에 남는다.
  Future<List<Map<String, dynamic>>?> classifyImage(File imageFile, {int topK = 3}) async {
    final predictions = await predict(imageFile, topK: topK);
    return predictions?.map((p) => p.toMap()).toList();
  }

  Future<List<Prediction>?> predict(File imageFile, {int topK = 3}) async {
    return predictBytes(await imageFile.readAsBytes(), topK: topK);
  }

  /// 인코딩된 이미지 바이트(JPEG/PNG/WebP 등)를 분류한다. 파일을 읽는 시간은
  /// 구간 계측에 넣지 않는다.
  Future<List<Prediction>?> predictBytes(Uint8List bytes, {int topK = 3}) async {
    final interpreter = _interpreter;
    final labels = _labels;
    if (interpreter == null || labels == null) return null;

    final inputTensor = interpreter.getInputTensor(0);
    final shape = inputTensor.shape; // [1, H, W, 3]
    final height = shape[1];
    final width = shape[2];

    final watch = Stopwatch()..start();

    if (decoder == ImageDecoder.platform && inputTensor.type == TensorType.float32) {
      return _predictWithPlatformDecoder(bytes, interpreter, labels, width, height, topK, watch);
    }

    final decoded = img.decodeImage(bytes);
    if (decoded == null) return null;
    lastSourceWidth = decoded.width;
    lastSourceHeight = decoded.height;
    final decodeUs = watch.elapsedMicroseconds;

    watch.reset();
    // 기본값 nearest는 쓰면 안 된다. 1024px → 224px만 해도 4~5칸마다 한 픽셀을 집어
    // 모아레가 생기고, 모델은 이를 쇠사슬·벌집·방충망 같은 질감으로 읽는다.
    // average는 대상 픽셀이 덮는 원본 영역을 평균 낸다 (PIL BOX와 같다).
    final resized = img.copyResize(
      decoded,
      width: width,
      height: height,
      interpolation: img.Interpolation.average,
    );
    final resizeUs = watch.elapsedMicroseconds;

    // ── 전처리: 픽셀 → 정규화된 float32 ───────────────────────────────────
    watch.reset();
    final int tensorFillUs;
    final bool floatInput = inputTensor.type == TensorType.float32;

    if (floatInput && useFastPreprocess) {
      final buffer = _fillFloat32(resized, width, height);
      tensorFillUs = watch.elapsedMicroseconds;

      watch.reset();
      // Float32List을 그대로 넘기면 안 된다. tflite_flutter의
      // ByteConversionUtils.convertObjectToBytes는 Uint8List/ByteBuffer만
      // 그대로 통과시키고, 그 밖의 List는 원소마다 4바이트 버퍼를 새로 만들어
      // growable List<int>에 addAll 한다 (15만 회). 같은 메모리를 가리키는
      // Uint8List 뷰로 넘기면 변환 없이 memcpy 한 번으로 끝난다.
      inputTensor.setTo(buffer.buffer.asUint8List());
      interpreter.invoke();
    } else {
      // 예전 경로: 중첩 List. 픽셀당 Dart 객체를 거치므로 224²×3 = 15만 번
      // 대입이 일어난다. 비교용으로만 남겨 둔다.
      final nested = _fillNestedList(resized, width, height);
      tensorFillUs = watch.elapsedMicroseconds;

      watch.reset();
      final output = List.filled(labels.length, 0.0).reshape([1, labels.length]);
      interpreter.run(nested, output);
      final scores = (output[0] as List).cast<double>();
      final inferenceUs = watch.elapsedMicroseconds;

      watch.reset();
      final top = _topK(scores, labels, topK);
      lastTimings = ClassificationTimings(
        decodeUs: decodeUs,
        resizeUs: resizeUs,
        tensorFillUs: tensorFillUs,
        inferenceUs: inferenceUs,
        postprocessUs: watch.elapsedMicroseconds,
      );
      return top;
    }

    final inferenceUs = watch.elapsedMicroseconds;

    watch.reset();
    final top = _readTopK(interpreter, labels, topK);
    final postprocessUs = watch.elapsedMicroseconds;

    lastTimings = ClassificationTimings(
      decodeUs: decodeUs,
      resizeUs: resizeUs,
      tensorFillUs: tensorFillUs,
      inferenceUs: inferenceUs,
      postprocessUs: postprocessUs,
    );
    return top;
  }

  /// [ImageDecoder.platform] 경로. 구간 이름은 [ClassificationTimings]를 그대로
  /// 쓰되 의미가 다르다: decode = 네이티브 디코드+축소, resize = 픽셀 읽기(RGBA).
  Future<List<Prediction>?> _predictWithPlatformDecoder(
    Uint8List bytes,
    Interpreter interpreter,
    List<String> labels,
    int width,
    int height,
    int topK,
    Stopwatch watch,
  ) async {
    final buffer = await ui.ImmutableBuffer.fromUint8List(bytes);
    final descriptor = await ui.ImageDescriptor.encoded(buffer);
    lastSourceWidth = descriptor.width;
    lastSourceHeight = descriptor.height;
    // 두 크기를 모두 주면 비율을 무시하고 정확히 이 크기로 줄인다.
    // dartImage 경로(크롭 없이 전체 리사이즈)와 같은 기하 변환이다.
    final codec = await descriptor.instantiateCodec(targetWidth: width, targetHeight: height);
    final frame = await codec.getNextFrame();
    final image = frame.image;
    final decodeUs = watch.elapsedMicroseconds;

    watch.reset();
    // 사진은 불투명하므로 premultiplied RGBA여도 RGB 값은 그대로다.
    final rgba = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    image.dispose();
    codec.dispose();
    descriptor.dispose();
    buffer.dispose();
    if (rgba == null) return null;
    final resizeUs = watch.elapsedMicroseconds;

    watch.reset();
    final input = _fillFloat32FromRgba(
      rgba.buffer.asUint8List(rgba.offsetInBytes, rgba.lengthInBytes),
      width,
      height,
    );
    final tensorFillUs = watch.elapsedMicroseconds;

    watch.reset();
    interpreter.getInputTensor(0).setTo(input.buffer.asUint8List());
    interpreter.invoke();
    final inferenceUs = watch.elapsedMicroseconds;

    watch.reset();
    final top = _readTopK(interpreter, labels, topK);
    lastTimings = ClassificationTimings(
      decodeUs: decodeUs,
      resizeUs: resizeUs,
      tensorFillUs: tensorFillUs,
      inferenceUs: inferenceUs,
      postprocessUs: watch.elapsedMicroseconds,
    );
    return top;
  }

  List<Prediction> _readTopK(Interpreter interpreter, List<String> labels, int topK) {
    final raw = interpreter.getOutputTensor(0).data;
    final scores = raw.buffer
        .asFloat32List(raw.offsetInBytes, labels.length)
        .map((v) => v.toDouble())
        .toList(growable: false);
    return _topK(scores, labels, topK);
  }

  Float32List _fillFloat32FromRgba(Uint8List rgba, int width, int height) {
    final buffer = Float32List(width * height * 3);
    final scale = normalization == InputNormalization.zeroToOne ? 1 / 255.0 : 1 / 127.5;
    final shift = normalization == InputNormalization.zeroToOne ? 0.0 : -1.0;

    final n = width * height * 4;
    var o = 0;
    for (var i = 0; i < n; i += 4) {
      buffer[o++] = rgba[i] * scale + shift;
      buffer[o++] = rgba[i + 1] * scale + shift;
      buffer[o++] = rgba[i + 2] * scale + shift;
    }
    return buffer;
  }

  Float32List _fillFloat32(img.Image image, int width, int height) {
    final buffer = Float32List(width * height * 3);
    final scale = normalization == InputNormalization.zeroToOne ? 1 / 255.0 : 1 / 127.5;
    final shift = normalization == InputNormalization.zeroToOne ? 0.0 : -1.0;

    var i = 0;
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final p = image.getPixel(x, y);
        buffer[i++] = p.r * scale + shift;
        buffer[i++] = p.g * scale + shift;
        buffer[i++] = p.b * scale + shift;
      }
    }
    return buffer;
  }

  List<List<List<List<double>>>> _fillNestedList(img.Image image, int width, int height) {
    final scale = normalization == InputNormalization.zeroToOne ? 1 / 255.0 : 1 / 127.5;
    final shift = normalization == InputNormalization.zeroToOne ? 0.0 : -1.0;

    return [
      List.generate(
        height,
        (y) => List.generate(width, (x) {
          final p = image.getPixel(x, y);
          return [p.r * scale + shift, p.g * scale + shift, p.b * scale + shift];
        }),
      ),
    ];
  }

  List<Prediction> _topK(List<double> scores, List<String> labels, int k) {
    final indices = List<int>.generate(scores.length, (i) => i)
      ..sort((a, b) => scores[b].compareTo(scores[a]));
    return indices
        .take(k)
        .map((i) => Prediction(index: i, label: labels[i], confidence: scores[i]))
        .toList();
  }

  /// 같은 이미지를 [iterations]번 분류해 구간별 **중앙값**을 돌려준다.
  ///
  /// 첫 호출은 지연 초기화·캐시 워밍 때문에 항상 느리므로 [warmup]회는 버린다.
  /// 평균이 아니라 중앙값을 쓰는 이유는 모바일에서 GC·스케줄링 때문에 한두 번
  /// 크게 튀는 값이 섞이기 때문이다.
  Future<ClassificationTimings?> benchmark(
    File imageFile, {
    int iterations = 20,
    int warmup = 3,
  }) async {
    if (!isModelLoaded()) return null;

    for (var i = 0; i < warmup; i++) {
      await predict(imageFile);
    }

    final samples = <ClassificationTimings>[];
    for (var i = 0; i < iterations; i++) {
      await predict(imageFile);
      final t = lastTimings;
      if (t != null) samples.add(t);
    }
    if (samples.isEmpty) return null;

    int median(int Function(ClassificationTimings) pick) {
      final values = samples.map(pick).toList()..sort();
      return values[values.length ~/ 2];
    }

    return ClassificationTimings(
      decodeUs: median((t) => t.decodeUs),
      resizeUs: median((t) => t.resizeUs),
      tensorFillUs: median((t) => t.tensorFillUs),
      inferenceUs: median((t) => t.inferenceUs),
      postprocessUs: median((t) => t.postprocessUs),
    );
  }

  /// 전처리 최적화 전/후를 같은 기기에서 비교한다. README의 before/after 숫자가
  /// 여기서 나온다.
  Future<String> benchmarkPreprocess(File imageFile, {int iterations = 20}) async {
    final previous = useFastPreprocess;
    final previousDecoder = decoder;
    decoder = ImageDecoder.dartImage; // 중첩 List 경로는 dartImage 디코더에만 있다

    useFastPreprocess = false;
    final slow = await benchmark(imageFile, iterations: iterations);
    useFastPreprocess = true;
    final fast = await benchmark(imageFile, iterations: iterations);

    useFastPreprocess = previous;
    decoder = previousDecoder;
    if (slow == null || fast == null) return '측정 실패 — 모델이 로드되지 않았습니다';

    final ratio = slow.tensorFillUs / (fast.tensorFillUs == 0 ? 1 : fast.tensorFillUs);
    return '| 텐서 변환 방식 | 소요 시간 |\n'
        '|---|---:|\n'
        '| 중첩 List (기존) | ${(slow.tensorFillUs / 1000).toStringAsFixed(1)} ms |\n'
        '| Float32List (현재) | ${(fast.tensorFillUs / 1000).toStringAsFixed(1)} ms |\n'
        '\n'
        '${ratio.toStringAsFixed(1)}배 단축. 추론 시간은 '
        '${(fast.inferenceUs / 1000).toStringAsFixed(1)} ms.';
  }

  /// 디코더 두 가지를 같은 사진으로 번갈아 측정해 구간별 중앙값과 상위 1개를
  /// 나란히 보여준다. 순서 효과(발열·캐시)를 줄이려고 dartImage → platform →
  /// dartImage → platform 순으로 두 번씩 재고 각 디코더의 두 중앙값 중 작은 쪽을 쓴다.
  Future<String> benchmarkDecoders(File imageFile, {int iterations = 20}) async {
    final previous = decoder;
    final results = <ImageDecoder, ClassificationTimings>{};
    final top1 = <ImageDecoder, Prediction?>{};

    for (var round = 0; round < 2; round++) {
      for (final d in ImageDecoder.values) {
        decoder = d;
        final t = await benchmark(imageFile, iterations: iterations);
        if (t == null) continue;
        final prev = results[d];
        if (prev == null || t.totalUs < prev.totalUs) results[d] = t;
        final top = await predict(imageFile, topK: 1);
        top1[d] = top?.isNotEmpty == true ? top!.first : null;
      }
    }
    decoder = previous;

    final a = results[ImageDecoder.dartImage];
    final b = results[ImageDecoder.platform];
    if (a == null || b == null) return '디코더 비교 실패 — 모델이 로드되지 않았거나 디코드에 실패했습니다';

    String ms(int us) => (us / 1000).toStringAsFixed(1);
    String label(ImageDecoder d) {
      final p = top1[d];
      return p == null ? '—' : '${p.label} ${(p.confidence * 100).toStringAsFixed(1)}%';
    }

    final speedup = a.totalUs / (b.totalUs == 0 ? 1 : b.totalUs);
    return '| 디코더 | 디코드(+축소) | 리사이즈 / 픽셀 읽기 | 텐서 | 추론 | 전체 | 상위 1개 |\n'
        '|---|---:|---:|---:|---:|---:|---|\n'
        '| dartImage | ${ms(a.decodeUs)} | ${ms(a.resizeUs)} | ${ms(a.tensorFillUs)} | '
        '${ms(a.inferenceUs)} | **${ms(a.totalUs)}** | ${label(ImageDecoder.dartImage)} |\n'
        '| platform | ${ms(b.decodeUs)} | ${ms(b.resizeUs)} | ${ms(b.tensorFillUs)} | '
        '${ms(b.inferenceUs)} | **${ms(b.totalUs)}** | ${label(ImageDecoder.platform)} |\n'
        '\n'
        '전체 ${speedup.toStringAsFixed(1)}배 (ms, 중앙값).';
  }

  /// 두 정규화 범위로 각각 분류해 상위 1개를 비교한다.
  ///
  /// **이 결과 한 장으로 정규화를 고르면 안 된다.** 틀린 정규화도 쉬운 사진은
  /// 대개 맞히고, 확률이 오히려 더 높게 나오기도 한다. 1000장에서 "확률이 높은
  /// 쪽"을 골랐을 때 맞는 정규화([-1,1])를 고른 비율은 65%뿐이었다. 결정은
  /// 모델 메타데이터와 라벨 있는 여러 장의 정확도로 한다
  /// (`tools/check_normalization.py`). 여기서는 기기에서도 두 경로가 모두
  /// 동작하는지 보는 확인용으로만 쓴다.
  Future<String> compareNormalizations(File imageFile) async {
    final previous = normalization;
    final lines = <String>['| 정규화 | 상위 1개 | 확률 |', '|---|---|---:|'];

    for (final mode in InputNormalization.values) {
      normalization = mode;
      final top = await predict(imageFile, topK: 1);
      final best = top?.isNotEmpty == true ? top!.first : null;
      lines.add(
        '| ${mode.name} | ${best?.label ?? '—'} | '
        '${best == null ? '—' : (best.confidence * 100).toStringAsFixed(1)}% |',
      );
    }

    normalization = previous;
    return lines.join('\n');
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
