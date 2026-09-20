import 'dart:math' as math;

import '../api/api_service.dart';
import 'image_classifier.dart';

/// 라벨 있는 사진으로 두 디코더([ImageDecoder])의 정확도를 **기기에서** 비교한다.
///
/// 2026-09-15 iPhone 17 Pro Max 결과: dartImage 86.0% / platform 86.4% (p = 0.69).
/// dartImage(면적 평균) 86.0%는 PC의 PIL BOX 86.0%와 같아 평가 배관 자체도 검증됐다.
///
/// 왜 기기에서 재는가: 플랫폼 디코더의 축소 필터는 OS·엔진 구현이라 PC에서
/// 흉내 낼 수 없다. 사진 3장의 확률을 PC의 nearest/bilinear/box/lanczos/mipmap
/// 결과와 대 봤지만 어느 것과도 맞지 않았다. 그래서 같은 사진을 같은 기기에서
/// 두 경로로 분류해 짝지어 비교한다.
///
/// 사진은 모델 서버(`tools/model_server.py --eval`)가 ImageNet 클래스당 1장을
/// 한 장씩 내려준다. 전부 메모리에 올리지 않고 받자마자 분류하고 버린다.
class DecoderEvalResult {
  DecoderEvalResult(this.decoders);

  final List<ImageDecoder> decoders;

  /// 사진별 결과. key = 디코더, value = [top1 정답, top5 정답] (null = 디코드 실패)
  final List<Map<ImageDecoder, List<bool>?>> rows = [];

  /// 디코더별 한 장 처리 시간(마이크로초, 파일 전송 제외).
  final Map<ImageDecoder, List<int>> totalUs = {};

  /// 두 디코더의 상위 1개가 같았던 사진 수 (둘 다 성공한 사진 중).
  int sameTop1 = 0;

  int get attempted => rows.length;

  List<Map<ImageDecoder, List<bool>?>> get paired =>
      rows.where((r) => decoders.every((d) => r[d] != null)).toList();

  int failures(ImageDecoder d) => rows.where((r) => r[d] == null).length;

  String toMarkdown() {
    final p = paired;
    final n = p.length;
    if (n == 0) return '평가 실패 — 두 디코더가 모두 성공한 사진이 없습니다.';

    String pct(int k) => '${(k / n * 100).toStringAsFixed(1)}%';
    int count(ImageDecoder d, int k) => p.where((r) => r[d]![k]).length;
    String medMs(ImageDecoder d) {
      final v = [...?totalUs[d]]..sort();
      return v.isEmpty ? '—' : (v[v.length ~/ 2] / 1000).toStringAsFixed(1);
    }

    final lines = <String>[
      '사진 $n장 (시도 $attempted장, 실패: '
          '${decoders.map((d) => '${d.name} ${failures(d)}').join(', ')})',
      '',
      '| 디코더 | top-1 | top-5 | 한 장 전체 중앙값 |',
      '|---|---:|---:|---:|',
      for (final d in decoders)
        '| ${d.name} | ${pct(count(d, 0))} (${count(d, 0)}) | ${pct(count(d, 1))} | ${medMs(d)} ms |',
    ];

    if (decoders.length == 2) {
      final a = decoders[0], b = decoders[1];
      final onlyA = p.where((r) => r[a]![0] && !r[b]![0]).length;
      final onlyB = p.where((r) => !r[a]![0] && r[b]![0]).length;
      lines
        ..add('')
        ..add('top-1 짝지은 비교: ${a.name}만 맞힘 $onlyA장, ${b.name}만 맞힘 $onlyB장, '
            '정확 McNemar p = ${_mcnemarExact(onlyA, onlyB).toStringAsExponential(1)}')
        ..add('상위 1개 일치: $sameTop1/$n (${pct(sameTop1)})');
    }
    lines
      ..add('')
      ..add('주의: ImageNet 표본은 긴 변 500px 안팎이라 축소 배율이 2배 남짓입니다. '
          '카메라 사진(1024px→224px, 4.6배)에서는 필터 차이가 더 커질 수 있습니다.');
    return lines.join('\n');
  }
}

/// 양측 정확 McNemar 검정. b + c가 1000을 넘어도 넘치지 않게 로그 공간에서 계산한다.
double _mcnemarExact(int b, int c) {
  final n = b + c;
  if (n == 0) return 1.0;
  final k = math.min(b, c);
  // log C(n, i)를 i = 0부터 누적
  var logC = 0.0;
  final logTerms = <double>[];
  for (var i = 0; i <= k; i++) {
    if (i > 0) logC += math.log(n - i + 1) - math.log(i);
    logTerms.add(logC - n * math.ln2);
  }
  final maxLog = logTerms.reduce((x, y) => x > y ? x : y);
  final sum = logTerms.fold<double>(0, (s, t) => s + math.exp(t - maxLog));
  return math.min(1.0, 2 * math.exp(maxLog) * sum);
}

class DecoderAccuracyEval {
  DecoderAccuracyEval(this.classifier);

  final ImageClassifier classifier;
  bool _cancelled = false;

  void cancel() => _cancelled = true;

  /// [limit]장까지 평가한다. [onProgress]는 25장마다와 마지막에 불린다.
  Future<DecoderEvalResult> run({
    int? limit,
    void Function(int done, int total)? onProgress,
  }) async {
    final items = await ApiService.getEvalList();
    if (items.isEmpty) {
      throw StateError('평가 사진 목록이 비어 있습니다. 맥에서 '
          '`python3 tools/model_server.py --eval` 로 서버를 띄웠는지 확인하세요.');
    }
    final total = limit == null ? items.length : math.min(limit, items.length);

    // 1001-클래스 모델은 0번이 background라 정답 인덱스를 1 민다.
    final offset = classifier.labels.length - 1000;
    if (offset != 0 && offset != 1) {
      throw StateError('ImageNet 1000/1001 클래스 모델에서만 평가할 수 있습니다 '
          '(라벨 ${classifier.labels.length}개).');
    }

    const decoders = ImageDecoder.values;
    final result = DecoderEvalResult(decoders);
    final previous = classifier.decoder;

    try {
      for (var i = 0; i < total && !_cancelled; i++) {
        final item = items[i];
        final bytes = await ApiService.fetchEvalImage(item.file);
        final gt = item.labelIndex + offset;
        final row = <ImageDecoder, List<bool>?>{};
        final top1 = <ImageDecoder, int>{};

        // 순서 효과를 줄이려고 사진마다 디코더 순서를 번갈아 바꾼다.
        final order = i.isEven ? decoders : decoders.reversed.toList();
        for (final d in order) {
          if (bytes == null) {
            row[d] = null;
            continue;
          }
          classifier.decoder = d;
          List<Prediction>? top;
          try {
            top = await classifier.predictBytes(bytes, topK: 5);
          } catch (_) {
            top = null; // 한 장의 디코드 실패로 평가 전체를 멈추지 않는다
          }
          if (top == null || top.isEmpty) {
            row[d] = null;
            continue;
          }
          row[d] = [top.first.index == gt, top.any((p) => p.index == gt)];
          top1[d] = top.first.index;
          final t = classifier.lastTimings;
          if (t != null) (result.totalUs[d] ??= []).add(t.totalUs);
        }

        if (top1.length == decoders.length && top1.values.toSet().length == 1) {
          result.sameTop1++;
        }
        result.rows.add(row);
        if (onProgress != null && ((i + 1) % 25 == 0 || i + 1 == total)) {
          onProgress(i + 1, total);
        }
      }
    } finally {
      classifier.decoder = previous;
    }
    return result;
  }
}
