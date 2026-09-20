import 'dart:io';
import 'package:flutter/foundation.dart' show kDebugMode, kProfileMode, kReleaseMode;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:path_provider/path_provider.dart';
import '../api/api_service.dart';
import '../services/accuracy_eval.dart';
import '../services/image_classifier.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImageClassifier _classifier = ImageClassifier();
  final ImagePicker _picker = ImagePicker();

  // 앱 상태를 관리하는 변수들
  String _statusMessage = '앱을 초기화 중입니다...';
  String? _currentModelName;
  bool _isModelReady = false;
  bool _isProcessing = false;

  // 이미지 및 결과 표시를 위한 변수들
  File? _selectedImage;
  List<Map<String, dynamic>>? _classificationResult;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  /// 앱 시작 시 저장된 모델을 불러오는 초기화 함수
  Future<void> _initialize() async {
    final prefs = await SharedPreferences.getInstance();
    // 저장된 모델 이름 불러오기, 없으면 'mobilenetv2'를 기본값으로 사용
    String modelToLoad = prefs.getString('selected_model') ?? 'mobilenetv2';
    await _loadOrDownloadModel(modelToLoad);
  }

  /// 특정 모델을 로드하거나, 없으면 다운로드 후 로드하는 함수
  Future<void> _loadOrDownloadModel(String modelName) async {
    setState(() {
      _currentModelName = modelName;
      _isModelReady = false;
      _isProcessing = true; // 로딩 시작
      _statusMessage = '$modelName 모델을 준비 중입니다...';
      _selectedImage = null;
      _classificationResult = null;
    });

    final appDir = await getApplicationDocumentsDirectory();
    final modelFile = File('${appDir.path}/models/$modelName/model.tflite');

    if (!await modelFile.exists()) {
      setState(() => _statusMessage = '$modelName 모델을 다운로드 중입니다...');
      bool success = await ApiService.downloadAndUnzipModel(modelName);
      if (!success) {
        setState(() {
          _statusMessage = '모델 다운로드 실패';
          _isProcessing = false;
        });
        return;
      }
    }

    await _classifier.initializeClassifier(modelName);
    
    if (_classifier.isModelLoaded()) {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('selected_model', modelName); // 성공 시 모델 이름 저장
      setState(() {
        _isModelReady = true;
        _statusMessage = '준비 완료! 사진을 선택하세요.';
      });
    } else {
      setState(() => _statusMessage = '모델 로딩 실패: ${_classifier.lastError ?? "원인 불명"}');
    }
    setState(() => _isProcessing = false); // 로딩 끝
  }

  /// 모델 선택 다이얼로그를 보여주는 함수
  Future<void> _showModelSelectionDialog() async {
    List<String> models = await ApiService.getAvailableModels();
    if (!mounted) return;

    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('모델 선택'),
          content: SizedBox(
            width: double.maxFinite,
            child: ListView.builder(
              shrinkWrap: true,
              itemCount: models.length,
              itemBuilder: (context, index) {
                final modelName = models[index];
                return ListTile(
                  title: Text(modelName),
                  trailing: _currentModelName == modelName ? const Icon(Icons.check, color: Colors.blue) : null,
                  onTap: () {
                    Navigator.of(context).pop();
                    if (_currentModelName != modelName) {
                      _loadOrDownloadModel(modelName); // 새 모델 선택
                    }
                  },
                );
              },
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('닫기'),
            ),
          ],
        );
      },
    );
  }

  /// 갤러리에서 사진을 선택하고 분류하는 함수
  Future<void> _pickAndClassifyImage() async {
    var status = await Permission.storage.request();
    if (status.isDenied || status.isPermanentlyDenied) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('저장소 접근 권한이 필요합니다.')),
      );
      return;
    }

    // 원본(아이폰 12MP)을 그대로 받으면 Dart 디코더가 느리고, 224px까지 한 번에
    // 줄이면 정확도도 떨어진다. 피커가 네이티브에서 고품질로 먼저 줄이게 한다.
    final XFile? image = await _picker.pickImage(
      source: ImageSource.gallery,
      maxWidth: 1024,
      maxHeight: 1024,
    );
    if (image == null) return;

    setState(() {
      _selectedImage = File(image.path);
      _isProcessing = true;
      _statusMessage = '이미지를 분류 중입니다...';
      _classificationResult = null;
    });

    final results = await _classifier.classifyImage(_selectedImage!);
    
    // 온디바이스에서는 구간별 소요 시간이 정확도만큼 중요하다. 별도 도구 없이
    // 화면에서 바로 읽을 수 있게 상태 줄에 붙인다.
    final timings = _classifier.lastTimings;
    setState(() {
      _classificationResult = results;
      _isProcessing = false;
      _statusMessage = results != null
          ? '분류 완료!${timings != null ? "\n$timings" : ""}'
          : '분류 실패';
    });
  }

  /// 모델 서버의 ImageNet 표본(클래스당 1장)으로 두 디코더의 정확도를 기기에서 비교한다.
  ///
  /// 속도는 [_runBenchmark]로 이미 쟀다 (platform 3.6~4.2배). 기본 디코더를 바꾸려면
  /// 정확도가 떨어지지 않는다는 근거가 필요한데, 플랫폼 축소 필터는 PC에서 재현할 수
  /// 없어서 기기에서 직접 잰다. 1000장 × 2경로라 1~2분 걸린다.
  Future<void> _runDecoderAccuracyEval() async {
    if (!_classifier.isModelLoaded()) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('모델을 먼저 로드하세요.')),
      );
      return;
    }

    setState(() {
      _isProcessing = true;
      _statusMessage = '디코더 정확도 평가 준비 중...';
    });

    String report;
    try {
      final result = await DecoderAccuracyEval(_classifier).run(
        onProgress: (done, total) {
          if (mounted) setState(() => _statusMessage = '디코더 정확도 평가 $done / $total');
        },
      );
      final buildMode = kReleaseMode
          ? 'release'
          : kProfileMode
              ? 'profile'
              : 'debug';
      report = [
        '| 빌드 | $buildMode |',
        '| 정규화 | ${_classifier.normalization.name} |',
        '',
        result.toMarkdown(),
      ].join('\n');
    } catch (e) {
      report = '평가 실패: $e';
    }

    if (!mounted) return;
    setState(() {
      _isProcessing = false;
      _statusMessage = '평가 완료';
    });

    await showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('디코더 정확도 (ImageNet 표본)'),
        content: SingleChildScrollView(
          child: SelectableText(report, style: const TextStyle(fontSize: 12)),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('닫기'),
          ),
        ],
      ),
    );
  }

  /// 선택한 사진으로 성능을 측정해 결과를 보여준다.
  ///
  /// 온디바이스 비전에서 제일 먼저 물어보는 숫자(추론 시간)를 별도 프로파일러
  /// 없이 실기기에서 바로 얻기 위한 화면이다. 결과 텍스트는 그대로 README의
  /// 「측정」 표에 붙일 수 있다.
  Future<void> _runBenchmark() async {
    final image = _selectedImage;
    if (image == null || !_classifier.isModelLoaded()) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('모델을 로드하고 사진을 먼저 선택하세요.')),
      );
      return;
    }

    setState(() {
      _isProcessing = true;
      _statusMessage = '측정 중입니다 (약 170회 추론, 10초 안팎)...';
    });

    final timings = await _classifier.benchmark(image);
    final source = '${_classifier.lastSourceWidth ?? '?'}×${_classifier.lastSourceHeight ?? '?'}';
    final preprocess = await _classifier.benchmarkPreprocess(image);
    final decoders = await _classifier.benchmarkDecoders(image);
    final normalization = await _classifier.compareNormalizations(image);

    if (!mounted) return;
    setState(() {
      _isProcessing = false;
      _statusMessage = '측정 완료';
    });

    // 디버그 빌드는 Dart가 JIT로 돌아 전처리 시간이 몇 배 부풀려진다.
    // README에 붙일 숫자는 --profile 또는 --release 빌드에서만 유효하다.
    final buildMode = kReleaseMode
        ? 'release'
        : kProfileMode
            ? 'profile'
            : 'debug';

    final report = [
      if (kDebugMode) '⚠ debug 빌드입니다. 이 숫자는 README에 쓰지 마세요 (flutter run --profile).',
      '| 항목 | 값 |',
      '|---|---:|',
      '| 빌드 | $buildMode |',
      '| 정규화 | ${_classifier.normalization.name} |',
      '| 입력 크기 (피커 축소 후) | $source |',
      '| 디코더 | ${_classifier.decoder.name} |',
      timings?.toMarkdownRows() ?? '| 측정 실패 | — |',
      '',
      preprocess,
      '',
      decoders,
      '',
      normalization,
      '',
      '정규화 비교는 두 경로가 기기에서 동작하는지 보는 확인용입니다. '
          '한 장의 확률로 정규화를 고르지 마세요 (README 「정규화 범위 확인」).',
    ].join('\n');

    await showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('성능 측정 (중앙값)'),
        content: SingleChildScrollView(
          child: SelectableText(report, style: const TextStyle(fontSize: 12)),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('닫기'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('AutoFoto${_currentModelName != null ? " ($_currentModelName)" : ""}'),
        actions: [
          IconButton(
            icon: const Icon(Icons.fact_check_outlined),
            onPressed: _isProcessing ? null : _runDecoderAccuracyEval,
            tooltip: '디코더 정확도 평가',
          ),
          IconButton(
            icon: const Icon(Icons.speed_rounded),
            onPressed: _isProcessing ? null : _runBenchmark,
            tooltip: '성능 측정',
          ),
          IconButton(
            icon: const Icon(Icons.swap_horiz_rounded),
            onPressed: _showModelSelectionDialog,
            tooltip: '모델 선택',
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // 이미지 표시 영역
              Expanded(
                child: Container(
                  width: double.infinity,
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey.shade300),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: _selectedImage != null
                      ? ClipRRect(
                          borderRadius: BorderRadius.circular(11),
                          child: Image.file(_selectedImage!, fit: BoxFit.cover),
                        )
                      : const Center(child: Text('분류할 이미지를 선택하세요')),
                ),
              ),
              const SizedBox(height: 20),

              // 결과 표시 영역
              SizedBox(
                height: 100, // 결과 표시 영역 높이 고정
                child: _isProcessing
                    ? Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const CircularProgressIndicator(),
                          const SizedBox(height: 10),
                          Text(_statusMessage),
                        ],
                      )
                    : _classificationResult != null
                        ? Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('분류 결과:', style: Theme.of(context).textTheme.titleMedium),
                              const SizedBox(height: 4),
                              ..._classificationResult!.map((result) {
                                final label = result['label'];
                                final confidence = (result['confidence'] as double) * 100;
                                return Text(
                                  '- $label (${confidence.toStringAsFixed(1)}%)',
                                  style: Theme.of(context).textTheme.bodyLarge,
                                );
                              }).toList(),
                            ],
                          )
                        : Center(child: Text(_statusMessage)),
              ),
              const SizedBox(height: 20),

              // 액션 버튼
              ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  minimumSize: const Size(double.infinity, 50),
                  textStyle: const TextStyle(fontSize: 18),
                ),
                icon: const Icon(Icons.photo_library_outlined),
                label: const Text('사진 선택 및 분류'),
                onPressed: _isModelReady && !_isProcessing ? _pickAndClassifyImage : null,
              ),
            ],
          ),
        ),
      ),
    );
  }
}