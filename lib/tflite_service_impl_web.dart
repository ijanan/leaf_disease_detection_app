import 'dart:typed_data';
import 'dart:developer';

class TFLiteService {
  Future<void> loadModel() async {
    log('TFLiteService.loadModel() stub called on web — model inference not supported on web.',
        name: 'TFLiteService');
  }

  Future<Map<String, double>?> runInferenceOnBytes(Uint8List imageBytes) async {
    log('runInferenceOnBytes() stub called on web — return null.', name: 'TFLiteService');
    return null;
  }

  void dispose() {}
}