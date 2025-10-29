import 'dart:typed_data';
import 'dart:developer';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class TFLiteService {
  Interpreter? _interpreter;
  List<String>? _labels;

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'leaf_disease_efficientnetb0.tflite',
        options: InterpreterOptions(),
      );

      final labelsData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelsData.split('\n').map((s) => s.trim()).where((s) => s.isNotEmpty).toList();

      final inputTensor = _interpreter!.getInputTensor(0);
      final outputTensor = _interpreter!.getOutputTensor(0);
      log('Input tensor: shape=${inputTensor.shape}, type=${inputTensor.type}', name: 'TFLiteService');
      log('Output tensor: shape=${outputTensor.shape}, type=${outputTensor.type}', name: 'TFLiteService');

      log('Model and labels loaded successfully', name: 'TFLiteService');
    } catch (e) {
      log('Error loading model: $e', name: 'TFLiteService', level: 1000);
    }
  }

  Future<Map<String, double>?> runInferenceOnBytes(Uint8List imageBytes) async {
    if (_interpreter == null || _labels == null) {
      log('Model not loaded', name: 'TFLiteService', level: 900);
      return null;
    }

    final input = _preprocessImage(imageBytes);
    if (input == null) return null;

    final outputTensor = _interpreter!.getOutputTensor(0);
    final outputShape = outputTensor.shape;
    final int numClasses = (outputShape.length == 1) ? outputShape[0] : outputShape[1];

    // Output as a batched 2D list: [1, numClasses]
    final output = List.generate(1, (_) => List.filled(numClasses, 0.0));

    try {
      // pass the batched input directly (shape: [1,224,224,3])
      _interpreter!.run(input, output);
    } catch (e) {
      log('Error running inference: $e', name: 'TFLiteService', level: 1000);
      return null;
    }

    final results = (output[0] as List).cast<double>();
    final topResult = _getTopResult(results);

    return topResult;
  }

  /// Returns a batched 4D list [1, height, width, 3] with values normalized to 0..1
  List<List<List<List<double>>>>? _preprocessImage(Uint8List imageBytes) {
    try {
      final image = img.decodeImage(imageBytes);
      if (image == null) return null;

      final resizedImage = img.copyResize(image, width: 224, height: 224);

      final imageMatrix = List.generate(
        224,
        (y) => List.generate(
          224,
          (x) {
            final pixel = resizedImage.getPixel(x, y);
            return [
              img.getRed(pixel) / 255.0,
              img.getGreen(pixel) / 255.0,
              img.getBlue(pixel) / 255.0,
            ];
          },
        ),
      );

      // add batch dim: [1, height, width, channels]
      return [imageMatrix];
    } catch (e) {
      log('Error preprocessing image: $e', name: 'TFLiteService', level: 1000);
      return null;
    }
  }

  Map<String, double> _getTopResult(List<double> results) {
    double maxScore = -double.infinity;
    int maxIndex = -1;

    for (int i = 0; i < results.length; i++) {
      if (results[i] > maxScore) {
        maxScore = results[i];
        maxIndex = i;
      }
    }

    if (maxIndex != -1 && _labels != null && maxIndex < _labels!.length) {
      final label = _labels![maxIndex].trim();
      return {label: maxScore};
    }

    return {};
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
  }
}
