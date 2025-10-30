import 'dart:developer';
import 'dart:typed_data';
import 'dart:async';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'dart:ui' as ui;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:math' as math;

class TFLiteService {
  Interpreter? _interpreter;
  List<String>? _labels;
  String? _lastLoadError;

  /// Last error message observed during model load (if any).
  String? get lastLoadError => _lastLoadError;

  /// Loads the TFLite model and labels. Returns true on success, false on failure.
  Future<bool> loadModel() async {
    try {
      // Quick asset existence check to produce a clearer error earlier if the
      // model file is missing from the bundled assets. This will throw if the
      // asset isn't packaged.
      try {
        final modelAsset = await rootBundle.load('assets/leaf_disease_efficientnetb0.tflite');
        log('Model asset found, size=${modelAsset.lengthInBytes} bytes', name: 'TFLiteService');
      } catch (e) {
        log('Model asset check failed: $e', name: 'TFLiteService', level: 1000);
        // continue to let Interpreter.fromAsset throw a clearer error too
      }

      // Try loading the model. Some packaging setups require the plain
      // filename, others use the full asset path. Try both so the app works
      // regardless of how the asset was declared.
      try {
        _interpreter = await Interpreter.fromAsset(
          'leaf_disease_efficientnetb0.tflite',
          options: InterpreterOptions(),
        );
      } catch (e1) {
        log('Interpreter.fromAsset("leaf_disease_efficientnetb0.tflite") failed: $e1', name: 'TFLiteService', level: 900);
        try {
          _interpreter = await Interpreter.fromAsset(
            'assets/leaf_disease_efficientnetb0.tflite',
            options: InterpreterOptions(),
          );
        } catch (e2) {
          // Rethrow a combined error so the outer catch records it.
          throw Exception('Interpreter.fromAsset failed for both keys: $e1 ; $e2');
        }
      }

      final labelsData = await rootBundle.loadString('assets/labels.txt');
      _labels = labelsData
          .split('\n')
          .map((s) => s.trim())
          .where((s) => s.isNotEmpty)
          .toList();

      final inputTensor = _interpreter!.getInputTensor(0);
      final outputTensor = _interpreter!.getOutputTensor(0);
      log('Input tensor: shape=${inputTensor.shape}, type=${inputTensor.type}',
          name: 'TFLiteService');
      log('Output tensor: shape=${outputTensor.shape}, type=${outputTensor.type}',
          name: 'TFLiteService');

      log('Model and labels loaded successfully', name: 'TFLiteService');
      _lastLoadError = null;
      return true;
    } catch (e) {
      log('Error loading model: $e', name: 'TFLiteService', level: 1000);
      _interpreter = null;
      _labels = null;
      _lastLoadError = e.toString();
      return false;
    }
  }

  Future<Map<String, double>?> runInferenceOnBytes(Uint8List imageBytes) async {
    if (_interpreter == null || _labels == null) {
      log('Model not loaded', name: 'TFLiteService', level: 900);
      return null;
    }

    // Inspect input tensor type so we can prepare data in the expected dtype
    final inputTensor = _interpreter!.getInputTensor(0);
    final outputTensor = _interpreter!.getOutputTensor(0);
  final outputShape = outputTensor.shape;

    log('Running inference: inputType=${inputTensor.type}, inputShape=${inputTensor.shape}, outputShape=${outputShape}', name: 'TFLiteService');

    // Delegate to the debug-friendly runner and return the top result for
    // backwards compatibility.
    final debug = await runInferenceDebug(imageBytes);
    if (debug == null) return null;
    if (debug.containsKey('error')) {
      // Respect errors returned by the debug runner.
      log('Inference debug reported error: ${debug['error']}', name: 'TFLiteService', level: 900);
      return null;
    }
    final raw = debug['rawOutputs'] as List<double>?;
    if (raw == null) return null;
    return _getTopResult(raw);
  }

  /// Run inference and return detailed debug information useful for
  /// inspecting model input/output types, shapes and raw outputs.
  Future<Map<String, dynamic>?> runInferenceDebug(Uint8List imageBytes) async {
    if (_interpreter == null || _labels == null) {
      log('Model not loaded', name: 'TFLiteService', level: 900);
      return null;
    }

    final inputTensor = _interpreter!.getInputTensor(0);
    final outputTensor = _interpreter!.getOutputTensor(0);
    final outputShape = outputTensor.shape;
    final int numClasses = (outputShape.length == 1) ? outputShape[0] : outputShape[1];

    log('Running inference: inputType=${inputTensor.type}, inputShape=${inputTensor.shape}, outputShape=${outputShape}', name: 'TFLiteService');

    try {
      final sig = _imageSignature(imageBytes, 8);
      final inputTypeStr = inputTensor.type.toString().toLowerCase();
      final isUint8 = inputTypeStr.contains('uint8') || inputTypeStr.contains('u8');
      if (isUint8) {
        final u8 = _preprocessImageAsUint8(imageBytes, inputTensor.shape[1], inputTensor.shape[2]);
        if (u8 == null) return {'error': 'Failed to preprocess image as uint8'};
        final output = List.generate(1, (_) => List.filled(numClasses, 0.0));
  // Wrap the typed buffer in a List to represent the batch dimension when
  // passing to the interpreter. This is accepted by tflite_flutter.
  _interpreter!.run([u8], output);
        final results = (output[0] as List).map((e) => (e as num).toDouble()).toList();
        // uint8 models often emit probabilities already; ensure we have
        // a probability vector (apply softmax if values don't sum to ~1).
        final probs = _maybeSoftmax(results);
        log('Raw output (uint8 model) signature=$sig probs=$probs', name: 'TFLiteService');
        return {
          'inputType': inputTensor.type.toString(),
          'inputShape': inputTensor.shape,
          'outputShape': outputShape,
          'rawOutputs': probs,
          'topResults': _topK(probs, math.min(3, probs.length)),
          'labels': _labels,
          'imageSignature': sig,
        };
      } else {
  // Prepare a typed Float32List flattened in NHWC order — this is more
  // reliable with the interpreter than large nested Dart lists.
  final input = await _preprocessImageAsFloat(imageBytes, inputTensor.shape[1], inputTensor.shape[2]);
  if (input == null) return {'error': 'Failed to preprocess image as float (decode/convert failed)'};
        final output = List.generate(1, (_) => List.filled(numClasses, 0.0));
  // Wrap the Float32List in a list to represent the batch dimension.
  _interpreter!.run([input], output);
        // output may be List<List<double>>; ensure we cast to List<double>
        final results = (output[0] as List).map((e) => (e as num).toDouble()).toList();
        final probs = _maybeSoftmax(results);
        log('Raw output (float model) signature=$sig probs=$probs', name: 'TFLiteService');
        return {
          'inputType': inputTensor.type.toString(),
          'inputShape': inputTensor.shape,
          'outputShape': outputShape,
          'rawOutputs': probs,
          'topResults': _topK(probs, math.min(3, probs.length)),
          'labels': _labels,
          'imageSignature': sig,
        };
      }
    } catch (e) {
      log('Error running inference: $e', name: 'TFLiteService', level: 1000);
      return {'error': 'Exception during inference: ${e.toString()}'};
    }
  }

  // Prepare Uint8List input flattened in NHWC order
  Uint8List? _preprocessImageAsUint8(Uint8List imageBytes, int targetH, int targetW) {
    try {
      final image = img.decodeImage(imageBytes);
      if (image == null) return null;
      final resized = img.copyResize(image, width: targetW, height: targetH);
      final bytes = resized.getBytes(); // RGBA
      final out = Uint8List(targetH * targetW * 3);
      int dst = 0;
      for (int y = 0; y < targetH; y++) {
        for (int x = 0; x < targetW; x++) {
          final int idx = (y * targetW + x) * 4;
          out[dst++] = bytes[idx];
          out[dst++] = bytes[idx + 1];
          out[dst++] = bytes[idx + 2];
        }
      }
      // Return as a flattened list wrapped in an outer list to match interpreter input
      return out;
    } catch (e) {
      log('Error preprocessing uint8 image: $e', name: 'TFLiteService', level: 1000);
      return null;
    }
  }

  // Prepare a flattened Float32List in NHWC order normalized to 0..1
  Future<Float32List?> _preprocessImageAsFloat(Uint8List imageBytes, int targetH, int targetW) async {
    try {
      // First attempt: decode with package:image
      img.Image? image = img.decodeImage(imageBytes);
      if (image == null) {
        // Robust engine fallbacks:
        // 1) Try ui.decodeImageFromList which sometimes handles formats that
        //    package:image cannot. If decoded, scale it using a PictureRecorder
        //    so we control the final size.
        // 2) If that fails, fall back to instantiateImageCodec with targetWidth/Height.
        try {
          // Try decodeImageFromList first (completes via callback)
          final completer = Completer<ui.Image>();
          ui.decodeImageFromList(imageBytes, (img) => completer.complete(img));
          final decoded = await completer.future;

          ui.Image resizedUiImage = decoded;
          if (decoded.width != targetW || decoded.height != targetH) {
            // Scale using a PictureRecorder + Canvas -> gives a ui.Image of the
            // target size.
            final recorder = ui.PictureRecorder();
            final canvas = ui.Canvas(recorder);
            final paint = ui.Paint();
            final srcRect = ui.Rect.fromLTWH(0, 0, decoded.width.toDouble(), decoded.height.toDouble());
            final dstRect = ui.Rect.fromLTWH(0, 0, targetW.toDouble(), targetH.toDouble());
            canvas.drawImageRect(decoded, srcRect, dstRect, paint);
            final picture = recorder.endRecording();
            resizedUiImage = await picture.toImage(targetW, targetH);
          }

          final byteData = await resizedUiImage.toByteData(format: ui.ImageByteFormat.rawRgba);
          if (byteData == null) return null;
          final bytes = byteData.buffer.asUint8List();

          final out = Float32List(targetH * targetW * 3);
          int dst = 0;
          for (int i = 0; i < bytes.length; i += 4) {
            final int r = bytes[i];
            final int g = bytes[i + 1];
            final int b = bytes[i + 2];
            out[dst++] = r / 255.0;
            out[dst++] = g / 255.0;
            out[dst++] = b / 255.0;
          }
          return out;
        } catch (e, st) {
          log('decodeImageFromList fallback failed, trying instantiateImageCodec: $e\n$st', name: 'TFLiteService', level: 900);
          // Try instantiateImageCodec as a final fallback
          try {
            final codec = await ui.instantiateImageCodec(
              imageBytes,
              targetWidth: targetW,
              targetHeight: targetH,
            );
            final frame = await codec.getNextFrame();
            final uiImage = frame.image;
            final byteData = await uiImage.toByteData(format: ui.ImageByteFormat.rawRgba);
            if (byteData == null) return null;
            final bytes = byteData.buffer.asUint8List();

            final out = Float32List(targetH * targetW * 3);
            int dst = 0;
            for (int i = 0; i < bytes.length; i += 4) {
              final int r = bytes[i];
              final int g = bytes[i + 1];
              final int b = bytes[i + 2];
              out[dst++] = r / 255.0;
              out[dst++] = g / 255.0;
              out[dst++] = b / 255.0;
            }
            return out;
          } catch (e2, st2) {
            log('Fallback decode+resize failed entirely: $e2\n$st2', name: 'TFLiteService', level: 1000);
            return null;
          }
        }
      }

      final resized = img.copyResize(image, width: targetW, height: targetH);
      final bytes = resized.getBytes(); // RGBA

      final out = Float32List(targetH * targetW * 3);
      int dst = 0;
      for (int y = 0; y < targetH; y++) {
        for (int x = 0; x < targetW; x++) {
          final int idx = (y * targetW + x) * 4;
          final int r = bytes[idx];
          final int g = bytes[idx + 1];
          final int b = bytes[idx + 2];
          out[dst++] = r / 255.0;
          out[dst++] = g / 255.0;
          out[dst++] = b / 255.0;
        }
      }
      return out;
    } catch (e, st) {
      log('Error preprocessing float image: $e\n$st', name: 'TFLiteService', level: 1000);
      return null;
    }
  }

  // (old _preprocessImage removed - replaced by _preprocessImageAsUint8 and _preprocessImageAsFloat)

  // Returns the first `n` bytes of the image as a hex string for lightweight
  // debugging (does not reveal full image contents).
  String _imageSignature(Uint8List bytes, int n) {
    final len = math.min(n, bytes.length);
    final part = bytes.sublist(0, len);
    final sb = StringBuffer();
    for (final b in part) {
      sb.write(b.toRadixString(16).padLeft(2, '0'));
    }
    return sb.toString();
  }

  // Apply softmax if the values look like logits or do not sum to ~1.
  List<double> _maybeSoftmax(List<double> values) {
    if (values.isEmpty) return values;
    final sum = values.fold(0.0, (a, b) => a + b);
    final hasNegative = values.any((v) => v.isNegative);
    if (hasNegative || sum <= 0.0 || (sum < 0.99 || sum > 1.01)) {
      return _safeSoftmax(values);
    }
    // Already probabilities
    return values.map((v) => v.clamp(0.0, 1.0)).toList();
  }

  List<double> _safeSoftmax(List<double> logits) {
    final maxLogit = logits.reduce((a, b) => a > b ? a : b);
    final exps = logits.map((l) => math.exp(l - maxLogit)).toList();
    final sumExp = exps.fold(0.0, (a, b) => a + b);
    if (sumExp == 0) return List.filled(logits.length, 0.0);
    return exps.map((e) => e / sumExp).toList();
  }

  // Return top-k label+score maps
  List<Map<String, double>> _topK(List<double> probs, int k) {
    final indexed = List<int>.generate(probs.length, (i) => i);
    indexed.sort((a, b) => probs[b].compareTo(probs[a]));
    final take = math.min(k, probs.length);
    final out = <Map<String, double>>[];
    for (int i = 0; i < take; i++) {
      final idx = indexed[i];
      final label = (idx < (_labels?.length ?? 0)) ? _labels![idx].trim() : 'class_$idx';
      out.add({label: probs[idx]});
    }
    return out;
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