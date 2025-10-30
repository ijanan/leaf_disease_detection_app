import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'dart:ui';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image/image.dart' as img;

import 'tflite_service.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Leaf Disease Detector',
      theme: ThemeData(
        primaryColor: const Color.fromARGB(255, 24, 58, 11),
        hintColor: const Color.fromARGB(255, 28, 69, 31),
        textTheme: GoogleFonts.montserratTextTheme(Theme.of(context).textTheme),
        appBarTheme: const AppBarTheme(elevation: 0, centerTitle: true),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ValueNotifier<Uint8List?> _imageBytes = ValueNotifier(null);
  final ValueNotifier<String?> _result = ValueNotifier(null);
  final ValueNotifier<double?> _confidence = ValueNotifier(null);
  final ValueNotifier<Map<String, dynamic>?> _lastDebug = ValueNotifier(null);
  final ValueNotifier<bool> _isLoading = ValueNotifier(false);
  final ValueNotifier<bool> _modelLoaded = ValueNotifier(false);

  final TFLiteService _tfliteService = TFLiteService();

  @override
  void initState() {
    super.initState();
    // load model and flip _modelLoaded when ready (only set true on success)
    _tfliteService.loadModel().then((ok) {
      _modelLoaded.value = ok == true;
    });
  }

  @override
  void dispose() {
    _imageBytes.dispose();
    _result.dispose();
    _confidence.dispose();
    _isLoading.dispose();
    _lastDebug.dispose();
    _modelLoaded.dispose();
    _tfliteService.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      // Ensure permissions on native platforms
      final allowed = await _ensurePermissions(source);
      if (!allowed) {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
            content: Text('Permission denied. Cannot access camera/gallery.')));
        return;
      }

      // Wait for model to finish loading (if it hasn't yet)
      if (!_modelLoaded.value) {
        _isLoading.value = true;
        final ok = await _tfliteService.loadModel();
        _modelLoaded.value = ok == true;
        _isLoading.value = false;
        if (!ok) {
          // show immediate message to user
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                content: Text('Model failed to load. Check logs.')));
          }
        }
      }

      final picker = ImagePicker();
      XFile? pickedFile;
      try {
        pickedFile = await picker.pickImage(
            source: source,
            maxWidth: 1024,
            maxHeight: 1024,
            preferredCameraDevice: CameraDevice.rear);
      } catch (e) {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Could not open camera: ${e.toString()}')));
        return;
      }
      if (pickedFile == null) return;
      final bytes = await pickedFile.readAsBytes();

      _imageBytes.value = bytes;
      _isLoading.value = true;
      _result.value = null;
      _confidence.value = null;

      // Run the debug inference which returns raw outputs and metadata so we
      // can display them in-app for quick verification without adb/logcat.
      final debug = await _tfliteService.runInferenceDebug(bytes);
      _lastDebug.value = debug;
      if (debug != null) {
        if (debug.containsKey('error')) {
          _result.value = 'Error: ${debug['error']}';
          _confidence.value = 0.0;
        } else if (debug['topResults'] != null) {
          final topList = (debug['topResults'] as List<dynamic>);
          if (topList.isNotEmpty) {
            final first = topList.first as Map<dynamic, dynamic>;
            final label = first.keys.first?.toString() ?? 'Unknown';
            final score = (first.values.first as num).toDouble();
            _result.value = label;
            _confidence.value = score;
          }
        } else if (debug['topResult'] != null) {
          // Backwards compatibility
          final top = (debug['topResult'] as Map<String, double>);
          if (top.isNotEmpty) {
            _result.value = top.keys.first;
            _confidence.value = top.values.first;
          }
        } else {
          _result.value = "Could not classify image.";
          _confidence.value = 0.0;
        }
      } else {
        _result.value = "Could not classify image.";
        _confidence.value = 0.0;
      }
    } catch (e) {
      _result.value = 'Error: $e';
      _confidence.value = 0.0;
    } finally {
      _isLoading.value = false;
    }
  }

  Future<bool> _ensurePermissions(ImageSource source) async {
    if (kIsWeb) return true;
    try {
      if (source == ImageSource.camera) {
        final status = await Permission.camera.request();
        if (status.isPermanentlyDenied) {
          // tell user how to enable
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                content: Text(
                    'Camera permission permanently denied. Open settings to enable.')));
          }
          openAppSettings();
          return false;
        }
        return status.isGranted;
      } else {
        // gallery / photos
        // request photos on iOS and storage on Android
        final photos = await Permission.photos.request();
        if (photos.isGranted) return true;
        if (photos.isPermanentlyDenied) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                content: Text(
                    'Photo permission permanently denied. Open settings to enable.')));
          }
          openAppSettings();
          return false;
        }
        final storage = await Permission.storage.request();
        if (storage.isPermanentlyDenied) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(const SnackBar(
                content: Text(
                    'Storage permission permanently denied. Open settings to enable.')));
          }
          openAppSettings();
          return false;
        }
        return storage.isGranted;
      }
    } catch (_) {
      return false;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text('Leaf Disease Detector'),
      ),
      body: Stack(
        children: [
          // decorative blobs
          Positioned(
            left: -80,
            top: -120,
            child: Container(
              width: 260,
              height: 260,
              decoration: BoxDecoration(
                gradient: RadialGradient(
                    colors: [Colors.green.shade100, Colors.green.shade50]),
                borderRadius: BorderRadius.circular(200),
              ),
            ),
          ),
          Positioned(
            right: -100,
            bottom: -140,
            child: Container(
              width: 320,
              height: 320,
              decoration: BoxDecoration(
                gradient: RadialGradient(
                    colors: [Colors.teal.shade100, Colors.teal.shade50]),
                borderRadius: BorderRadius.circular(200),
              ),
            ),
          ),
          // main content
          SingleChildScrollView(
            child: Container(
              padding: const EdgeInsets.fromLTRB(20, 110, 20, 40),
              child: Column(
                children: [
                  // preview glass card
                  ValueListenableBuilder<Uint8List?>(
                    valueListenable: _imageBytes,
                    builder: (context, bytes, child) {
                      return ClipRRect(
                        borderRadius: BorderRadius.circular(20),
                        child: BackdropFilter(
                          filter: ImageFilter.blur(sigmaX: 6, sigmaY: 6),
                          child: Container(
                            width: double.infinity,
                            height: 300,
                            decoration: BoxDecoration(
                              color: const Color.fromRGBO(255, 255, 255, 0.65),
                              borderRadius: BorderRadius.circular(20),
                              boxShadow: const [
                                BoxShadow(
                                  color: Color.fromRGBO(0, 0, 0, 0.08),
                                  blurRadius: 12,
                                  offset: Offset(0, 8),
                                )
                              ],
                            ),
                            child: bytes == null
                                ? const Center(
                                    child: Column(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Icon(Icons.camera_alt_outlined,
                                            size: 54, color: Colors.green),
                                        SizedBox(height: 12),
                                        Text(
                                            'Tap the camera to take a photo, or open gallery',
                                            textAlign: TextAlign.center,
                                            style: TextStyle(
                                                color: Colors.black54)),
                                      ],
                                    ),
                                  )
                                : ClipRRect(
                                    borderRadius: BorderRadius.circular(20),
                                    child: Image.memory(bytes,
                                        fit: BoxFit.cover,
                                        width: double.infinity,
                                        height: 300),
                                  ),
                          ),
                        ),
                      );
                    },
                  ),
                  const SizedBox(height: 18),

                  // result card
                  ValueListenableBuilder<bool>(
                    valueListenable: _isLoading,
                    builder: (context, loading, child) {
                      return Card(
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(14)),
                        elevation: 4,
                        child: Padding(
                          padding: const EdgeInsets.all(14),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  const Text('Result',
                                      style: TextStyle(
                                          fontWeight: FontWeight.bold,
                                          fontSize: 16)),
                                  if (loading)
                                    const SizedBox(
                                        width: 18,
                                        height: 18,
                                        child: CircularProgressIndicator(
                                            strokeWidth: 2))
                                  else
                                    ValueListenableBuilder<bool>(
                                      valueListenable: _modelLoaded,
                                      builder: (context, loaded, child) {
                                        return Row(children: [
                                          Icon(
                                              loaded
                                                  ? Icons.check_circle
                                                  : Icons.hourglass_bottom,
                                              color: loaded
                                                  ? Colors.green
                                                  : Colors.orange),
                                          const SizedBox(width: 6),
                                          Text(
                                              loaded
                                                  ? 'Model ready'
                                                  : 'Loading model',
                                              style: const TextStyle(
                                                  fontSize: 12)),
                                        ]);
                                      },
                                    ),
                                ],
                              ),
                              const SizedBox(height: 12),
                              ValueListenableBuilder<String?>(
                                valueListenable: _result,
                                builder: (context, result, child) {
                                  // If the model isn't loaded and the service has a
                                  // last load error, show that to the user so they
                                  // don't just see a generic 'Could not classify'.
                                  if (!_modelLoaded.value &&
                                      _tfliteService.lastLoadError != null) {
                                    return Text(
                                        'Model load error: ${_tfliteService.lastLoadError}',
                                        style: const TextStyle(
                                            fontSize: 16, color: Colors.red));
                                  }
                                  return Text(result ?? 'No prediction yet',
                                      style: const TextStyle(fontSize: 18));
                                },
                              ),
                              const SizedBox(height: 12),
                              ValueListenableBuilder<double?>(
                                valueListenable: _confidence,
                                builder: (context, conf, child) {
                                  final pct = conf == null
                                      ? 0.0
                                      : (conf.clamp(0.0, 1.0));
                                  return Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      ClipRRect(
                                          borderRadius:
                                              BorderRadius.circular(8),
                                          child: LinearProgressIndicator(
                                              value: pct, minHeight: 8)),
                                      const SizedBox(height: 6),
                                      Text(
                                          '${(pct * 100).toStringAsFixed(1)}% confidence',
                                          style: const TextStyle(
                                              fontSize: 12,
                                              color: Colors.black54)),
                                    ],
                                  );
                                },
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),

                  const SizedBox(height: 26),
                  // action buttons as responsive chips (wrap to next line on narrow screens)
                  Wrap(
                    alignment: WrapAlignment.center,
                    spacing: 12,
                    runSpacing: 12,
                    children: [
                      FloatingActionButton.extended(
                        heroTag: 'camera',
                        label: const Text('Camera'),
                        icon: const Icon(Icons.camera_alt),
                        backgroundColor: Colors.green[600],
                        onPressed: () async {
                          if (kIsWeb) {
                            if (!mounted) return;
                            ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                    content:
                                        Text('Camera not supported on web.')));
                            return;
                          }
                          await _pickImage(ImageSource.camera);
                        },
                      ),
                      const SizedBox(width: 16),
                      FloatingActionButton.extended(
                        heroTag: 'gallery',
                        label: const Text('Gallery'),
                        icon: const Icon(Icons.photo_library),
                        backgroundColor: Colors.white,
                        foregroundColor: Colors.black87,
                        onPressed: () async =>
                            await _pickImage(ImageSource.gallery),
                      ),
                      const SizedBox(width: 16),
                      FloatingActionButton.extended(
                        heroTag: 'sample',
                        label: const Text('Test'),
                        icon: const Icon(Icons.bug_report_outlined),
                        backgroundColor: Colors.blueGrey[700],
                        onPressed: () async {
                          // Create a synthetic sample image (224x224) and run inference
                          try {
                            final int target = 224;
                            final sample =
                                img.Image(width: target, height: target);
                            // fill with a green color that looks like a leaf
                            for (int y = 0; y < target; y++) {
                              for (int x = 0; x < target; x++) {
                                sample.setPixelRgba(x, y, 80, 160, 60, 255);
                              }
                            }
                            // add a darker spot to vary the image a bit
                            for (int y = 40; y < 120; y++) {
                              for (int x = 40; x < 120; x++) {
                                sample.setPixelRgba(x, y, 60, 120, 40, 255);
                              }
                            }
                            final jpg = img.encodeJpg(sample, quality: 90);
                            final Uint8List bytes = Uint8List.fromList(jpg);

                            _imageBytes.value = bytes;
                            _isLoading.value = true;
                            _result.value = null;
                            _confidence.value = null;

                            final debug =
                                await _tfliteService.runInferenceDebug(bytes);
                            _lastDebug.value = debug;
                            if (debug != null) {
                              if (debug.containsKey('error')) {
                                _result.value = 'Error: ${debug['error']}';
                                _confidence.value = 0.0;
                              } else if (debug['topResults'] != null) {
                                final topList =
                                    (debug['topResults'] as List<dynamic>);
                                if (topList.isNotEmpty) {
                                  final first =
                                      topList.first as Map<dynamic, dynamic>;
                                  final label =
                                      first.keys.first?.toString() ?? 'Unknown';
                                  final score =
                                      (first.values.first as num).toDouble();
                                  _result.value = label;
                                  _confidence.value = score;
                                }
                              } else if (debug['topResult'] != null) {
                                final top =
                                    (debug['topResult'] as Map<String, double>);
                                if (top.isNotEmpty) {
                                  _result.value = top.keys.first;
                                  _confidence.value = top.values.first;
                                }
                              } else {
                                _result.value =
                                    'Could not classify sample image.';
                                _confidence.value = 0.0;
                              }
                            }
                          } catch (e) {
                            _result.value = 'Error: $e';
                            _confidence.value = 0.0;
                          } finally {
                            _isLoading.value = false;
                          }
                        },
                      ),
                    ],
                  ),
                  const SizedBox(height: 18),
                  // Debug panel showing raw outputs and model info (toggle is simple)
                  ValueListenableBuilder<Map<String, dynamic>?>(
                    valueListenable: _lastDebug,
                    builder: (context, debug, child) {
                      if (debug == null) return const SizedBox.shrink();
                      final inputType = debug['inputType'] ?? '';
                      final inputShape = debug['inputShape']?.toString() ?? '';
                      final outputShape =
                          debug['outputShape']?.toString() ?? '';
                      final raw = debug['rawOutputs'] as List<dynamic>? ?? [];
                      final error = debug['error'] as String?;
                      return Card(
                        color: Colors.grey[50],
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12)),
                        child: Padding(
                          padding: const EdgeInsets.all(12),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text('Debug',
                                  style:
                                      TextStyle(fontWeight: FontWeight.bold)),
                              const SizedBox(height: 8),
                              Text('Input: $inputType $inputShape',
                                  style: const TextStyle(fontSize: 12)),
                              const SizedBox(height: 4),
                              Text('Output shape: $outputShape',
                                  style: const TextStyle(fontSize: 12)),
                              if (error != null) ...[
                                const SizedBox(height: 6),
                                Text('Error: $error',
                                    style: const TextStyle(
                                        fontSize: 12, color: Colors.red)),
                              ],
                              const SizedBox(height: 6),
                              Text(
                                  'Image sig: ${debug['imageSignature'] ?? ''}',
                                  style: const TextStyle(
                                      fontSize: 10, color: Colors.black45)),
                              const SizedBox(height: 8),
                              Text('Raw outputs: ${raw.take(20).toList()}',
                                  style: const TextStyle(fontSize: 12)),
                              const SizedBox(height: 6),
                              Text(
                                  'Labels: ${(debug['labels'] as List<dynamic>?)?.join(', ') ?? ''}',
                                  style: const TextStyle(fontSize: 12)),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
