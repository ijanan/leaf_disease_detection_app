import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';

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
  final ValueNotifier<bool> _isLoading = ValueNotifier(false);

  final TFLiteService _tfliteService = TFLiteService();

  @override
  void initState() {
    super.initState();
    _tfliteService.loadModel();
  }

  @override
  void dispose() {
    _imageBytes.dispose();
    _result.dispose();
    _confidence.dispose();
    _isLoading.dispose();
    _tfliteService.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final picker = ImagePicker();
      final pickedFile = await picker.pickImage(
          source: source, maxWidth: 1024, maxHeight: 1024);
      if (pickedFile == null) return;
      final bytes = await pickedFile.readAsBytes();

      _imageBytes.value = bytes;
      _isLoading.value = true;
      _result.value = null;
      _confidence.value = null;

      final result = await _tfliteService.runInferenceOnBytes(bytes);
      if (result != null && result.isNotEmpty) {
        _result.value = result.keys.first;
        _confidence.value = result.values.first;
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Leaf Disease Detector'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            ValueListenableBuilder<Uint8List?>(
              valueListenable: _imageBytes,
              builder: (context, bytes, child) {
                if (bytes == null) {
                  return Container(
                    height: 250,
                    color: Colors.grey[200],
                    child: const Center(child: Text('No image selected')),
                  );
                }
                return Image.memory(bytes, height: 250, fit: BoxFit.contain);
              },
            ),
            const SizedBox(height: 16),
            ValueListenableBuilder<bool>(
              valueListenable: _isLoading,
              builder: (context, loading, child) {
                if (loading) return const CircularProgressIndicator();
                return Column(
                  children: [
                    ValueListenableBuilder<String?>(
                      valueListenable: _result,
                      builder: (context, result, child) {
                        if (result == null) return const SizedBox();
                        return Text('Result: $result',
                            style: const TextStyle(fontSize: 18));
                      },
                    ),
                    ValueListenableBuilder<double?>(
                      valueListenable: _confidence,
                      builder: (context, conf, child) {
                        if (conf == null) return const SizedBox();
                        return Text(
                            'Confidence: ${(conf * 100).toStringAsFixed(1)}%');
                      },
                    ),
                  ],
                );
              },
            ),
            const Spacer(),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Camera'),
                  onPressed: kIsWeb
                      ? null // camera via image_picker may not be supported on web; disable
                      : () => _pickImage(ImageSource.camera),
                ),
                ElevatedButton.icon(
                  icon: const Icon(Icons.photo),
                  label: const Text('Gallery'),
                  onPressed: () => _pickImage(ImageSource.gallery),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
