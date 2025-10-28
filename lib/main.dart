import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:myapp/tflite_service.dart';

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
        textTheme: GoogleFonts.montserratTextTheme(Theme.of(context).textTheme).copyWith(
          titleLarge: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white),
          bodyLarge: const TextStyle(fontSize: 16, color: Colors.black87),
          labelLarge: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white),
        ),
        appBarTheme: const AppBarTheme(
          elevation: 0,
          centerTitle: true,
        ),
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
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);

    if (pickedFile != null) {
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
      _isLoading.value = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Colors.green.shade50,
              Colors.green.shade200,
            ],
          ),
        ),
        child: Column(
          children: [
            _buildAppBar(),
            Expanded(
              child: SingleChildScrollView(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 24.0, vertical: 16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      const SizedBox(height: 20),
                      _buildImagePicker(),
                      const SizedBox(height: 40),
                      _buildButtons(),
                      const SizedBox(height: 30),
                      _buildResult(),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAppBar() {
    return Container(
      padding: const EdgeInsets.fromLTRB(24, 40, 24, 20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [Theme.of(context).primaryColor, const Color.fromARGB(255, 27, 68, 15)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: const BorderRadius.vertical(bottom: Radius.circular(30)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withAlpha(64),
            spreadRadius: 2,
            blurRadius: 10,
            offset: const Offset(0, 4),
          )
        ],
      ),
      child: Row(
        children: [
          const Icon(Icons.eco, color: Colors.white, size: 32),
          const SizedBox(width: 12),
          Text('Leaf Disease Detector', style: Theme.of(context).textTheme.titleLarge),
        ],
      ),
    );
  }

  Widget _buildImagePicker() {
    return ValueListenableBuilder<Uint8List?>(
      valueListenable: _imageBytes,
      builder: (context, imageBytes, child) {
        return Container(
          height: 300,
          decoration: BoxDecoration(
            color: Colors.white.withAlpha(204),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Theme.of(context).primaryColor, width: 2),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withAlpha(26),
                spreadRadius: 2,
                blurRadius: 8,
              ),
            ],
          ),
          child: imageBytes != null
              ? ClipRRect(
                  borderRadius: BorderRadius.circular(18),
                  child: Image.memory(imageBytes, fit: BoxFit.cover, width: double.infinity),
                )
              : Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(Icons.cloud_upload_outlined, size: 80, color: Theme.of(context).primaryColor.withAlpha(178)),
                      const SizedBox(height: 16),
                      Text(
                        'Upload an image to get started',
                        style: TextStyle(
                          fontSize: 16,
                          color: Colors.grey.shade700,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ),
        );
      },
    );
  }

  Widget _buildButtons() {
    return Column(
      children: [
        _buildButton(
          icon: Icons.camera_alt,
          label: 'Scan with Camera',
          onPressed: () => _pickImage(ImageSource.camera),
        ),
        const SizedBox(height: 20),
        _buildButton(
          icon: Icons.photo_library,
          label: 'Upload from Gallery',
          onPressed: () => _pickImage(ImageSource.gallery),
        ),
      ],
    );
  }

  Widget _buildButton({required IconData icon, required String label, required VoidCallback onPressed}) {
    return ElevatedButton.icon(
      onPressed: onPressed,
      icon: Icon(icon, size: 28),
      label: Text(label),
      style: ElevatedButton.styleFrom(
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
        backgroundColor: Theme.of(context).primaryColor,
        foregroundColor: Colors.white,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
        elevation: 8,
        shadowColor: Theme.of(context).hintColor.withAlpha(128),
        textStyle: Theme.of(context).textTheme.labelLarge,
      ),
    );
  }

  Widget _buildResult() {
    return ValueListenableBuilder<bool>(
      valueListenable: _isLoading,
      builder: (context, isLoading, child) {
        if (isLoading) {
          return Center(
            child: CircularProgressIndicator(
              valueColor: AlwaysStoppedAnimation<Color>(Theme.of(context).primaryColor),
            ),
          );
        }
        return ValueListenableBuilder<String?>(
          valueListenable: _result,
          builder: (context, result, child) {
            if (result == null) {
              return const SizedBox.shrink();
            }
            return Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withAlpha(26),
                    spreadRadius: 2,
                    blurRadius: 10,
                  )
                ],
              ),
              child: Column(
                children: [
                  Text(
                    'Prediction',
                    style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Theme.of(context).primaryColor),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    result,
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.w600, color: Theme.of(context).hintColor),
                  ),
                  const SizedBox(height: 10),
                  ValueListenableBuilder<double?>(
                    valueListenable: _confidence,
                    builder: (context, confidence, child) {
                      if (confidence == null) return const SizedBox.shrink();
                      return Text(
                        'Confidence: ${(confidence * 100).toStringAsFixed(2)}%',
                        style: const TextStyle(fontSize: 16, color: Colors.black54, fontWeight: FontWeight.w500),
                      );
                    },
                  ),
                ],
              ),
            );
          },
        );
      },
    );
  }
}
