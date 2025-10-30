// Conditional export: choose native implementation on IO platforms, and a web stub on web.
export 'tflite_service_impl_io.dart'
    if (dart.library.html) 'tflite_service_impl_web.dart';
