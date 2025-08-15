import 'package:flutter/material.dart';
import 'package:lottie/lottie.dart';
import 'package:camera/camera.dart';
import 'pose_detector_screen.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  bool _isCameraReady = false;
  CameraDescription? _camera;

  @override
  void initState() {
    super.initState();
    _initCameraAndNavigate();
  }

  Future<void> _initCameraAndNavigate() async {
    final cameras = await availableCameras();
    final firstCamera = cameras.first;
    await Future.delayed(const Duration(milliseconds: 4000));
    if (mounted) {
      setState(() {
        _isCameraReady = true;
        _camera = firstCamera;
      });
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(
          builder: (_) => PoseDetectorView(camera: firstCamera),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Lottie.asset('assets/animations/splash_camera.json', height: 220),
            const SizedBox(height: 32),
            const Text(
              'Menginisialisasi kamera...',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.black87,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
