import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:permission_handler/permission_handler.dart';
import '../utils/pose_utils.dart';

class PoseDetectorView extends StatefulWidget {
  final CameraDescription camera;

  const PoseDetectorView({super.key, required this.camera});

  @override
  State<PoseDetectorView> createState() => _PoseDetectorViewState();
}

class _PoseDetectorViewState extends State<PoseDetectorView> {
  CameraController? _cameraController;
  Future<void>? _initializeControllerFuture;
  PoseDetector? _poseDetector;
  bool _canProcess = false;
  bool _isBusy = false;
  String? _text;
  List<CameraDescription>? cameras;
  int _cameraIndex = 0;

  Map<PoseLandmarkType, List<PoseLandmark>> _landmarkHistory = {};
  final int _historySize = 5;
  final double _confidenceThreshold = 0.7;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _initializePoseDetector();
  }

  Future<void> _requestCameraPermission() async {
    final status = await Permission.camera.request();
    if (status != PermissionStatus.granted) {
      debugPrint('Camera permission denied');
    }
  }

  Future<void> _initializeCamera() async {
    await _requestCameraPermission();

    try {
      cameras ??= await availableCameras();
      if (cameras == null || cameras!.isEmpty) {
        debugPrint('No cameras available');
        return;
      }

      // controller lokal untuk menghindari race
      final controller = CameraController(
        cameras![_cameraIndex],
        ResolutionPreset.high,
        enableAudio: false,
      );

      _initializeControllerFuture = controller.initialize();
      await _initializeControllerFuture;

      if (!mounted) {
        await controller.dispose();
        return;
      }

      if (!controller.value.isStreamingImages) {
        await controller.startImageStream(_processCameraImage);
      }

      setState(() {
        _cameraController = controller;
      });
    } catch (e) {
      debugPrint('Error initializing camera: $e');
    }
  }

  Future<void> _switchCamera() async {
    if (cameras == null || cameras!.isEmpty) return;

    _cameraIndex = (_cameraIndex + 1) % cameras!.length;

    final old = _cameraController;
    _cameraController = null;
    setState(() {
      _text = null;
    });

    try {
      if (old != null) {
        if (old.value.isStreamingImages) {
          await old.stopImageStream();
        }
        await old.dispose();
      }
    } catch (e) {
      debugPrint('Error disposing old camera: $e');
    }

    await _initializeCamera();
  }

  void _initializePoseDetector() async {
    final options = PoseDetectorOptions(
      model: PoseDetectionModel.accurate,
      mode: PoseDetectionMode.stream,
    );
    _poseDetector = PoseDetector(options: options);
    await Future.delayed(const Duration(milliseconds: 300));
    _canProcess = true;
  }

  bool _isControllerReadySafely() {
    final c = _cameraController;
    if (c == null) return false;
    try {
      return c.value.isInitialized;
    } catch (_) {
      return false;
    }
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (!_canProcess || _isBusy || _cameraController == null) return;
    _isBusy = true;

    final int imageRotation = cameras![_cameraIndex].sensorOrientation;
    final InputImageRotation rotation = InputImageRotation.values.firstWhere(
          (element) => element.rawValue == imageRotation,
      orElse: () => InputImageRotation.rotation0deg,
    );

    try {
      final InputImage inputImage;
      if (Platform.isAndroid) {
        final WriteBuffer allBytes = WriteBuffer();
        for (Plane plane in image.planes) {
          allBytes.putUint8List(plane.bytes);
        }
        final bytes = allBytes.done().buffer.asUint8List();

        inputImage = InputImage.fromBytes(
          bytes: bytes,
          metadata: InputImageMetadata(
            size: Size(image.width.toDouble(), image.height.toDouble()),
            rotation: rotation,
            format: InputImageFormat.yuv420,
            bytesPerRow: image.planes[0].bytesPerRow,
          ),
        );
      } else {
        inputImage = InputImage.fromBytes(
          bytes: image.planes[0].bytes,
          metadata: InputImageMetadata(
            size: Size(image.width.toDouble(), image.height.toDouble()),
            rotation: rotation,
            format: InputImageFormat.bgra8888,
            bytesPerRow: image.planes[0].bytesPerRow,
          ),
        );
      }

      final poses = await _poseDetector?.processImage(inputImage);
      final imageSize = Size(image.width.toDouble(), image.height.toDouble());

      if (poses != null && poses.isNotEmpty) {
        final pose = poses.first;
        final Map<PoseLandmarkType, PoseLandmark> filteredLandmarks = {};

        pose.landmarks.forEach((type, landmark) {
          if (landmark.likelihood >= _confidenceThreshold) {
            filteredLandmarks[type] = landmark;
          }
        });

        final filteredPose = Pose(landmarks: filteredLandmarks);
        final smoothedPose = PoseUtils.applySmoothing(
          filteredPose,
          _landmarkHistory,
          _historySize,
        );

        if (mounted) {
          if (filteredLandmarks.length >= 4) {
            final statusList = PoseUtils.analyzePose(smoothedPose, imageSize);
            setState(() {
              _text = statusList.join('\n');
            });
          } else {
            setState(() {
              _text = 'Bergeraklah agar terdeteksi lebih baik';
            });
          }
        }
      } else {
        if (mounted) {
          setState(() {
            _text = 'Tidak ada pose terdeteksi';
          });
        }
      }
    } catch (e) {
      debugPrint('Error pada deteksi pose: $e');
      if (mounted) {
        final s = e.toString();
        setState(() {
          _text = 'Error: ${s.substring(0, s.length > 50 ? 50 : s.length)}';
        });
      }
    } finally {
      _isBusy = false;
    }
  }

  @override
  void dispose() {
    _canProcess = false;

    try {
      _poseDetector?.close();
    } catch (_) {}

    final c = _cameraController;
    _cameraController = null;

    if (c != null) {
      try {
        if (c.value.isStreamingImages) {
          c.stopImageStream();
        }
      } catch (_) {}
      try {
        c.dispose();
      } catch (_) {}
    }

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isControllerReadySafely()) {
      return const Scaffold(
        backgroundColor: Colors.white,
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text(
                'Menginisialisasi kamera...',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.black,
        actions: [
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 16.0),
            child: IconButton(
              icon: const Icon(Icons.flip_camera_ios),
              iconSize: 40,
              onPressed: _switchCamera,
              tooltip: 'Ganti kamera',
            ),
          ),
        ],
      ),
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: <Widget>[
          Align(
            alignment: Alignment.center,
            child: AspectRatio(
              aspectRatio: 9 / 16,
              child: CameraPreview(_cameraController!),
            ),
          ),
          if (_text != null)
            Align(
              alignment: Alignment.bottomCenter,
              child: Padding(
                padding: const EdgeInsets.only(bottom: 60, left: 20, right: 20),
                child: Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.7),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.white, width: 2),
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: _text!.split('\n').map((line) {
                      final parts = line.split(': ');
                      if (parts.length == 2) {
                        final label = parts[0];
                        final value = parts[1];
                        Color valueColor = Colors.white;

                        if (value.contains('normal') ||
                            value == 'Berdiri' ||
                            value == 'Kedua siku lurus') {
                          valueColor = Colors.green;
                        } else {
                          valueColor = Colors.orange;
                        }

                        return Padding(
                          padding: const EdgeInsets.only(bottom: 6),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(
                                label,
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              Text(
                                value,
                                style: TextStyle(
                                  color: valueColor,
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                        );
                      } else {
                        return Text(
                          line,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 16,
                          ),
                        );
                      }
                    }).toList(),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
