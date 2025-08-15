import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

/// Kelas untuk utilitas terkait pose detection
class PoseUtils {
  // List keypoints yang dibutuhkan pertiap anggota tubuh
  static const List<PoseLandmarkType> handKeypoints = [
    PoseLandmarkType.leftWrist, 
    PoseLandmarkType.rightWrist,
    PoseLandmarkType.leftShoulder,
    PoseLandmarkType.rightShoulder
  ];
  
  static const List<PoseLandmarkType> elbowKeypoints = [
    PoseLandmarkType.leftWrist,
    PoseLandmarkType.rightWrist,
    PoseLandmarkType.leftElbow,
    PoseLandmarkType.rightElbow,
    PoseLandmarkType.leftShoulder,
    PoseLandmarkType.rightShoulder
  ];
  
  static const List<PoseLandmarkType> bodyKeypoints = [
    PoseLandmarkType.leftKnee, 
    PoseLandmarkType.rightKnee,
    PoseLandmarkType.leftHip,
    PoseLandmarkType.rightHip
  ];

  static const List<PoseLandmarkType> headTiltKeypoints = [
    PoseLandmarkType.leftEar,
    PoseLandmarkType.rightEar,
  ];

  /// Memeriksa apakah semua keypoint yang diperlukan tersedia
  static bool hasKeypoints(Pose pose, List<PoseLandmarkType> types) {
    for (final type in types) {
      final landmark = pose.landmarks[type];
      if (landmark == null || landmark.likelihood < 0.5) {
        return false;
      }
    }
    return true;
  }

  /// Menghitung sudut antara tiga titik
  static double calculateAngle(PoseLandmark a, PoseLandmark b, PoseLandmark c) {
    double radians = math.atan2(c.y - b.y, c.x - b.x) -
        math.atan2(a.y - b.y, a.x - b.x);
    double angle = (radians * 180.0 / math.pi);
    if (angle < 0) {
      angle = -angle;
    }
    if (angle > 180.0) {
      angle = 360 - angle;
    }
    return angle;
  }

  /// Menganalisis pose dan mengembalikan daftar status
  static List<String> analyzePose(Pose pose, Size imageSize) {
    List<String> statusList = [];

    // Analyze head tilt
    if (hasKeypoints(pose, headTiltKeypoints)) {
      final leftEar = pose.landmarks[PoseLandmarkType.leftEar]!;
      final rightEar = pose.landmarks[PoseLandmarkType.rightEar]!;

      double earDiff = leftEar.y - rightEar.y;

      if (earDiff > 15) {
        statusList.add('Kepala: Miring ke kiri');
      } else if (earDiff < -15) {
        statusList.add('Kepala: Miring ke kanan');
      } else {
        statusList.add('Kepala: Posisi normal');
      }
    }

    // Analyze elbow angles
    if (hasKeypoints(pose, elbowKeypoints)) {
      final leftWrist = pose.landmarks[PoseLandmarkType.leftWrist]!;
      final rightWrist = pose.landmarks[PoseLandmarkType.rightWrist]!;
      final leftElbow = pose.landmarks[PoseLandmarkType.leftElbow]!;
      final rightElbow = pose.landmarks[PoseLandmarkType.rightElbow]!;
      final leftShoulder = pose.landmarks[PoseLandmarkType.leftShoulder]!;
      final rightShoulder = pose.landmarks[PoseLandmarkType.rightShoulder]!;

      double leftElbowAngle = calculateAngle(leftWrist, leftElbow, leftShoulder);
      double rightElbowAngle = calculateAngle(rightWrist, rightElbow, rightShoulder);

      // Siku dianggap ditekuk jika sudutnya kurang dari 120 derajat
      bool leftElbowBent = leftElbowAngle < 120;
      bool rightElbowBent = rightElbowAngle < 120;

      if (leftElbowBent && rightElbowBent) {
        statusList.add('Siku: Kedua siku ditekuk');
      } else if (leftElbowBent) {
        statusList.add('Siku: Siku kiri ditekuk');
      } else if (rightElbowBent) {
        statusList.add('Siku: Siku kanan ditekuk');
      } else {
        statusList.add('Siku: Kedua siku lurus');
      }
    }
    
    // Analyze hand positions
    if (hasKeypoints(pose, handKeypoints)) {
      final leftWrist = pose.landmarks[PoseLandmarkType.leftWrist]!;
      final rightWrist = pose.landmarks[PoseLandmarkType.rightWrist]!;
      final leftShoulder = pose.landmarks[PoseLandmarkType.leftShoulder]!;
      final rightShoulder = pose.landmarks[PoseLandmarkType.rightShoulder]!;
      
      bool leftHandRaised = leftWrist.y < leftShoulder.y;
      bool rightHandRaised = rightWrist.y < rightShoulder.y;
      
      if (leftHandRaised && rightHandRaised) {
        statusList.add('Gerakan: Kedua tangan terangkat');
      } else if (leftHandRaised) {
        statusList.add('Gerakan: Tangan kiri terangkat');
      } else if (rightHandRaised) {
        statusList.add('Gerakan: Tangan kanan terangkat');
      } else {
        statusList.add('Gerakan: Tangan normal');
      }
    }
    
    // Analyze body position
    if (hasKeypoints(pose, bodyKeypoints)) {
      final leftKnee = pose.landmarks[PoseLandmarkType.leftKnee]!;
      final rightKnee = pose.landmarks[PoseLandmarkType.rightKnee]!;
      final leftHip = pose.landmarks[PoseLandmarkType.leftHip]!;
      final rightHip = pose.landmarks[PoseLandmarkType.rightHip]!;
      
      // Calculate knee-hip distance ratio for better squat detection
      double leftRatio = (leftKnee.y - leftHip.y) / imageSize.height;
      double rightRatio = (rightKnee.y - rightHip.y) / imageSize.height;
      double avgRatio = (leftRatio + rightRatio) / 2;
      
      if (avgRatio > 0.15) { // Adjusted threshold for better accuracy
        statusList.add('Posisi: Berdiri');
      } else {
        // statusList.add('Posisi: Jongkok');
        statusList.add('');
      }
    }
    
    if (statusList.isEmpty) {
      statusList.add('Bergeraklah agar terdeteksi lebih baik');
    }
    
    return statusList;
  }
  
  /// Menerapkan smoothing pada pose dengan algoritma yang dioptimalkan
  static Pose applySmoothing(
    Pose currentPose, 
    Map<PoseLandmarkType, List<PoseLandmark>> landmarkHistory,
    int historySize
  ) {
    if (currentPose.landmarks.isEmpty) {
      return currentPose;
    }
    
    final Map<PoseLandmarkType, PoseLandmark> smoothedLandmarks = {};
    final Map<PoseLandmarkType, List<PoseLandmark>> history = Map.from(landmarkHistory);
    
    currentPose.landmarks.forEach((type, landmark) {
      if (!history.containsKey(type)) {
        history[type] = [];
      }
      
      history[type]!.add(landmark);
      
      if (history[type]!.length > historySize) {
        history[type]!.removeAt(0);
      }
      
      if (history[type]!.isNotEmpty) {
        // Optimized weighted average calculation
        double sumX = 0, sumY = 0, sumZ = 0, totalWeight = 0;
        final length = history[type]!.length;
        
        for (int i = 0; i < length; i++) {
          final hist = history[type]![i];
          // Exponential weight decay for smoother transitions
          final weight = math.exp(-(length - i - 1) * 0.5) * hist.likelihood;
          
          sumX += hist.x * weight;
          sumY += hist.y * weight;
          sumZ += hist.z * weight;
          totalWeight += weight;
        }
        
        if (totalWeight > 0) {
          smoothedLandmarks[type] = PoseLandmark(
            type: type,
            x: sumX / totalWeight,
            y: sumY / totalWeight,
            z: sumZ / totalWeight,
            likelihood: landmark.likelihood,
          );
        } else {
          smoothedLandmarks[type] = landmark;
        }
      } else {
        smoothedLandmarks[type] = landmark;
      }
    });
    
    return Pose(landmarks: smoothedLandmarks);
  }
} 