class PredictionResult {
  final String diagnosis;
  final String result;
  final double? confidence;
  final double? probability;
  final String? explanation;
  final String? filename;
  final DateTime timestamp;
  final bool doctorReviewed;
  final String? doctorNotes;

  PredictionResult({
    required this.diagnosis,
    required this.result,
    this.confidence,
    this.probability,
    this.explanation,
    this.filename,
    required this.timestamp,
    this.doctorReviewed = false,
    this.doctorNotes,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      diagnosis: json['type'] ?? '-',
      result: json['result'] ?? '-',
      confidence: json['confidence']?.toDouble(),
      probability: json['probability']?.toDouble(),
      explanation: json['additional_notes'],
      filename: json['filename'],
      timestamp: json['timestamp'] != null
          ? DateTime.parse(json['timestamp'])
          : DateTime.now(),
      doctorReviewed: json['doctor_reviewed'] ?? false,
      doctorNotes: json['doctor_notes'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'diagnosis': diagnosis,
      'result': result,
      'confidence': confidence,
      'probability': probability,
      'explanation': explanation,
      'filename': filename,
      'timestamp': timestamp.toIso8601String(),
      'doctor_reviewed': doctorReviewed,
      'doctor_notes': doctorNotes,
    };
  }
}