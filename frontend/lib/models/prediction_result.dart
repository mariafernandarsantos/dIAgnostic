class PredictionResult {
  final String id;
  final String diagnosis;
  final String result;
  final double? confidence;
  final double? probability;
  final String? explanation;
  final String? filename;
  final DateTime timestamp;
  final bool doctorReviewed;
  final bool doctorConfirmed;
  final String? doctorNotes;

  PredictionResult({
    required this.id,
    required this.diagnosis,
    required this.result,
    this.confidence,
    this.probability,
    this.explanation,
    this.filename,
    required this.timestamp,
    this.doctorReviewed = false,
    this.doctorConfirmed = false,
    this.doctorNotes,
  });

  factory PredictionResult.fromJson(Map<String, dynamic> json) {
    return PredictionResult(
      id: json['id'] ?? '-',
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
      doctorConfirmed: json['confirmed_by_doctor'] ?? false,
      doctorNotes: json['doctor_notes'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'diagnosis': diagnosis,
      'result': result,
      'confidence': confidence,
      'probability': probability,
      'explanation': explanation,
      'filename': filename,
      'timestamp': timestamp.toIso8601String(),
      'doctor_reviewed': doctorReviewed,
      'confirmed_by_doctor': doctorConfirmed,
      'doctor_notes': doctorNotes,
    };
  }
}