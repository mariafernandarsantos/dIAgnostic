class DiabetesData {
  final int pregnancies;
  final int glucose;
  final int bloodPressure;
  final int skinThickness;
  final int insulin;
  final double bmi;
  final double diabetesPedigree;
  final int age;

  DiabetesData({
    required this.pregnancies,
    required this.glucose,
    required this.bloodPressure,
    required this.skinThickness,
    required this.insulin,
    required this.bmi,
    required this.diabetesPedigree,
    required this.age,
  });

  factory DiabetesData.fromControllers(Map<String, String> data) {
    return DiabetesData(
      pregnancies: int.parse(data['pregnancies'] ?? '0'),
      glucose: int.parse(data['glucose'] ?? '0'),
      bloodPressure: int.parse(data['bloodPressure'] ?? '0'),
      skinThickness: int.parse(data['skinThickness'] ?? '0'),
      insulin: int.parse(data['insulin'] ?? '0'),
      bmi: double.parse(data['bmi'] ?? '0.0'),
      diabetesPedigree: double.parse(data['diabetesPedigree'] ?? '0.0'),
      age: int.parse(data['age'] ?? '0'),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'pregnancies': pregnancies,
      'glucose': glucose,
      'blood_pressure': bloodPressure,
      'skin_thickness': skinThickness,
      'insulin': insulin,
      'bmi': bmi,
      'diabetes_pedigree': diabetesPedigree,
      'age': age,
    };
  }
}
