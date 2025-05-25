class UserModel {
  final String id;
  final String name;
  final String email;
  final String? phone;
  final String? gender;
  final DateTime? birthDate;

  UserModel({
    required this.id,
    required this.name,
    required this.email,
    this.phone,
    this.gender,
    this.birthDate,
  });

  factory UserModel.fromJson(Map<String, dynamic> json) {
    return UserModel(
      id: json['id'] ?? '',
      name: json['name'] ?? '',
      email: json['email'] ?? '',
      phone: json['phone'],
      gender: json['gender'],
      birthDate: json['birth_date'] != null 
          ? DateTime.parse(json['birth_date']) 
          : null,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'email': email,
      'phone': phone,
      'gender': gender,
      'birth_date': birthDate?.toIso8601String(),
    };
  }
}