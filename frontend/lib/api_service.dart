import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'dart:io';
import 'package:http_parser/http_parser.dart'; 

class ApiService {
  static final String? _baseUrl = dotenv.env['API_BASE_URL'];

  static Future<Map<String, dynamic>> login(String email, String password) async {
    final response = await http.post(
      Uri.parse('$_baseUrl/auth/login'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'email': email,
        'password': password,
      }),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Falha no login: ${response.body}');
    }
  }

  static Future<http.Response> register(String nome, String email, String senha) async {
    final url = Uri.parse('$_baseUrl/auth/register');

    final response = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'name': nome,
        'email': email,
        'password': senha,
      }),
    );

    return response;
  }

  // Predição de Pneumonia 
  static Future<Map<String, dynamic>> predictPneumonia({
    required File imageFile,
    required String token,
  }) async {
    final uri = Uri.parse('$_baseUrl/predict/pneumonia?get_explanation=true');

    String mimeType = imageFile.path.endsWith('.png') ? 'png' : 'jpeg';

    var request = http.MultipartRequest('POST', uri)
      ..headers['Authorization'] = 'Bearer $token'
      ..files.add(
        await http.MultipartFile.fromPath(
          'file',
          imageFile.path,
          contentType: MediaType('image', mimeType),
        ),
      );

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Erro na predição: ${response.body}');
    }
  }

  // Predição de Diabetes
  static Future<Map<String, dynamic>> predictDiabetes({
  required String token,
  required int pregnancies,
  required int glucose,
  required int bloodPressure,
  required int skinThickness,
  required int insulin,
  required double bmi,
  required double diabetesPedigree,
  required int age,
  bool getExplanation = true, // Sempre true
  }) async {
    final url = Uri.parse('$_baseUrl/predict/diabetes');

    final body = {
      "pregnancies": pregnancies,
      "glucose": glucose,
      "blood_pressure": bloodPressure,
      "skin_thickness": skinThickness,
      "insulin": insulin,
      "bmi": bmi,
      "diabetes_pedigree": diabetesPedigree,
      "age": age,
      "get_explanation": getExplanation
    };

    final response = await http.post(
      url,
      headers: {
        'Authorization': 'Bearer $token',
        'Content-Type': 'application/json',
      },
      body: jsonEncode(body),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Erro na predição de diabetes: ${response.body}');
    }
  }
}