import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import '../constants/app_endpoints.dart';

class ApiService {
  static final String? _baseUrl = dotenv.env['API_BASE_URL'];

  static Future<Map<String, dynamic>> login(String email, String password) async {
    final response = await http.post(
      Uri.parse('$_baseUrl${ApiEndpoints.login}'),
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
    final url = Uri.parse('$_baseUrl${ApiEndpoints.register}');

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

  static Future<Map<String, dynamic>> predictPneumonia({
    required File imageFile,
    required String token,
  }) async {
    final uri = Uri.parse('$_baseUrl${ApiEndpoints.predictPneumonia}?get_explanation=true');

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
    bool getExplanation = true,
  }) async {
    final url = Uri.parse('$_baseUrl${ApiEndpoints.predictDiabetes}');

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

  static Future<List<Map<String, dynamic>>> getPredictionHistory({
    required String token,
    String? predictionType, // 'pneumonia', 'diabetes' ou null
    int limit = 20,
  }) async {
    final queryParams = {
      if (predictionType != null) 'prediction_type': predictionType,
      'limit': limit.toString(),
    };

    final uri = Uri.parse('$_baseUrl${ApiEndpoints.predictHistory}')
        .replace(queryParameters: queryParams);

    final response = await http.get(
      uri,
      headers: {
        'Authorization': 'Bearer $token',
        'accept': 'application/json',
      },
    );

    if (response.statusCode == 200) {
      final List<dynamic> data = jsonDecode(response.body);
      return data.cast<Map<String, dynamic>>();
    } else {
      throw Exception('Erro ao buscar histórico: ${response.body}');
    }
  }
}