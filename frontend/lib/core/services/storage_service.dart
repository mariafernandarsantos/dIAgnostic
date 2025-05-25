import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

class StorageService {
  static const String _consultationHistoryKey = 'consultation_history';
  static Future<void> setString(String key, String value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(key, value);
  }

  static Future<String?> getString(String key) async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(key);
  }

  static Future<void> setBool(String key, bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(key, value);
  }

  static Future<bool?> getBool(String key) async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(key);
  }

  static Future<void> remove(String key) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(key);
  }

  static Future<void> clear() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.clear();
  }

    static Future<void> addConsultation(Map<String, dynamic> consultation) async {
    final prefs = await SharedPreferences.getInstance();
    final currentHistory = await getConsultationHistory();
    
    currentHistory.add(consultation);
    await prefs.setString(_consultationHistoryKey, jsonEncode(currentHistory));
  }

  // Novo: Obter todo o histórico de consultas
  static Future<List<Map<String, dynamic>>> getConsultationHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final historyJson = prefs.getString(_consultationHistoryKey);
    
    if (historyJson == null || historyJson.isEmpty) {
      return [];
    }
    
    try {
      final List<dynamic> historyList = jsonDecode(historyJson);
      return historyList.cast<Map<String, dynamic>>();
    } catch (e) {
      print('Erro ao decodificar histórico: $e');
      return [];
    }
  }

  // Novo: Limpar o histórico de consultas
  static Future<void> clearConsultationHistory() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_consultationHistoryKey);
  }
}