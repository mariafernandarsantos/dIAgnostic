class ApiEndpoints {
  static const String login = '/auth/login';
  static const String register = '/auth/register';
  static const String predictPneumonia = '/predict/pneumonia';
  static const String predictDiabetes = '/predict/diabetes';
  static const String predictHistory = '/predict/history';
    static String predictReview(String predictionId) => '/predict/$predictionId/review';
  static const String getUser = '/user';
}