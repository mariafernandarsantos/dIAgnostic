import 'package:flutter/material.dart';
import '../../widgets/common/custom_app_bar.dart';
import 'widgets/user_info_card.dart';
import 'widgets/alert_card.dart';
import 'widgets/diagnostic_card.dart';
import '../diagnostic/upload_screen.dart';
import '../history/history_screen.dart';

class HomeScreen extends StatelessWidget {
  final String userName;
  final String token;

  const HomeScreen({
    Key? key,
    required this.userName,
    required this.token,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF6ddbd7),
      appBar: CustomAppBar(
        title: 'dIAgnostic',
        showActions: true,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              UserInfoCard(
                userName: userName,
                onProfilePressed: () {
                  // Implementar perfil
                },
                onHistoryPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => HistoryScreen(userName: userName),
                    ),
                  );
                },
              ),
              const SizedBox(height: 20),
              
              const AlertCard(
                title: 'Alerta de Dengue',
                message: 'Os casos de dengue estão aumentando na região. Elimine qualquer foco de água parada, mantenha caixas d\'água bem tampadas e use repelente diariamente. Cuide da sua saúde e ajude a prevenir a doença!',
                color: Colors.yellow,
              ),
              const SizedBox(height: 20),
              
              DiagnosticCard(
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => UploadScreen(token: token),
                    ),
                  );
                },
              ),
              const SizedBox(height: 20),
              
              // Imagens informativas
              ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.asset(
                  "assets/dengue.png",
                  width: 400,
                  height: 140,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) {
                    return Container(
                      width: 400,
                      height: 140,
                      color: Colors.grey[300],
                      child: const Icon(Icons.image_not_supported),
                    );
                  },
                ),
              ),
              const SizedBox(height: 16),
              
              ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.asset(
                  "assets/dengue.png",
                  width: 400,
                  height: 140,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) {
                    return Container(
                      width: 400,
                      height: 140,
                      color: Colors.grey[300],
                      child: const Icon(Icons.image_not_supported),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}