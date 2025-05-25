import 'package:flutter/material.dart';
import '../../../widgets/common/custom_button.dart';

class UserInfoCard extends StatelessWidget {
  final String userName;
  final VoidCallback onProfilePressed;
  final VoidCallback onHistoryPressed;

  const UserInfoCard({
    Key? key,
    required this.userName,
    required this.onProfilePressed,
    required this.onHistoryPressed,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF3D4D64),
        borderRadius: BorderRadius.circular(15),
      ),
      child: Column(
        children: [
          Row(
            children: [
              const CircleAvatar(
                backgroundColor: Colors.white,
                child: Icon(Icons.person, color: Colors.grey),
              ),
              const SizedBox(width: 12),
              Expanded(
                flex: 3,
                child: Text(
                  userName,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              Expanded(
                flex: 2,
                child: CustomButton(
                  text: 'Abrir perfil',
                  onPressed: onProfilePressed,
                  textColor: Colors.white,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          SizedBox(
            width: 400,
            child: CustomButton(
              text: 'Histórico de Consultas',
              onPressed: onHistoryPressed,
              backgroundColor: const Color(0xFF75A7BD),
              textColor: Colors.white,
            ),
          ),
        ],
      ),
    );
  }
}
