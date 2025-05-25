import 'package:flutter/material.dart';
import '../../core/constants/app_colors.dart';

class CustomAppBar extends StatelessWidget implements PreferredSizeWidget {
  final String title;
  final List<Widget>? actions;  // Add this line
  final bool showActions;
  final VoidCallback? onNotificationPressed;
  final VoidCallback? onMessagePressed;
  final bool showBackButton;

  const CustomAppBar({
    super.key,
    required this.title,
    this.actions,  // Add this parameter
    this.showActions = true,
    this.onNotificationPressed,
    this.onMessagePressed,
    this.showBackButton = false,
  });

  @override
  Widget build(BuildContext context) {
    return AppBar(
      backgroundColor: AppColors.primary,
      elevation: 0,
      automaticallyImplyLeading: showBackButton,
      iconTheme: const IconThemeData(color: Colors.black),
      title: Text.rich(
        TextSpan(
          children: [
            const TextSpan(
              text: 'dIA',
              style: TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 26,
                color: Colors.white,
              ),
            ),
            const TextSpan(
              text: 'gnostic',
              style: TextStyle(fontSize: 26, color: Colors.white),
            ),
          ],
        ),
      ),
      actions: actions ?? (showActions  // Modified this line
          ? [
              IconButton(
                icon: const Icon(Icons.notifications_none, size: 24, color: Colors.black),
                onPressed: onNotificationPressed ?? () => print('Notificação clicada'),
              ),
              IconButton(
                icon: const Icon(Icons.mail_outline, size: 24, color: Colors.black),
                onPressed: onMessagePressed ?? () => print('Mensagem clicada'),
              ),
              const SizedBox(width: 16),
            ]
          : null),
    );
  }

  @override
  Size get preferredSize => const Size.fromHeight(kToolbarHeight);
}