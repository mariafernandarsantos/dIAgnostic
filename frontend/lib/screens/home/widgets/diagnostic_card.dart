import 'package:flutter/material.dart';

class DiagnosticCard extends StatelessWidget {
  final VoidCallback onTap;

  const DiagnosticCard({
    Key? key,
    required this.onTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(15),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: const Color(0xFF3D4D64),
          borderRadius: BorderRadius.circular(15),
        ),
        child: const Row(
          children: [
            Icon(
              Icons.description_outlined,
              color: Colors.white,
              size: 28,
            ),
            SizedBox(width: 12),
            Text(
              'dIAgnostic',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.white,
              ),
            ),
            Spacer(),
            Icon(
              Icons.arrow_forward_ios,
              color: Colors.white,
              size: 20,
            ),
          ],
        ),
      ),
    );
  }
}