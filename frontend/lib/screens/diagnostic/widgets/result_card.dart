import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import '../../../core/constants/app_colors.dart';

class ResultCard extends StatelessWidget {
  final Map<String, dynamic> result;
  final VoidCallback? onSave;
  final VoidCallback? onShare;

  const ResultCard({
    super.key,
    required this.result,
    this.onSave,
    this.onShare,
  });

  @override
  Widget build(BuildContext context) {
    final String diagnosis = result['diagnosis'] ?? 'Não identificado';
    final double? confidence = result['confidence']?.toDouble();
    final double? probability = result['probability']?.toDouble();
    final String? explanation = result['explanation'];
    final String? filename = result['filename'];
    
    // Determinar cor do resultado baseado no diagnóstico
    Color resultColor = _getResultColor(diagnosis);
    IconData resultIcon = _getResultIcon(diagnosis);

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 8),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.1),
            spreadRadius: 1,
            blurRadius: 4,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header do resultado
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: resultColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Icon(
                  resultIcon,
                  color: resultColor,
                  size: 24,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Resultado da Análise',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: AppColors.primary,
                      ),
                    ),
                    Text(
                      DateTime.now().toString().substring(0, 19),
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey[600],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          
          const SizedBox(height: 16),
          
          // Diagnóstico principal
          Container(
            width: 400,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: resultColor.withOpacity(0.05),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: resultColor.withOpacity(0.2)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Diagnóstico:',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: AppColors.secondary,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  diagnosis,
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: resultColor,
                  ),
                ),
              ],
            ),
          ),
          
          const SizedBox(height: 16),
          
          // Informações adicionais
          if (filename != null) ...[
            _buildInfoRow('Arquivo:', filename),
            const SizedBox(height: 8),
          ],
          
          if (confidence != null) ...[
            _buildInfoRow(
              'Confiança:', 
              '${(confidence * 100).toStringAsFixed(1)}%'
            ),
            const SizedBox(height: 8),
          ],
          
          if (probability != null) ...[
            _buildInfoRow(
              'Probabilidade:', 
              '${(probability * 100).toStringAsFixed(1)}%'
            ),
            const SizedBox(height: 8),
          ],
          
          // Explicação da IA
          if (explanation != null) ...[
            const SizedBox(height: 16),
            const Text(
              'Explicação da IA:',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: AppColors.primary,
              ),
            ),
            const SizedBox(height: 8),
            Container(
              width: 400,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.grey[50],
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.grey[200]!),
              ),
              child: MarkdownBody(
                data: explanation,
                styleSheet: MarkdownStyleSheet(
                  p: const TextStyle(fontSize: 14, color: Colors.black87),
                  h2: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: AppColors.primary,
                  ),
                  listBullet: const TextStyle(fontSize: 14),
                  strong: const TextStyle(fontWeight: FontWeight.bold),
                ),
              ),
            ),
          ],
          
          const SizedBox(height: 20),
          
          // Botões de ação
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: onSave,
                  icon: const Icon(Icons.bookmark_outline),
                  label: const Text('Salvar'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: AppColors.primary,
                    side: const BorderSide(color: AppColors.primary),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: onShare,
                  icon: const Icon(Icons.share_outlined),
                  label: const Text('Compartilhar'),
                  style: OutlinedButton.styleFrom(
                    foregroundColor: AppColors.primary,
                    side: const BorderSide(color: AppColors.primary),
                  ),
                ),
              ),
            ],
          ),
          
          // Aviso médico
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.amber[50],
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.amber[200]!),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(
                  Icons.warning_amber_outlined,
                  color: Colors.amber[700],
                  size: 20,
                ),
                const SizedBox(width: 8),
                const Expanded(
                  child: Text(
                    'Este resultado é apenas uma análise auxiliar. Sempre consulte um médico para diagnóstico definitivo.',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.black87,
                      height: 1.3,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildInfoRow(String label, String value) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SizedBox(
          width: 100,
          child: Text(
            label,
            style: const TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: AppColors.secondary,
            ),
          ),
        ),
        Expanded(
          child: Text(
            value,
            style: const TextStyle(
              fontSize: 14,
              color: AppColors.primary,
            ),
          ),
        ),
      ],
    );
  }
  
  Color _getResultColor(String diagnosis) {
    final String lowerDiagnosis = diagnosis.toLowerCase();
    
    if (lowerDiagnosis.contains('pneumonia') || 
        lowerDiagnosis.contains('positive') ||
        lowerDiagnosis.contains('diabetes')) {
      return Colors.red;
    } else if (lowerDiagnosis.contains('normal') || 
               lowerDiagnosis.contains('negative') ||
               lowerDiagnosis.contains('não')) {
      return Colors.green;
    } else {
      return Colors.orange;
    }
  }
  
  IconData _getResultIcon(String diagnosis) {
    final String lowerDiagnosis = diagnosis.toLowerCase();
    
    if (lowerDiagnosis.contains('pneumonia') || 
        lowerDiagnosis.contains('positive') ||
        lowerDiagnosis.contains('diabetes')) {
      return Icons.warning;
    } else if (lowerDiagnosis.contains('normal') || 
               lowerDiagnosis.contains('negative') ||
               lowerDiagnosis.contains('não')) {
      return Icons.check_circle;
    } else {
      return Icons.help;
    }
  }
}