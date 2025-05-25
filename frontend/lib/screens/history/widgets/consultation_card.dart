import 'package:flutter/material.dart';
import '../../../../models/prediction_result.dart';
import '../../../../core/constants/app_colors.dart';

class ConsultationCard extends StatelessWidget {
  final PredictionResult consultation;
  final VoidCallback? onTap;
  final VoidCallback? onDelete;

  const ConsultationCard({
    super.key,
    required this.consultation,
    this.onTap,
    this.onDelete,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(12),
        child: Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(12),
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Colors.white,
                Colors.grey.shade50,
              ],
            ),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header com tipo e data
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 4,
                        ),
                        decoration: BoxDecoration(
                          color: _getTypeColor(consultation.diagnosis),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text(
                          consultation.diagnosis,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Icon(
                        _getTypeIcon(consultation.diagnosis),
                        size: 20,
                        color: _getTypeColor(consultation.diagnosis),
                      ),
                    ],
                  ),
                  Row(
                    children: [
                      Text(
                        consultation.timestamp.toLocal().toString().split(' ')[0],
                        style: TextStyle(
                          color: Colors.grey[600],
                          fontSize: 12,
                        ),
                      ),
                      if (onDelete != null) ...[
                        const SizedBox(width: 8),
                        InkWell(
                          onTap: onDelete,
                          borderRadius: BorderRadius.circular(16),
                          child: Container(
                            padding: const EdgeInsets.all(4),
                            child: Icon(
                              Icons.delete_outline,
                              size: 16,
                              color: Colors.red[400],
                            ),
                          ),
                        ),
                      ],
                    ],
                  ),
                ],
              ),
              
              const SizedBox(height: 12),
              
              // Diagnóstico
              Row(
                children: [
                  Icon(
                    _getDiagnosisIcon(consultation.diagnosis),
                    size: 16,
                    color: _getDiagnosisColor(consultation.diagnosis),
                  ),
                  const SizedBox(width: 6),
                  const Text(
                    'Diagnóstico:',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                      color: Colors.black87,
                    ),
                  ),
                ],
              ),
              
              const SizedBox(height: 4),
              
              Text(
                consultation.diagnosis,
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  color: _getDiagnosisColor(consultation.diagnosis),
                ),
              ),
              
              // Confiança/Probabilidade
              if (consultation.confidence != null) ...[
                const SizedBox(height: 8),
                Row(
                  children: [
                    Icon(
                      Icons.analytics_outlined,
                      size: 14,
                      color: Colors.grey[600],
                    ),
                    const SizedBox(width: 4),
                    Text(
                      'Confiança: ${(consultation.confidence! * 100).toStringAsFixed(1)}%',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey[600],
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: LinearProgressIndicator(
                        value: consultation.confidence,
                        backgroundColor: Colors.grey[300],
                        valueColor: AlwaysStoppedAnimation<Color>(
                          _getConfidenceColor(consultation.confidence!),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
              
              // Explicação prévia (se houver)
              if (consultation.explanation != null && 
                  consultation.explanation!.isNotEmpty) ...[
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.blue.shade50,
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(
                      color: Colors.blue.shade200,
                      width: 1,
                    ),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        Icons.lightbulb_outline,
                        size: 14,
                        color: Colors.blue[700],
                      ),
                      const SizedBox(width: 6),
                      Expanded(
                        child: Text(
                          consultation.explanation!.length > 100
                              ? '${consultation.explanation!.substring(0, 100)}...'
                              : consultation.explanation!,
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.blue[700],
                          ),
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
              
              const SizedBox(height: 8),
              
              // Footer com ação
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  TextButton.icon(
                    onPressed: onTap,
                    icon: const Icon(
                      Icons.visibility_outlined,
                      size: 16,
                    ),
                    label: const Text(
                      'Ver detalhes',
                      style: TextStyle(fontSize: 12),
                    ),
                    style: TextButton.styleFrom(
                      foregroundColor: AppColors.primary,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Color _getTypeColor(String type) {
    switch (type.toLowerCase()) {
      case 'pneumonia':
        return Colors.blue[600]!;
      case 'diabetes':
        return Colors.orange[600]!;
      default:
        return Colors.grey[600]!;
    }
  }

  IconData _getTypeIcon(String type) {
    switch (type.toLowerCase()) {
      case 'pneumonia':
        return Icons.air;
      case 'diabetes':
        return Icons.water_drop;
      default:
        return Icons.medical_services;
    }
  }

  Color _getDiagnosisColor(String diagnosis) {
    final lowerDiagnosis = diagnosis.toLowerCase();
    if (lowerDiagnosis.contains('positiv') || 
        lowerDiagnosis.contains('detectad')) {
      return Colors.red[600]!;
    } else if (lowerDiagnosis.contains('negativ') || 
               lowerDiagnosis.contains('normal')) {
      return Colors.green[600]!;
    }
    return Colors.orange[600]!;
  }

  IconData _getDiagnosisIcon(String diagnosis) {
    final lowerDiagnosis = diagnosis.toLowerCase();
    if (lowerDiagnosis.contains('positiv') || 
        lowerDiagnosis.contains('detectad')) {
      return Icons.warning_rounded;
    } else if (lowerDiagnosis.contains('negativ') || 
               lowerDiagnosis.contains('normal')) {
      return Icons.check_circle_rounded;
    }
    return Icons.info_rounded;
  }

  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.8) {
      return Colors.green;
    } else if (confidence >= 0.6) {
      return Colors.orange;
    } else {
      return Colors.red;
    }
  }
}