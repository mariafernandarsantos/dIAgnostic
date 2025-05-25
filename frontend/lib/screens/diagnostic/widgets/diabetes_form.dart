import 'package:flutter/material.dart';
import '../../../core/services/api_service.dart';
import '../../../core/constants/app_colors.dart';
import '../../../widgets/forms/number_field.dart';

class DiabetesForm extends StatefulWidget {
  final String token;
  final Function(Map<String, dynamic>) onResult;
  final VoidCallback? onLoading;

  const DiabetesForm({
    super.key,
    required this.token,
    required this.onResult,
    this.onLoading,
  });

  @override
  State<DiabetesForm> createState() => _DiabetesFormState();
}

class _DiabetesFormState extends State<DiabetesForm> {
  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false;

  // Controllers
  final TextEditingController _pregnanciesController = TextEditingController();
  final TextEditingController _glucoseController = TextEditingController();
  final TextEditingController _bloodPressureController = TextEditingController();
  final TextEditingController _skinThicknessController = TextEditingController();
  final TextEditingController _insulinController = TextEditingController();
  final TextEditingController _bmiController = TextEditingController();
  final TextEditingController _diabetesPedigreeController = TextEditingController();
  final TextEditingController _ageController = TextEditingController();

  @override
  void dispose() {
    _pregnanciesController.dispose();
    _glucoseController.dispose();
    _bloodPressureController.dispose();
    _skinThicknessController.dispose();
    _insulinController.dispose();
    _bmiController.dispose();
    _diabetesPedigreeController.dispose();
    _ageController.dispose();
    super.dispose();
  }

  Future<void> _submitForm() async {
    if (!_formKey.currentState!.validate()) return;

    if (!mounted) return; // Add mounted check

    setState(() {
      _isLoading = true;
    });

    widget.onLoading?.call();

    try {
      final response = await ApiService.predictDiabetes(
        token: widget.token,
        pregnancies: int.parse(_pregnanciesController.text),
        glucose: int.parse(_glucoseController.text),
        bloodPressure: int.parse(_bloodPressureController.text),
        skinThickness: int.parse(_skinThicknessController.text),
        insulin: int.parse(_insulinController.text),
        bmi: double.parse(_bmiController.text),
        diabetesPedigree: double.parse(_diabetesPedigreeController.text),
        age: int.parse(_ageController.text),
      );

      // Always call onResult, even if widget is unmounted
      // The parent needs this to clear its loading state
      widget.onResult(response);

      if (mounted) {
        _clearForm();
        _showSuccessMessage('Dados analisados com sucesso!');
      }
    } catch (e) {
      if (mounted) {
        _showErrorMessage('Erro na análise: $e');
      }
    } finally {
      // Always check mounted before setState
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  void _clearForm() {
    _pregnanciesController.clear();
    _glucoseController.clear();
    _bloodPressureController.clear();
    _skinThicknessController.clear();
    _insulinController.clear();
    _bmiController.clear();
    _diabetesPedigreeController.clear();
    _ageController.clear();
  }

  void _showErrorMessage(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
      ),
    );
  }

  void _showSuccessMessage(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.green,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
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
      child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Análise de Diabetes',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: AppColors.primary,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            const Text(
              'Preencha os dados abaixo para análise de risco de diabetes',
              style: TextStyle(
                fontSize: 14,
                color: AppColors.secondary,
              ),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),

            // Campos do formulário
            NumberField(
              controller: _pregnanciesController,
              label: 'Número de Gestações',
              isRequired: true,
            ),
            const SizedBox(height: 16),

            NumberField(
              controller: _glucoseController,
              label: 'Glicose (mg/dL)',
              isRequired: true,
            ),
            const SizedBox(height: 16),

            NumberField(
              controller: _bloodPressureController,
              label: 'Pressão Arterial (mmHg)',
              isRequired: true,
            ),
            const SizedBox(height: 16),

            NumberField(
              controller: _skinThicknessController,
              label: 'Espessura da Pele (mm)',
              isRequired: true,
            ),
            const SizedBox(height: 16),

            NumberField(
              controller: _insulinController,
              label: 'Insulina (μU/mL)',
              isRequired: true,
            ),
            const SizedBox(height: 16),

            NumberField(
              controller: _bmiController,
              label: 'IMC (Índice de Massa Corporal)',
              isRequired: true,
              isDecimal: true,
            ),
            const SizedBox(height: 16),

            NumberField(
              controller: _diabetesPedigreeController,
              label: 'Histórico Familiar de Diabetes',
              isRequired: true,
              isDecimal: true,
            ),
            const SizedBox(height: 16),

            NumberField(
              controller: _ageController,
              label: 'Idade (anos)',
              isRequired: true,
            ),
            const SizedBox(height: 24),

            // Botão de análise
            ElevatedButton(
              onPressed: _isLoading ? null : _submitForm,
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.primary,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              child: _isLoading
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(
                        strokeWidth: 2,
                        valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                      ),
                    )
                  : const Text(
                      'Analisar Dados',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
            ),
            const SizedBox(height: 16),

            // Botão limpar
            OutlinedButton(
              onPressed: _isLoading ? null : _clearForm,
              style: OutlinedButton.styleFrom(
                foregroundColor: AppColors.primary,
                side: const BorderSide(color: AppColors.primary),
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              child: const Text(
                'Limpar Formulário',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}