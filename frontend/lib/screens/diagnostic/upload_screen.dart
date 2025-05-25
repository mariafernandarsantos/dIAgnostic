import 'package:flutter/material.dart';
import '../../widgets/common/custom_app_bar.dart';
import '../../widgets/common/loading_widget.dart';
import 'widgets/pneumonia_form.dart';
import 'widgets/diabetes_form.dart';
import 'widgets/result_card.dart';

class UploadScreen extends StatefulWidget {
  final String token;

  const UploadScreen({Key? key, required this.token}) : super(key: key);

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  bool _isDiabetes = false;
  bool _isLoading = false;
  Map<String, dynamic>? _result;

  void _onDiagnosisTypeChanged(bool isDiabetes) {
    setState(() {
      _isDiabetes = isDiabetes;
      _result = null;
    });
  }

  void _onResultReceived(Map<String, dynamic> result) {
    setState(() {
      _result = result;
      _isLoading = false;
    });
  }

  void _onLoadingChanged(bool isLoading) {
    setState(() {
      _isLoading = isLoading;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF6ddbd7),
      appBar: const CustomAppBar(
        title: 'dIAgnostic',
        showBackButton: true,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            children: [
              // Info Card
              Container(
                decoration: BoxDecoration(
                  color: const Color(0xFFFFD580),
                  borderRadius: BorderRadius.circular(12),
                ),
                padding: const EdgeInsets.all(16),
                child: const Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    CircleAvatar(
                      backgroundColor: Color(0xFF20C4D1),
                      radius: 16,
                      child: Icon(Icons.info, color: Colors.white, size: 20),
                    ),
                    SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'O dIAgnostic é um app com intuito de auxiliar o médico durante o processo de análise de exames clínicos.',
                        style: TextStyle(fontSize: 14),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              // Main Content
              Container(
                decoration: BoxDecoration(
                  color: const Color(0xFF244156),
                  borderRadius: BorderRadius.circular(12),
                ),
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Resultados do exame',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 16),

                    // Type Selection
                    Row(
                      children: [
                        ChoiceChip(
                          label: const Text('Pneumonia'),
                          selected: !_isDiabetes,
                          onSelected: (_) => _onDiagnosisTypeChanged(false),
                          selectedColor: const Color(0xFF75A7BD),
                          labelStyle: TextStyle(
                            color: !_isDiabetes ? Colors.white : Colors.black,
                          ),
                        ),
                        const SizedBox(width: 12),
                        ChoiceChip(
                          label: const Text('Diabetes'),
                          selected: _isDiabetes,
                          onSelected: (_) => _onDiagnosisTypeChanged(true),
                          selectedColor: const Color(0xFF75A7BD),
                          labelStyle: TextStyle(
                            color: _isDiabetes ? Colors.white : Colors.black,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 20),

                    // Form Content
                    if (_isLoading)
                      const Center(child: LoadingWidget())
                    else if (_isDiabetes)
                      DiabetesForm(
                        token: widget.token,
                        onResult: _onResultReceived,
                        onLoading: () => _onLoadingChanged(true),
                      )
                    else
                      PneumoniaForm(
                        token: widget.token,
                        onResult: _onResultReceived,
                        onLoading: () => _onLoadingChanged(true),
                      ),

                    // Result Display
                    if (_result != null) ...[
                      const SizedBox(height: 20),
                      ResultCard(result: _result!),
                    ],
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}