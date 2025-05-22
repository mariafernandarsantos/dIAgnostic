import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'api_service.dart';

class Upload extends StatefulWidget {
  final String token;
  const Upload({required this.token, super.key});

  @override
  _Upload createState() => _Upload();
}

class _Upload extends State<Upload> {
  bool showUpload = false;
  bool isDiabetes = false;

  // Pneumonia
  File? _image;
  final ImagePicker _picker = ImagePicker();

  // Diabetes controllers
  final _formKey = GlobalKey<FormState>();
  final TextEditingController pregnanciesController = TextEditingController();
  final TextEditingController glucoseController = TextEditingController();
  final TextEditingController bloodPressureController = TextEditingController();
  final TextEditingController skinThicknessController = TextEditingController();
  final TextEditingController insulinController = TextEditingController();
  final TextEditingController bmiController = TextEditingController();
  final TextEditingController diabetesPedigreeController = TextEditingController();
  final TextEditingController ageController = TextEditingController();

  bool isLoading = false;
  Map<String, dynamic>? result;

  // Selecionar imagem
  Future<void> _pickImage() async {
    final XFile? pickedFile = await _picker.pickImage(
      source: ImageSource.gallery,
    );
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
    }
  }

  // Enviar imagem
  Future<void> _sendImage() async {
    if (_image == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Selecione uma imagem primeiro!')),
      );
      return;
    }

    setState(() {
      isLoading = true;
      result = null;
    });

    try {
      final response = await ApiService.predictPneumonia(
        imageFile: _image!,
        token: widget.token,
      );

      setState(() {
        result = response;
        showUpload = false;
        _image = null;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Imagem enviada e analisada com sucesso!')),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Erro: $e')),
      );
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  // Enviar dados diabetes
  Future<void> _sendDiabetes() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      isLoading = true;
      result = null;
    });

    try {
      final response = await ApiService.predictDiabetes(
        token: widget.token,
        pregnancies: int.parse(pregnanciesController.text),
        glucose: int.parse(glucoseController.text),
        bloodPressure: int.parse(bloodPressureController.text),
        skinThickness: int.parse(skinThicknessController.text),
        insulin: int.parse(insulinController.text),
        bmi: double.parse(bmiController.text),
        diabetesPedigree: double.parse(diabetesPedigreeController.text),
        age: int.parse(ageController.text),
      );

      setState(() {
        result = response;
        showUpload = false;
      });

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Dados enviados e analisados com sucesso!')),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Erro: $e')),
      );
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  Widget _buildDiabetesForm() {
    return Form(
      key: _formKey,
      child: Column(
        children: [
          _buildNumberField('Gravidez', pregnanciesController),
          _buildNumberField('Glicose', glucoseController),
          _buildNumberField('Pressão sanguínea', bloodPressureController),
          _buildNumberField('Espessura da pele', skinThicknessController),
          _buildNumberField('Insulina', insulinController),
          _buildNumberField('IMC', bmiController, isDecimal: true),
          _buildNumberField('Histórico familiar', diabetesPedigreeController, isDecimal: true),
          _buildNumberField('Idade', ageController),
          const SizedBox(height: 10),
          ElevatedButton(
            style: ElevatedButton.styleFrom(backgroundColor: Colors.green),
            onPressed: isLoading ? null : _sendDiabetes,
            child: isLoading
                ? const CircularProgressIndicator(color: Colors.white)
                : const Text('Enviar dados de diabetes'),
          ),
        ],
      ),
    );
  }

  Widget _buildNumberField(String label, TextEditingController controller, {bool isDecimal = false}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: TextFormField(
        controller: controller,
        keyboardType: TextInputType.numberWithOptions(decimal: isDecimal),
        validator: (value) {
          if (value == null || value.isEmpty) return 'Campo obrigatório';
          return null;
        },
        decoration: InputDecoration(
          labelText: label,
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
        ),
      ),
    );
  }

  // Resultado da predição
  Widget _buildResultCard() {
    return Card(
      color: Colors.white,
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (result!['filename'] != null)
              Text('Arquivo: ${result!['filename']}',
                  style: const TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Text('Diagnóstico: ${result!['diagnosis']}'),
            if (result!.containsKey('confidence'))
              Text('Confiança: ${( (result!['confidence'] ?? 0) * 100).toStringAsFixed(2)}%'),
            if (result!.containsKey('probability'))
              Text('Probabilidade: ${( (result!['probability'] ?? 0) * 100).toStringAsFixed(2)}%'),
            const SizedBox(height: 10),
            if (result!['explanation'] != null) ...[
              const Text(
                'Explicação da IA:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              Text(result!['explanation']),
            ],
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF6ddbd7),
      appBar: AppBar(
        backgroundColor: const Color(0xFF6ddbd7),
        automaticallyImplyLeading: true,
        iconTheme: const IconThemeData(color: Colors.black),
        title: RichText(
          text: const TextSpan(
            text: 'dI',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Colors.black,
            ),
            children: [
              TextSpan(
                text: 'A',
                style: TextStyle(fontWeight: FontWeight.w900, fontSize: 24),
              ),
              TextSpan(
                text: 'gnostic',
                style: TextStyle(fontWeight: FontWeight.w400),
              ),
            ],
          ),
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Info card
              Container(
                decoration: BoxDecoration(
                  color: const Color(0xFFFFD580),
                  borderRadius: BorderRadius.circular(10),
                ),
                padding: const EdgeInsets.all(16),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const CircleAvatar(
                      backgroundColor: Color(0xFF20C4D1),
                      child: Icon(Icons.info, color: Colors.white),
                      radius: 14,
                    ),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Text(
                        'O dIAgnostic é um app com intuito de auxiliar o médico durante o processo de análise de exames clínicos.',
                        style: TextStyle(fontSize: 14),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 20),

              // Upload area
              Container(
                decoration: BoxDecoration(
                  color: const Color(0xFF244156),
                  borderRadius: BorderRadius.circular(10),
                ),
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Resultados do exame',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 12),

                    // Botões de seleção
                    Row(
                      children: [
                        ChoiceChip(
                          label: const Text('Pneumonia'),
                          selected: !isDiabetes,
                          onSelected: (_) {
                            setState(() {
                              isDiabetes = false;
                              showUpload = true;
                            });
                          },
                        ),
                        const SizedBox(width: 8),
                        ChoiceChip(
                          label: const Text('Diabetes'),
                          selected: isDiabetes,
                          onSelected: (_) {
                            setState(() {
                              isDiabetes = true;
                              showUpload = true;
                            });
                          },
                        ),
                      ],
                    ),

                    if (showUpload) ...[
                      const SizedBox(height: 20),
                      isDiabetes
                          ? _buildDiabetesForm()
                          : Column(
                              children: [
                                ElevatedButton.icon(
                                  icon: const Icon(Icons.photo_library),
                                  label: const Text('Selecionar imagem'),
                                  onPressed: _pickImage,
                                ),
                                const SizedBox(height: 10),
                                _image != null
                                    ? Image.file(_image!, height: 150)
                                    : const Text(
                                        'Nenhuma imagem selecionada',
                                        style: TextStyle(color: Colors.white70),
                                      ),
                                const SizedBox(height: 10),
                                ElevatedButton(
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: Colors.green,
                                  ),
                                  onPressed: isLoading ? null : _sendImage,
                                  child: isLoading
                                      ? const CircularProgressIndicator(
                                          color: Colors.white,
                                        )
                                      : const Text('Enviar imagem'),
                                ),
                              ],
                            ),
                    ],

                    const SizedBox(height: 20),

                    if (result != null) _buildResultCard(),
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
