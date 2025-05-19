import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class Upload extends StatefulWidget {
  @override
  _Upload createState() => _Upload();
}

class _Upload extends State<Upload> {
  bool showUpload = false;
  File? _image;
  final ImagePicker _picker = ImagePicker();

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

  void _sendImage() {
    if (_image != null) {
      print('Imagem enviada: ${_image!.path}');
      setState(() {
        showUpload = false;
        _image = null;
      });
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Imagem enviada com sucesso!')));
    } else {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Selecione uma imagem primeiro!')));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Color(0xFF6ddbd7),
      appBar: AppBar(
        backgroundColor: const Color(0xFF6ddbd7),
        automaticallyImplyLeading: true,
        iconTheme: IconThemeData(color: Colors.black),
        title: RichText(
          text: TextSpan(
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
        actions: [
          IconButton(
            icon: Icon(Icons.notifications_none, size: 24, color: Colors.black),
            onPressed: () {
              print('Notificação clicada');
            },
          ),
          IconButton(
            icon: Icon(Icons.mail_outline, size: 24, color: Colors.black),
            onPressed: () {
              print('Mensagem clicada');
            },
          ),
          SizedBox(width: 16),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                decoration: BoxDecoration(
                  color: Color(0xFFFFD580),
                  borderRadius: BorderRadius.circular(10),
                ),
                padding: EdgeInsets.all(16),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    CircleAvatar(
                      backgroundColor: Color(0xFF20C4D1),
                      child: Icon(Icons.info, color: Colors.white),
                      radius: 14,
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
              SizedBox(height: 20),
              Container(
                decoration: BoxDecoration(
                  color: Color(0xFF244156),
                  borderRadius: BorderRadius.circular(10),
                ),
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Resultados do exame',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    SizedBox(height: 12),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Color(0xFF66A8BD),
                          foregroundColor: Colors.white,
                          padding: EdgeInsets.symmetric(vertical: 14),
                        ),
                        onPressed: () {
                          setState(() {
                            showUpload = !showUpload;
                          });
                        },
                        child: Text(
                          'Enviar resultados dos exames',
                          style: TextStyle(fontSize: 14),
                        ),
                      ),
                    ),
                    if (showUpload) ...[
                      SizedBox(height: 20),
                      ElevatedButton.icon(
                        icon: Icon(Icons.photo_library),
                        label: Text('Selecionar imagem'),
                        onPressed: _pickImage,
                      ),
                      SizedBox(height: 10),
                      _image != null
                          ? Image.file(_image!, height: 150)
                          : Text(
                            'Nenhuma imagem selecionada',
                            style: TextStyle(color: Colors.white70),
                          ),
                      SizedBox(height: 10),
                      ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green,
                        ),
                        onPressed: _sendImage,
                        child: Text('Enviar imagem'),
                      ),
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
