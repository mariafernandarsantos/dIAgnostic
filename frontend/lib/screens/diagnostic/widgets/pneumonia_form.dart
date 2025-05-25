import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../../../core/services/api_service.dart';
import '../../../core/constants/app_colors.dart';

class PneumoniaForm extends StatefulWidget {
  final String token;
  final Function(Map<String, dynamic>) onResult;
  final VoidCallback? onLoading;

  const PneumoniaForm({
    super.key,
    required this.token,
    required this.onResult,
    this.onLoading,
  });

  @override
  State<PneumoniaForm> createState() => _PneumoniaFormState();
}

class _PneumoniaFormState extends State<PneumoniaForm> {
  File? _selectedImage;
  final ImagePicker _picker = ImagePicker();
  bool _isLoading = false;

  Future<void> _pickImage() async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: ImageSource.gallery,
      );
      
      if (pickedFile != null) {
        setState(() {
          _selectedImage = File(pickedFile.path);
        });
      }
    } catch (e) {
      if (mounted) _showErrorMessage('Erro ao selecionar imagem: $e');
    }
  }

  Future<void> _takePicture() async {
    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: ImageSource.camera,
      );
      
      if (pickedFile != null) {
        setState(() {
          _selectedImage = File(pickedFile.path);
        });
      }
    } catch (e) {
      _showErrorMessage('Erro ao tirar foto: $e');
    }
  }

  Future<void> _submitImage() async {
    if (_selectedImage == null) {
      if (mounted) _showErrorMessage('Selecione uma imagem primeiro!');
      return;
    }

    if (!mounted) return;
    
    print('🔄 Setting loading state to true...');
    setState(() => _isLoading = true);
    widget.onLoading?.call();

    try {
      print('📞 Calling API...');
      final response = await ApiService.predictPneumonia(
        imageFile: _selectedImage!,
        token: widget.token,
      );

      print('✅ API call successful!');
      print('📊 Response received: ${response.toString()}');

      // Always call onResult, even if widget is unmounted
      // The parent needs to handle the response and clear its loading state
      print('📤 Calling onResult callback...');
      widget.onResult(response);

      if (mounted) {
        print('🔄 Widget mounted - Resetting UI state...');
        setState(() {
          _isLoading = false;
          _selectedImage = null;
        });
        
        print('🎉 Showing success message...');
        _showSuccessMessage('Imagem analisada com sucesso!');
      } else {
        print('⚠️ Widget unmounted - UI state not updated but onResult called');
      }
      
    } catch (e) {
      print('❌ Error in _submitImage: $e');
      print('❌ Error type: ${e.runtimeType}');
      
      // Always try to call onResult with error info so parent can clear loading
      try {
        widget.onResult({'error': true, 'message': e.toString()});
      } catch (callbackError) {
        print('❌ Error calling onResult: $callbackError');
      }
      
      if (mounted) {
        print('🔄 Resetting loading state after error...');
        setState(() => _isLoading = false);
        _showErrorMessage('Erro na análise: $e');
      }
    }
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

  void _removeImage() {
    setState(() {
      _selectedImage = null;
    });
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
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Text(
            'Análise de Pneumonia',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: AppColors.primary,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 8),
          const Text(
            'Faça upload de uma radiografia do tórax para análise',
            style: TextStyle(
              fontSize: 14,
              color: AppColors.secondary,
            ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 20),
          
          // Área de upload/preview da imagem
          Container(
            height: 200,
            decoration: BoxDecoration(
              color: Colors.grey[100],
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.grey[300]!),
            ),
            child: _selectedImage != null
                ? Stack(
                    children: [
                      ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Image.file(
                          _selectedImage!,
                          width: 400,
                          height: 400,
                          fit: BoxFit.cover,
                        ),
                      ),
                      Positioned(
                        top: 8,
                        right: 8,
                        child: IconButton(
                          onPressed: _removeImage,
                          icon: const Icon(Icons.close),
                          style: IconButton.styleFrom(
                            backgroundColor: Colors.red,
                            foregroundColor: Colors.white,
                          ),
                        ),
                      ),
                    ],
                  )
                : Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.cloud_upload_outlined,
                        size: 48,
                        color: Colors.grey[400],
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Nenhuma imagem selecionada',
                        style: TextStyle(
                          color: Colors.grey[600],
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
          ),
          
          const SizedBox(height: 16),
          
          // Botões de ação
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _isLoading ? null : _pickImage,
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Galeria'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _isLoading ? null : _takePicture,
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Câmera'),
                ),
              ),
            ],
          ),
          
          const SizedBox(height: 16),
          
          // Botão de análise
          ElevatedButton(
            onPressed: (_selectedImage != null && !_isLoading) ? _submitImage : null,
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
                    'Analisar Imagem',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
          ),
        ],
      ),
    );
  }
}