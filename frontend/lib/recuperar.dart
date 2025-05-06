import 'package:flutter/material.dart';

class Recuperar extends StatefulWidget {
  @override
  State<Recuperar> createState() => _Recuperar();
}

class _Recuperar extends State<Recuperar> {
  final TextEditingController recuperacaoController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFF6ddbd7),
        leading: IconButton(
          onPressed: () {
            Navigator.pop(context);
          }, 
          icon: const Icon(Icons.arrow_back),
        ),
      ),
      backgroundColor: const Color(0xFF6ddbd7),
      body: Center(
        child: SingleChildScrollView(
          child: Column(
            children: [
              Container(
                padding: const EdgeInsets.all(20),
                margin: const EdgeInsets.symmetric(horizontal: 30),
                decoration: BoxDecoration(
                  color: const Color.fromARGB(255, 61, 77, 100),
                  borderRadius: BorderRadius.circular(15),
                ),
                child: Column(
                  children: [
                    const Text(
                      'Um link será enviado para o seu email para redefinir sua senha.',
                      style: TextStyle(color: Colors.white),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 20),
                    TextField(
                      controller: recuperacaoController,
                      decoration: const InputDecoration(
                        hintText: 'Email',
                        filled: true,
                        fillColor: Colors.white,
                        border: OutlineInputBorder(),
                      ),
                    ),
                    const SizedBox(height: 20),
                    ElevatedButton(
                      onPressed: () {
                        print('Solicitação de recuperação para: ${recuperacaoController.text}');
                      },
                      child: const Text('ENVIAR'),
                    ),
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