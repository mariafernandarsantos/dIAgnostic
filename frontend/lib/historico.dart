import 'package:flutter/material.dart';

class ConsultasPage extends StatefulWidget {
  final String nomeUsuario;

  const ConsultasPage({super.key, required this.nomeUsuario});

  @override
  State<ConsultasPage> createState() => _ConsultasPage();
}

class _ConsultasPage extends State<ConsultasPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFF6ddbd7),
        title: Text.rich(
          TextSpan(
            children: [
              TextSpan(
                text: 'dlA',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 24),
              ),
              TextSpan(text: 'gnostic', style: TextStyle(fontSize: 24)),
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
      backgroundColor: const Color(0xFF6ddbd7),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: const Color(0xFF22364A),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  children: [
                    Row(
                      children: [
                        const CircleAvatar(
                          backgroundColor: Colors.white,
                          child: Icon(Icons.person, color: Colors.grey),
                        ),
                        const SizedBox(width: 8),
                        Expanded(
                          child: Text(
                            widget.nomeUsuario,
                            style: TextStyle(color: Colors.white),
                          ),
                        ),
                        ElevatedButton(
                          onPressed: () {},
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.blueGrey[100],
                            foregroundColor: Colors.black,
                          ),
                          child: const Text('Abrir perfil'),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 8,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.blueGrey[400],
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Row(
                        children: const [
                          Icon(Icons.folder, color: Colors.white),
                          SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              'dlAgnostic',
                              style: TextStyle(color: Colors.white),
                            ),
                          ),
                          Icon(Icons.chevron_right, color: Colors.white),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),
              const Center(
                child: Text(
                  'Minhas consultas',
                  style: TextStyle(
                    color: Color(0xFF22364A),
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              const SizedBox(height: 16),
              const Expanded(
                child: Center(
                  child: Text(
                    'Nenhuma consulta disponível no momento.',
                    style: TextStyle(color: Color(0xFF22364A), fontSize: 16),
                    textAlign: TextAlign.center,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
