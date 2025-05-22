import 'package:diaglogin/upload.dart';
import 'package:flutter/material.dart';
import 'package:diaglogin/historico.dart';

class Inicio extends StatefulWidget {
  final String nomeUsuario;
  final String token;

  const Inicio({super.key, required this.nomeUsuario, required this.token});

  @override
  State<Inicio> createState() => _InicioState();
}

class _InicioState extends State<Inicio> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF6ddbd7),
      appBar: AppBar(
        backgroundColor: const Color(0xFF6ddbd7),
        elevation: 0,
        title: Text.rich(
          TextSpan(
            children: [
              TextSpan(
                text: 'dIA',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 26,
                  color: Colors.white,
                ),
              ),
              TextSpan(
                text: 'gnostic',
                style: TextStyle(fontSize: 26, color: Colors.white),
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
      body: Column(
        children: [
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 5),
              child: Column(
                children: [
                  const SizedBox(height: 20),
                  Container(
                    padding: const EdgeInsets.all(20),
                    margin: const EdgeInsets.symmetric(horizontal: 10),
                    decoration: BoxDecoration(
                      color: const Color.fromARGB(255, 61, 77, 100),
                      borderRadius: BorderRadius.circular(15),
                    ),
                    child: Column(
                      children: [
                        Row(
                          children: [
                            CircleAvatar(
                              backgroundColor: Colors.white,
                              child: Icon(Icons.person, color: Colors.grey),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                widget.nomeUsuario,
                                style: TextStyle(
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                ),
                              ),
                            ),
                            ElevatedButton(
                              onPressed: () {
                                print('Abriu o perfil');
                              },
                              style: ElevatedButton.styleFrom(
                                backgroundColor: const Color.fromARGB(255, 117, 167, 189),
                                foregroundColor: Colors.white,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(20),
                                ),
                              ),
                              child: const Text('Abrir perfil'),
                            ),
                          ],
                        ),
                        const SizedBox(height: 20),
                        SizedBox(
                          width: double.infinity,
                          child: ElevatedButton(
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) => ConsultasPage(nomeUsuario: widget.nomeUsuario),
                                ),
                              );
                            },
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color.fromARGB(255, 117, 167, 189),
                              foregroundColor: Colors.white,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(20),
                              ),
                            ),
                            child: const Text('Histórico de Consultas'),
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 20),
                  Container(
                    padding: const EdgeInsets.all(20),
                    margin: const EdgeInsets.symmetric(horizontal: 10),
                    constraints: BoxConstraints(
                      minHeight: 50,
                      maxWidth: 400,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.yellow,
                      borderRadius: BorderRadius.circular(15),
                    ),
                    child: const Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Text(
                          'Alerta de Dengue',
                          style: TextStyle(
                            fontSize: 20,
                            fontWeight: FontWeight.bold,
                            color: Colors.black,
                          ),
                        ),
                        SizedBox(height: 5),
                        Text(
                          'Os casos de dengue estão aumentando na região. Elimine qualquer foco de água parada, mantenha caixas d\'água bem tampadas e use repelente diariamente. Cuide da sua saúde e ajude a prevenir a doença!',
                          style: TextStyle(
                            fontSize: 15,
                            fontWeight: FontWeight.bold,
                            color: Colors.black,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 20),
                  InkWell(
                   onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(builder: (context) => Upload(token: widget.token)),
                      );
                    },
                    borderRadius: BorderRadius.circular(8),
                    child: Container(
                      padding: const EdgeInsets.all(20),
                      margin: const EdgeInsets.symmetric(horizontal: 10),
                      decoration: BoxDecoration(
                        color: const Color.fromARGB(255, 61, 77, 100),
                        borderRadius: BorderRadius.circular(15),
                      ),
                      child: Row(
                        children: [
                          const Icon(
                            Icons.description_outlined, 
                            color: Colors.white, 
                            size: 28
                          ),
                          const SizedBox(width: 12),
                          const Text(
                            'dIAgnostic',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                              color: Colors.white,
                            ),
                          ),
                          const Spacer(),
                          const Icon(
                            Icons.arrow_forward_ios, 
                            color: Colors.white, 
                            size: 20
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 10),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 10),
                    child: Image.asset(
                      "assets/dengue.png",
                      width: 400,
                      height: 140,
                      fit: BoxFit.contain,
                    ),
                  ),
                  const SizedBox(height: 10),
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 10),
                    child: Image.asset(
                      "assets/dengue.png",
                      width: 400,
                      height: 140,
                      fit: BoxFit.contain,
                    ),
                  )
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}