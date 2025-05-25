import 'package:flutter/material.dart';
import 'dart:async';
import '../../core/constants/app_colors.dart';
import '../../core/constants/app_strings.dart';
import '../../core/services/api_service.dart';
import '../../core/services/auth_service.dart';
import '../../core/utils/validators.dart';
import '../../core/utils/helpers.dart';
import '../../widgets/common/custom_text_field.dart';
import '../../widgets/common/custom_button.dart';
import '../home/home_screen.dart';
import 'register_screen.dart';
import 'recovery_screen.dart';

class LoginScreen extends StatefulWidget {
  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _formKey = GlobalKey<FormState>();
  final TextEditingController emailController = TextEditingController();
  final TextEditingController passwordController = TextEditingController();
  bool rememberMe = false;
  bool isLoading = false;

  Future<void> _login() async {
    if (!_formKey.currentState!.validate()) return;

    if (!mounted) return;
    setState(() => isLoading = true);

    try {
      print('Iniciando login...'); // Debug
      
      final response = await ApiService.login(
        emailController.text,
        passwordController.text,
      ).timeout(const Duration(seconds: 15)); // Adiciona timeout

      print('Resposta recebida: $response'); // Debug

      if (response['access_token'] == null) {
        throw Exception('Token de acesso não recebido');
      }

      final String token = response['access_token'];
      final String name = response['nome'];

      print('Salvando token e nome...'); // Debug
      await AuthService.saveToken(token);
      await AuthService.saveUserName(name);

      if (!mounted) return;
      print('Navegando para HomeScreen...'); // Debug
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => HomeScreen(userName: name, token: token),
        ),
      );
    } on TimeoutException {
      print('Timeout no login'); // Debug
      if (mounted) {
        Helpers.showSnackBar(context, 'Tempo de conexão esgotado', isError: true);
      }
    } catch (e) {
      print('Erro no login: $e'); // Debug
      if (mounted) {
        Helpers.showSnackBar(
          context,
          e.toString().replaceAll('Exception: ', ''),
          isError: true,
        );
      }
    } finally {
      print('Finalizando login (isLoading=false)'); // Debug
      if (mounted) {
        setState(() => isLoading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      body: Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                AppStrings.appName,
                style: TextStyle(
                  fontSize: 32,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 30),
              Container(
                padding: const EdgeInsets.all(20),
                margin: const EdgeInsets.symmetric(horizontal: 30),
                decoration: BoxDecoration(
                  color: AppColors.cardBackground,
                  borderRadius: BorderRadius.circular(15),
                ),
                child: Form(
                  key: _formKey,
                  child: Column(
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.start,
                        children: [
                          const Text(
                            AppStrings.noAccount,
                            style: TextStyle(color: Colors.white),
                          ),
                          GestureDetector(
                            onTap: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(builder: (context) => RegisterScreen()),
                              );
                            },
                            child: const Text(
                              AppStrings.signUpHere,
                              style: TextStyle(
                                color: Colors.blueAccent,
                                decoration: TextDecoration.underline,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 20),
                      CustomTextField(
                        hintText: AppStrings.email,
                        controller: emailController,
                        keyboardType: TextInputType.emailAddress,
                        validator: Validators.validateEmail,
                      ),
                      const SizedBox(height: 10),
                      CustomTextField(
                        hintText: AppStrings.password,
                        controller: passwordController,
                        obscureText: true,
                      ),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Row(
                            children: [
                              Checkbox(
                                fillColor: MaterialStateProperty.all(AppColors.accent),
                                value: rememberMe,
                                onChanged: (value) {
                                  setState(() {
                                    rememberMe = value ?? false;
                                  });
                                },
                              ),
                              const Text(
                                AppStrings.rememberMe,
                                style: TextStyle(color: Colors.white),
                              ),
                            ],
                          ),
                          TextButton(
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(builder: (context) => RecoveryScreen()),
                              );
                            },
                            child: const Text(
                              AppStrings.forgotPassword,
                              style: TextStyle(color: Colors.white),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),
                      CustomButton(
                        text: AppStrings.loginTitle.toUpperCase(),
                        onPressed: _login,
                        isLoading: isLoading,
                      ),
                      const SizedBox(height: 10),
                    ],
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