# dIAgnostic API

Um aplicativo construído em flutter para integrar o projeto dIAgnostic, que realizará diagnósticos clínicos baseados em exames enviados pelo usuário.      

## 🧠 Visão Geral

Esse aplicativo permite:

- Cadastro e Login de usuário
- Uso de inteligência artificial para leitura e análise de exames enviados
- Histórico de consultas feitas
- Sugestões para cuidados com a saúde

## 🧰 Tecnologias e Requisitos

- Flutter 3.0+
- Android Studio
- Dart

## 📦 Instalação

1. Clone este repositório:

```bash
git clone https://github.com/mariafernandarsantos/dIAgnostic.git
cd dIAgnostic/frontend
```

2. Instale as dependências do projeto
```bash
flutter pub get
```

## Uso

### Primeiros passos

- Tela de Login, onde o usuário poderá entrar no sistema usando o email e senha cadastrado.
- Caso não haja cadastro ao clicar em 'Increva-se aqui!' o usuário será redirecionado para a tela de cadastro do usuário

### Utilizando o aplicativo

- Clicar no botão dIAgnostic
- Clicar em 'Enviar resultados dos exames' e selecionar as fotos que deseja para consulta
- A análise será exibida na tela logo em seguida

### Verificando histórico

- Na tela inicial, logo abaixo do perfil do usuário selecionar "Histórico de Consultas"
- Serão exibidias todas as consultas feitas, e o usuário poderá revê-las novamente caso desejado.

### Estrutura de pastas do projeto
```bash
lib/
├── core/
│   ├── constants/
│   │   ├── app_colors.dart
│   │   ├── app_strings.dart
│   │   └── api_endpoints.dart
│   ├── services/
│   │   ├── api_service.dart
│   │   ├── auth_service.dart
│   │   └── storage_service.dart
│   └── utils/
│       ├── validators.dart
│       └── helpers.dart
├── models/
│   ├── user_model.dart
│   ├── prediction_result.dart
│   └── diabetes_data.dart
├── screens/
│   ├── auth/
│   │   ├── login_screen.dart
│   │   ├── register_screen.dart
│   │   └── recovery_screen.dart
│   ├── home/
│   │   ├── home_screen.dart
│   │   └── widgets/
│   │       ├── user_info_card.dart
│   │       ├── alert_card.dart
│   │       └── diagnostic_card.dart
│   ├── diagnostic/
│   │   ├── upload_screen.dart
│   │   └── widgets/
│   │       ├── pneumonia_form.dart
│   │       ├── diabetes_form.dart
│   │       └── result_card.dart
│   └── history/
│       ├── history_screen.dart
│       └── widgets/
│           └── consultation_card.dart
├── widgets/
│   ├── common/
│   │   ├── custom_button.dart
│   │   ├── custom_text_field.dart
│   │   ├── loading_widget.dart
│   │   └── app_bar_widget.dart
│   └── forms/
│       ├── number_field.dart
│       └── dropdown_field.dart
└── main.dart
```
