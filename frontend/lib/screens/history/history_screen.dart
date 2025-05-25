import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import '../../../core/services/storage_service.dart';
import '../../../core/constants/app_colors.dart';
import '../../../core/constants/app_strings.dart';
import '../../../models/prediction_result.dart';
import '../../../widgets/common/custom_app_bar.dart';
import '../../../widgets/common/loading_widget.dart';
import '../../core/services/api_service.dart';
import '../../core/services/auth_service.dart';
import 'widgets/consultation_card.dart';

class HistoryScreen extends StatefulWidget {
  final String userName;

  const HistoryScreen({
    super.key,
    required this.userName,
  });

  @override 
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> with SingleTickerProviderStateMixin {
  List<PredictionResult> consultations = [];
  bool isLoading = true;
  late TabController _tabController;
  
  // Tabs disponíveis
  final List<String> _tabs = ['Todos', 'Pneumonia', 'Diabetes'];
  
  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: _tabs.length, vsync: this);
    _tabController.addListener(_onTabChanged);
    _loadConsultations();
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  void _onTabChanged() {
    if (_tabController.indexIsChanging) {
      _loadConsultations();
    }
  }

  String get selectedFilter => _tabs[_tabController.index];

  Future<void> _loadConsultations() async {
    setState(() {
      isLoading = true;
    });

    try {
      final token = await AuthService.getToken();
      if (token == null) throw Exception("Token não encontrado");

      // Determina o tipo de predição baseado na tab selecionada
      String? predictionType;
      if (selectedFilter != 'Todos') {
        predictionType = selectedFilter.toLowerCase();
      }

      final rawData = await ApiService.getPredictionHistory(
        token: token,
        predictionType: predictionType,
      );

      setState(() {
        consultations = rawData
            .map<PredictionResult>((json) => PredictionResult.fromJson(json))
            .toList();
        isLoading = false;
      });
    } catch (e) {
      setState(() {
        isLoading = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Erro ao carregar histórico: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  List<PredictionResult> get filteredConsultations {
    if (selectedFilter == 'Todos') {
      return consultations;
    }
    return consultations.where((consultation) => 
      consultation.diagnosis.toLowerCase() == selectedFilter.toLowerCase()
    ).toList();
  }

  Future<void> _clearHistory() async {
    final shouldClear = await showDialog<bool>(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text('Limpar Histórico'),
          content: const Text(
            'Tem certeza que deseja limpar todo o histórico de consultas? Esta ação não pode ser desfeita.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(false),
              child: const Text('Cancelar'),
            ),
            TextButton(
              onPressed: () => Navigator.of(context).pop(true),
              style: TextButton.styleFrom(foregroundColor: Colors.red),
              child: const Text('Limpar'),
            ),
          ],
        );
      },
    );

    if (shouldClear == true) {
      try {
        await StorageService.clearConsultationHistory();
        setState(() {
          consultations.clear();
        });
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Histórico limpo com sucesso'),
              backgroundColor: Colors.green,
            ),
          );
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Erro ao limpar histórico: $e'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.primary,
      appBar: CustomAppBar(
        title: AppStrings.appName,
        showBackButton: true,
        actions: [
          if (consultations.isNotEmpty)
            PopupMenuButton<String>(
              icon: const Icon(Icons.more_vert, color: Colors.black),
              onSelected: (value) {
                if (value == 'clear') {
                  _clearHistory();
                }
              },
              itemBuilder: (BuildContext context) => [
                const PopupMenuItem<String>(
                  value: 'clear',
                  child: Row(
                    children: [
                      Icon(Icons.clear_all, color: Colors.red),
                      SizedBox(width: 8),
                      Text('Limpar Histórico'),
                    ],
                  ),
                ),
              ],
            ),
        ],
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 16),
              
              // Card do usuário
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppColors.cardBackground,
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
                        const SizedBox(width: 12),
                        Expanded(
                          child: Text(
                            widget.userName,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ),
                        ElevatedButton(
                          onPressed: () {
                            // TODO: Implementar abertura do perfil
                          },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppColors.secondary,
                            foregroundColor: Colors.white,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(20),
                            ),
                            minimumSize: const Size(100, 36),
                          ),
                          child: const Text('Abrir perfil'),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 8,
                      ),
                      decoration: BoxDecoration(
                        color: AppColors.secondary,
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: const Row(
                        children: [
                          Icon(Icons.folder, color: Colors.white),
                          SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              'dIAgnostic',
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
              
              // Título
              const Center(
                child: Text(
                  AppStrings.myConsultations,
                  style: TextStyle(
                    color: AppColors.primary,
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              
              const SizedBox(height: 16),
              
              // TabBar
              Container(
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(25),
                ),
                child: TabBar(
                  controller: _tabController,
                  indicator: BoxDecoration(
                    color: AppColors.secondary,
                    borderRadius: BorderRadius.circular(25),
                  ),
                  indicatorSize: TabBarIndicatorSize.tab,
                  labelStyle: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                  ),
                  unselectedLabelStyle: const TextStyle(
                    fontWeight: FontWeight.normal,
                    fontSize: 14,
                  ),
                  tabs: _tabs.asMap().entries.map((entry) {
                    final index = entry.key;
                    final tab = entry.value;
                    final isSelected = _tabController.index == index;
                    
                    return Tab(
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        child: Text(
                          tab,
                          style: TextStyle(
                            color: isSelected ? Colors.white : AppColors.secondary,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    );
                  }).toList(),
                ),
              ),
              
              const SizedBox(height: 20),
              
              // Conteúdo das tabs
              Expanded(
                child: TabBarView(
                  controller: _tabController,
                  children: _tabs.map((tab) => _buildTabContent()).toList(),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTabContent() {
  if (isLoading) {
    return const LoadingWidget();
  }

  if (filteredConsultations.isEmpty) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            _getTabIcon(),
            size: 64,
            color: Colors.white.withOpacity(0.5),
          ),
          const SizedBox(height: 16),
          Text(
            _getEmptyMessage(),
            style: TextStyle(
              color: AppColors.primary,
              fontSize: 16,
            ),
            textAlign: TextAlign.center,
          ),
          if (consultations.isEmpty && selectedFilter == 'Todos') ...[
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: () => Navigator.pop(context),
              icon: const Icon(Icons.add),
              label: const Text('Fazer primeira consulta'),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppColors.secondary,
                foregroundColor: Colors.white,
              ),
            ),
          ],
        ],
      ),
    );
  }

  return RefreshIndicator(
    onRefresh: _loadConsultations,
    child: ListView.builder(
      itemCount: filteredConsultations.length,
      itemBuilder: (context, index) {
        // Add this check to prevent the error
        if (index >= filteredConsultations.length) {
          return const SizedBox(); // Return empty widget if index is invalid
        }
        
        final consultation = filteredConsultations[index];
        return Padding(
          padding: const EdgeInsets.only(bottom: 12.0),
          child: ConsultationCard(
            consultation: consultation,
            onTap: () {
              _showConsultationDetails(consultation);
            },
          ),
        );
      },
    ),
  );
}

  IconData _getTabIcon() {
    switch (selectedFilter) {
      case 'Pneumonia':
        return Icons.air;
      case 'Diabetes':
        return Icons.bloodtype;
      default:
        return Icons.history;
    }
  }

  String _getEmptyMessage() {
    if (consultations.isEmpty && selectedFilter == 'Todos') {
      return AppStrings.noConsultations;
    }
    
    return 'Nenhuma consulta de ${selectedFilter.toLowerCase()} encontrada.';
  }

  void _showConsultationDetails(PredictionResult consultation) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        height: MediaQuery.of(context).size.height * 0.8,
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
        ),
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Handle bar
            Center(
              child: Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey[300],
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),
            const SizedBox(height: 20),
            
            // Título
            Row(
              children: [
                Icon(
                  consultation.diagnosis.toLowerCase() == 'pneumonia' 
                    ? Icons.air 
                    : Icons.bloodtype,
                  color: AppColors.primary,
                ),
                const SizedBox(width: 8),
                const Text(
                  'Detalhes da Consulta',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),
            
            // Conteúdo
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _buildDetailRow('Tipo', consultation.diagnosis),
                    _buildDetailRow('Resultado', consultation.result),
                    _buildDetailRow('Data', consultation.timestamp.toLocal().toString()),
                    _buildDetailRow('Revisado por médico', consultation.doctorReviewed ? 'Sim' : 'Não'),
                    _buildDetailRow('Notas do médico', consultation.doctorNotes ?? 'Nenhuma nota disponível'),
                    if (consultation.confidence != null)
                      _buildDetailRow(
                        'Probabilidade',
                        '${(consultation.confidence! * 100).toStringAsFixed(1)}%',
                      ),
                    if (consultation.explanation != null) ...[
                      const SizedBox(height: 16),
                      const Text(
                        'Explicação da IA:',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 8),
                      MarkdownBody(
                        data: consultation.explanation!,
                        styleSheet: MarkdownStyleSheet(
                          p: const TextStyle(fontSize: 14, color: Colors.black87),
                          h2: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: AppColors.primary,
                          ),
                          listBullet: const TextStyle(fontSize: 14),
                          strong: const TextStyle(fontWeight: FontWeight.bold),
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
            
            // Botão fechar
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () => Navigator.pop(context),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.primary,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
                child: const Text(
                  'Fechar',
                  style: TextStyle(color: Colors.white),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildDetailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 100,
            child: Text(
              '$label:',
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                fontSize: 14,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(fontSize: 14),
            ),
          ),
        ],
      ),
    );
  }
}