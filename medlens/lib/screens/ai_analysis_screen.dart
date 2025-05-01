import 'package:flutter/material.dart';
import 'package:medlens/models/disease_model.dart';
import 'package:medlens/models/symptom_model.dart';
import 'package:medlens/services/medical_api_service.dart';
import 'package:medlens/utils/constants.dart';
import 'package:medlens/widgets/loading_indicator.dart';
import 'package:medlens/widgets/research_card.dart';
import 'package:medlens/utils/theme_helper.dart';

class AIAnalysisScreen extends StatefulWidget {
  final List<SymptomModel> selectedSymptoms;
  final Map<String, dynamic> vitalParameters;
  final List<DiseaseModel>? diagnosisResults;
  
  const AIAnalysisScreen({
    Key? key, 
    required this.selectedSymptoms, 
    required this.vitalParameters,
    this.diagnosisResults,
  }) : super(key: key);

  @override
  State<AIAnalysisScreen> createState() => _AIAnalysisScreenState();
}

class _AIAnalysisScreenState extends State<AIAnalysisScreen> {
  final MedicalApiService _apiService = MedicalApiService();
  
  bool _isLoading = true;
  bool _hasError = false;
  String _errorMessage = '';
  
  // Data from AI analysis
  List<Map<String, dynamic>> _researchData = [];
  List<Map<String, dynamic>> _riskFactors = [];
  Map<String, dynamic> _globalStatistics = {};
  
  @override
  void initState() {
    super.initState();
    _performAIAnalysis();
  }
  
  Future<void> _performAIAnalysis() async {
    setState(() {
      _isLoading = true;
      _hasError = false;
    });
    
    try {
      // Step 1: Convert selected symptoms to IDs for the API
      final symptomIds = widget.selectedSymptoms.map((s) => s.id).toList();
      
      // Step 2: Fetch medical data from internet sources via API
      final medicalData = await _apiService.searchMedicalData(
        symptomIds, 
        widget.vitalParameters
      );
      
      // Step 3: Extract research data
      if (medicalData.containsKey('researchData')) {
        _researchData = List<Map<String, dynamic>>.from(medicalData['researchData']);
      }
      
      // Step 4: Extract risk factors
      if (medicalData.containsKey('riskFactors')) {
        _riskFactors = List<Map<String, dynamic>>.from(medicalData['riskFactors']);
      }
      
      // Step 5: If we have a primary diagnosis, get global statistics for it
      if (widget.diagnosisResults != null && widget.diagnosisResults!.isNotEmpty) {
        final primaryDisease = widget.diagnosisResults!.first;
        _globalStatistics = await _apiService.getGlobalDiseaseStats(primaryDisease.name);
      }
      
      // Step 6: Check for medication interactions if previous prescriptions exist
      // This would need to be implemented based on your data model for prescriptions
      
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _hasError = true;
          _errorMessage = 'Failed to analyze medical data: ${e.toString()}';
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI Medical Analysis'),
        backgroundColor: AppColors.primaryColor,
        elevation: 2,
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: _showDisclaimerDialog,
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: LoadingIndicator(text: 'Analyzing medical data from trusted sources...'))
          : _hasError
              ? _buildErrorView()
              : _buildAnalysisResults(),
      floatingActionButton: FloatingActionButton(
        onPressed: _isLoading ? null : _performAIAnalysis,
        backgroundColor: _isLoading ? Colors.grey : AppColors.accentColor,
        child: const Icon(Icons.refresh),
        tooltip: 'Refresh Analysis',
      ),
    );
  }
  
  Widget _buildErrorView() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Icon(Icons.error_outline, size: 60, color: Colors.red),
          const SizedBox(height: 16),
          Text(
            'Analysis Error',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 8),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Text(
              _errorMessage,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
          ),
          const SizedBox(height: 24),
          ElevatedButton(
            onPressed: _performAIAnalysis,
            child: const Text('Try Again'),
          ),
        ],
      ),
    );
  }

  Widget _buildAnalysisResults() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildMedicalDisclaimer(),
          const SizedBox(height: 16),
          
          // Selected symptoms summary
          _buildSectionHeader('Selected Symptoms'),
          _buildSelectedSymptoms(),
          const SizedBox(height: 16),
          
          // Primary diagnosis
          if (widget.diagnosisResults != null && widget.diagnosisResults!.isNotEmpty)
            ...[
              _buildSectionHeader('Primary Diagnosis'),
              _buildPrimaryDiagnosis(widget.diagnosisResults!.first),
              const SizedBox(height: 16),
            ],
          
          // Research findings
          if (_researchData.isNotEmpty) ...[
            _buildSectionHeader('Medical Research Insights'),
            ..._researchData.map(_buildResearchItem).toList(),
            const SizedBox(height: 16),
          ],
          
          // Risk factors
          if (_riskFactors.isNotEmpty) ...[
            _buildSectionHeader('Identified Risk Factors'),
            ..._riskFactors.map(_buildRiskFactorItem).toList(),
            const SizedBox(height: 16),
          ],
          
          // Global statistics
          if (_globalStatistics.isNotEmpty) ...[
            _buildSectionHeader('Global Health Statistics'),
            _buildGlobalStatistics(),
            const SizedBox(height: 16),
          ],
          
          const SizedBox(height: 40),
        ],
      ),
    );
  }
  
  Widget _buildMedicalDisclaimer() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.amber.shade50,
        border: Border.all(color: Colors.amber.shade200),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.warning_amber_rounded, color: Colors.amber),
              SizedBox(width: 8),
              Text(
                'MEDICAL DISCLAIMER',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          const Text(
            'This AI analysis is for informational purposes only and does not constitute medical advice. '
            'Always consult with a qualified healthcare provider for diagnosis and treatment.',
            style: TextStyle(fontSize: 14),
          ),
        ],
      ),
    );
  }
  
  Widget _buildSectionHeader(String title) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
            color: AppColors.primaryColor,
          ),
        ),
        const Divider(thickness: 2),
      ],
    );
  }
  
  Widget _buildSelectedSymptoms() {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: widget.selectedSymptoms.map((symptom) {
        return Chip(
          label: Text(symptom.name),
          backgroundColor: AppColors.secondaryColor.withOpacity(0.2),
        );
      }).toList(),
    );
  }
  
  Widget _buildPrimaryDiagnosis(DiseaseModel disease) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    disease.name,
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: ThemeHelper.getConfidenceColor(disease.accuracy),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    '${(disease.accuracy * 100).toStringAsFixed(0)}%',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(disease.description),
            if (disease.specialisations.isNotEmpty) ...[
              const SizedBox(height: 16),
              const Text(
                'Recommended Specialists:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 4),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: disease.specialisations.map((spec) {
                  return Chip(
                    label: Text(spec),
                    backgroundColor: Colors.blue.shade100,
                    padding: const EdgeInsets.all(0),
                  );
                }).toList(),
              ),
            ],
          ],
        ),
      ),
    );
  }
  
  Widget _buildResearchItem(Map<String, dynamic> research) {
    return ResearchCard(
      title: research['title'] ?? 'Research Finding',
      summary: research['summary'] ?? 'No summary available',
      source: research['source'] ?? 'Medical Database',
      relevance: research['relevance'] ?? 'medium',
      date: research['date'] ?? 'Recent',
    );
  }
  
  Widget _buildRiskFactorItem(Map<String, dynamic> riskFactor) {
    final impact = riskFactor['impact'] ?? 'medium';
    
    Color getImpactColor() {
      switch (impact.toLowerCase()) {
        case 'high':
          return Colors.red.shade100;
        case 'medium':
          return Colors.orange.shade100;
        case 'low':
          return Colors.yellow.shade100;
        default:
          return Colors.grey.shade100;
      }
    }
    
    return Card(
      color: getImpactColor(),
      margin: const EdgeInsets.only(bottom: 8),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    riskFactor['factor'] ?? 'Risk Factor',
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                    ),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: impact.toLowerCase() == 'high' ? Colors.red : 
                           impact.toLowerCase() == 'medium' ? Colors.orange : Colors.yellow,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    impact.toUpperCase(),
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(riskFactor['description'] ?? 'No description available'),
            if (riskFactor.containsKey('recommendations') && 
                riskFactor['recommendations'] is List &&
                (riskFactor['recommendations'] as List).isNotEmpty) ...[
              const SizedBox(height: 8),
              const Text(
                'Recommendations:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              ...List<String>.from(riskFactor['recommendations']).map((rec) => 
                Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('• '),
                      Expanded(child: Text(rec)),
                    ],
                  ),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
  
  Widget _buildGlobalStatistics() {
    if (_globalStatistics.isEmpty) {
      return const Card(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text('No global statistics available.'),
        ),
      );
    }

    return Card(
      elevation: 3,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (_globalStatistics.containsKey('prevalence'))
              _buildStatItem('Global Prevalence', _globalStatistics['prevalence']),
            if (_globalStatistics.containsKey('annualCases'))
              _buildStatItem('Annual Cases', _globalStatistics['annualCases']),
            if (_globalStatistics.containsKey('mortalityRate'))
              _buildStatItem('Mortality Rate', _globalStatistics['mortalityRate']),
            if (_globalStatistics.containsKey('treatmentSuccess'))
              _buildStatItem('Treatment Success Rate', _globalStatistics['treatmentSuccess']),
            if (_globalStatistics.containsKey('sourceInfo')) ...[
              const SizedBox(height: 8),
              Text(
                'Source: ${_globalStatistics['sourceInfo']}',
                style: const TextStyle(
                  fontSize: 12,
                  fontStyle: FontStyle.italic,
                  color: Colors.grey,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
  
  Widget _buildStatItem(String label, dynamic value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(fontWeight: FontWeight.w500),
          ),
          Text(
            value.toString(),
            style: const TextStyle(
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
  
  void _showDisclaimerDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Medical AI Disclaimer'),
        content: const SingleChildScrollView(
          child: Text(
            'IMPORTANT: The information provided by this AI medical analysis feature is based on '
            'symptom correlation and medical research from reliable online sources. '
            'It is not a replacement for professional medical advice, diagnosis, or treatment.\n\n'
            'Always seek the advice of your physician or other qualified health provider with any '
            'questions you may have regarding a medical condition. Never disregard professional '
            'medical advice or delay in seeking it because of something you have read on this app.\n\n'
            'In case of a medical emergency, call your doctor or emergency services immediately.\n\n'
            'The creators of this application are not liable for any action taken or not taken based on '
            'the information provided by this AI analysis feature.',
          ),
        ),
        actions: [
          TextButton(
            child: const Text('I Understand'),
            onPressed: () => Navigator.of(context).pop(),
          ),
        ],
      ),
    );
  }
}