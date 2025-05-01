import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:medlens/models/disease_model.dart';
import 'package:medlens/models/symptom_model.dart';
import 'package:medlens/services/medical_service.dart';
import 'package:medlens/utils/app_config.dart';
import 'package:medlens/widgets/research_card.dart';
import 'package:medlens/widgets/risk_factor_card.dart';
import 'package:url_launcher/url_launcher.dart';

class DiagnosisResultScreen extends StatefulWidget {
  final List<DiseaseModel> diseases;
  final List<int> selectedSymptomIds;
  final Map<String, dynamic> vitalParameters;
  final List<SymptomModel>? selectedSymptoms;
  final dynamic prescription; // Added prescription parameter

  const DiagnosisResultScreen({
    Key? key,
    required this.diseases,
    required this.selectedSymptomIds,
    required this.vitalParameters,
    this.selectedSymptoms,
    this.prescription, // Added prescription parameter to constructor
  }) : super(key: key);

  @override
  State<DiagnosisResultScreen> createState() => _DiagnosisResultScreenState();
}

class _DiagnosisResultScreenState extends State<DiagnosisResultScreen> with SingleTickerProviderStateMixin {
  final MedicalService _medicalService = MedicalService();
  late TabController _tabController;
  bool _isLoadingResearch = false;
  bool _isLoadingStats = false;
  bool _isSavingDiagnosis = false;
  bool _diagnosisSaved = false;
  String? _saveErrorMessage;
  
  // Research and risk data
  List<Map<String, dynamic>> _researchData = [];
  List<Map<String, dynamic>> _riskFactors = [];
  Map<String, dynamic> _diseaseStats = {};
  
  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 4, vsync: this); // Changed from 3 to 4 tabs
    
    // Load research data if online research is enabled
    if (AppConfig.enableMedicalResearch) {
      _loadResearchData();
      _loadRiskFactors();
    }
    
    // Save diagnosis history automatically when screen opens
    _saveDiagnosisToHistory();
  }
  
  Future<void> _loadResearchData() async {
    setState(() {
      _isLoadingResearch = true;
    });
    
    try {
      // Get research data from the service
      final researchData = _medicalService.getLatestResearch();
      
      setState(() {
        _researchData = researchData;
        _isLoadingResearch = false;
      });
    } catch (e) {
      setState(() {
        _isLoadingResearch = false;
      });
    }
  }
  
  Future<void> _loadRiskFactors() async {
    try {
      // Get risk factors from the service
      final riskFactors = _medicalService.getRiskFactors();
      
      setState(() {
        _riskFactors = riskFactors;
      });
    } catch (e) {
      // Handle error
    }
  }
  
  Future<void> _loadDiseaseStatistics(String diseaseName) async {
    setState(() {
      _isLoadingStats = true;
    });
    
    try {
      // Use the MedicalApiService to get global statistics
      final stats = await _medicalService.getDiseaseStatistics(diseaseName);
      
      setState(() {
        _diseaseStats = stats;
        _isLoadingStats = false;
      });
    } catch (e) {
      setState(() {
        _isLoadingStats = false;
      });
    }
  }
  
  Future<void> _saveDiagnosisToHistory() async {
    // Only save if we have at least one disease and we're not in history viewing mode
    if (widget.diseases.isEmpty || widget.selectedSymptomIds.isEmpty) {
      print('Not saving diagnosis: No diseases or empty symptom IDs');
      return;
    }

    // Check if already saved
    if (_diagnosisSaved) return;
    
    setState(() {
      _isSavingDiagnosis = true;
      _saveErrorMessage = null;
    });
    
    try {
      final user = FirebaseAuth.instance.currentUser;
      if (user == null) {
        throw Exception('Not signed in');
      }
      
      // Get the primary disease (highest confidence)
      final primaryDisease = widget.diseases.first;
      
      // Prepare symptom data
      List<Map<String, dynamic>> symptomsData = [];
      
      // If we have the actual symptom models, use those
      if (widget.selectedSymptoms != null && widget.selectedSymptoms!.isNotEmpty) {
        symptomsData = widget.selectedSymptoms!.map((s) => {
          'id': s.id,
          'name': s.name,
        }).toList();
      } 
      // Otherwise use just the IDs
      else {
        symptomsData = widget.selectedSymptomIds.map((id) => {
          'id': id,
          'name': 'Symptom $id',
        }).toList();
      }
      
      // Create the diagnosis document
      final diagnosisData = {
        'userId': user.uid,
        'diseaseId': primaryDisease.id.toString(),
        'diseaseName': primaryDisease.name,
        'confidence': primaryDisease.accuracy,
        'specialisations': primaryDisease.specialisations,
        'description': primaryDisease.description,
        'symptoms': symptomsData,
        'advice': primaryDisease.treatmentDescription,
        'vitalParameters': widget.vitalParameters,
        'timestamp': FieldValue.serverTimestamp(),
      };
      
      // Save to Firestore with multiple attempts
      try {
        // First attempt with transaction
        await FirebaseFirestore.instance.runTransaction((transaction) async {
          // Create a reference to a new diagnosis document
          final diagnosisRef = FirebaseFirestore.instance
              .collection('users')
              .doc(user.uid)
              .collection('diagnoses')
              .doc();
          
          // Add document in transaction
          transaction.set(diagnosisRef, diagnosisData);
        });
      } catch (firstAttemptError) {
        print('First diagnosis save attempt failed: $firstAttemptError');
        
        // Second attempt with direct set
        try {
          await FirebaseFirestore.instance
              .collection('users')
              .doc(user.uid)
              .collection('diagnoses')
              .add(diagnosisData);
        } catch (secondAttemptError) {
          print('Second diagnosis save attempt failed: $secondAttemptError');
          throw secondAttemptError;
        }
      }
      
      // Update state to reflect successful save
      if (mounted) {
        setState(() {
          _diagnosisSaved = true;
          _isSavingDiagnosis = false;
        });
      }
    } catch (e) {
      print('Error saving diagnosis history: $e');
      if (mounted) {
        setState(() {
          _isSavingDiagnosis = false;
          _saveErrorMessage = 'Failed to save diagnosis to history: $e';
        });
      }
    }
  }
  
  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Diagnosis Results'),
        elevation: 0,
        bottom: TabBar(
          controller: _tabController,
          isScrollable: true, // Allow tabs to scroll
          tabs: const [
            Tab(text: 'DIAGNOSIS'),
            Tab(text: 'PRESCRIPTION'),
            Tab(text: 'RESEARCH'),
            Tab(text: 'RISK FACTORS'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _buildDiagnosisTab(),
          _buildPrescriptionTab(),
          _buildResearchTab(),
          _buildRiskFactorsTab(),
        ],
      ),
    );
  }
  
  Widget _buildDiagnosisTab() {
    return ListView(
      padding: const EdgeInsets.all(16.0),
      children: [
        Card(
          elevation: 4,
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'AI-Powered Diagnosis',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 16),
                const Text(
                  'Based on your symptoms and vital signs, our AI algorithm suggests the following possible conditions:',
                  style: TextStyle(fontSize: 15),
                ),
                const SizedBox(height: 16),
                ...widget.diseases.map((disease) => _buildDiseaseCard(disease)).toList(),
                const SizedBox(height: 16),
                const Text(
                  'IMPORTANT: This is not a replacement for professional medical advice. Please consult a healthcare provider for accurate diagnosis and treatment.',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: Colors.red,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
  
  Widget _buildDiseaseCard(DiseaseModel disease) {
    // Calculate confidence percentage
    final confidencePercent = (disease.accuracy * 100).toStringAsFixed(1);
    
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8.0),
      child: ExpansionTile(
        title: Row(
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
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
              decoration: BoxDecoration(
                color: _getConfidenceColor(disease.accuracy),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                '$confidencePercent%',
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        ),
        subtitle: Text('Specialists: ${disease.specialisations.join(", ")}'),
        children: [
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Description:',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                Text(disease.description),
                const SizedBox(height: 16),
                const Text(
                  'Recommended Treatments:',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                ...disease.treatmentDescription.map((treatment) => 
                  Padding(
                    padding: const EdgeInsets.only(bottom: 4.0),
                    child: Row(
                      children: [
                        const Icon(Icons.check_circle, color: Colors.green, size: 16),
                        const SizedBox(width: 8),
                        Expanded(child: Text(treatment)),
                      ],
                    ),
                  )
                ).toList(),
                const SizedBox(height: 16),
                ElevatedButton(
                  onPressed: () => _loadDiseaseStatistics(disease.name),
                  child: const Text('View Global Statistics'),
                ),
                if (_isLoadingStats)
                  const Padding(
                    padding: EdgeInsets.symmetric(vertical: 16.0),
                    child: Center(
                      child: SpinKitCircle(
                        color: Colors.blue,
                        size: 40.0,
                      ),
                    ),
                  )
                else if (_diseaseStats.isNotEmpty)
                  _buildStatisticsSection(),
              ],
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildStatisticsSection() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 16.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Global Statistics',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          _buildStatItem('Prevalence', _diseaseStats['prevalence'] ?? 'Unknown'),
          _buildStatItem('Annual Cases', _diseaseStats['annualCases'] ?? 'Unknown'),
          _buildStatItem('Mortality Rate', _diseaseStats['mortalityRate'] ?? 'Unknown'),
          _buildStatItem('Treatment Success Rate', _diseaseStats['treatmentSuccess'] ?? 'Unknown'),
          const SizedBox(height: 8),
          Text(
            'Source: ${_diseaseStats['sourceInfo'] ?? 'Medical database'}',
            style: const TextStyle(
              fontStyle: FontStyle.italic,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildStatItem(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 140,
            child: Text(
              '$label:',
              style: const TextStyle(fontWeight: FontWeight.w500),
            ),
          ),
          Expanded(
            child: Text(value),
          ),
        ],
      ),
    );
  }
  
  Widget _buildResearchTab() {
    if (!AppConfig.enableMedicalResearch) {
      return const Center(
        child: Text('Online medical research is currently disabled'),
      );
    }
    
    if (_isLoadingResearch) {
      return const Center(
        child: SpinKitCircle(
          color: Colors.blue,
          size: 50.0,
        ),
      );
    }
    
    if (_researchData.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text('No research data available for these symptoms'),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _loadResearchData,
              child: const Text('Refresh'),
            ),
          ],
        ),
      );
    }
    
    return ListView.builder(
      padding: const EdgeInsets.all(16.0),
      itemCount: _researchData.length,
      itemBuilder: (context, index) {
        final research = _researchData[index];
        return ResearchCard(
          research: research,
          onTap: () => _launchUrl(research['url']),
        );
      },
    );
  }
  
  Widget _buildRiskFactorsTab() {
    if (_riskFactors.isEmpty) {
      return const Center(
        child: Text('No specific risk factors identified'),
      );
    }
    
    return ListView.builder(
      padding: const EdgeInsets.all(16.0),
      itemCount: _riskFactors.length,
      itemBuilder: (context, index) {
        final riskFactor = _riskFactors[index];
        return RiskFactorCard(riskFactor: riskFactor);
      },
    );
  }
  
  Color _getConfidenceColor(double accuracy) {
    if (accuracy >= 0.7) {
      return Colors.green;
    } else if (accuracy >= 0.4) {
      return Colors.orange;
    } else {
      return Colors.red;
    }
  }
  
  Future<void> _launchUrl(String url) async {
    if (await canLaunchUrl(Uri.parse(url))) {
      await launchUrl(Uri.parse(url));
    }
  }
  
  Widget _buildPrescriptionTab() {
    if (widget.prescription == null) {
      return const Center(
        child: Text('No prescription available for this diagnosis'),
      );
    }
    
    return ListView(
      padding: const EdgeInsets.all(16.0),
      children: [
        Card(
          elevation: 4,
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Icon(Icons.medical_services, color: Colors.blue, size: 28),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Prescription for ${widget.prescription.diseaseName}',
                        style: const TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
                const Divider(height: 32),
                
                // Medications section
                const Text(
                  'Medications',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.blue,
                  ),
                ),
                const SizedBox(height: 12),
                
                // List all medications
                ...widget.prescription.medications.map((med) => _buildMedicationCard(med)).toList(),
                
                const SizedBox(height: 24),
                
                // Advice section
                const Text(
                  'Medical Advice',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.blue,
                  ),
                ),
                const SizedBox(height: 12),
                
                // List all advice
                ...widget.prescription.advices.map((advice) => _buildAdviceItem(advice)).toList(),
                
                const SizedBox(height: 20),
                const Divider(),
                const SizedBox(height: 10),
                
                // Disclaimer and generated date
                Text(
                  'Generated on: ${_formatDateTime(widget.prescription.createdAt)}',
                  style: TextStyle(
                    fontSize: 13,
                    color: Colors.grey.shade700,
                    fontStyle: FontStyle.italic,
                  ),
                ),
                const SizedBox(height: 8),
                const Text(
                  'DISCLAIMER: This AI-generated prescription is for informational purposes only and does not substitute professional medical advice. Please consult a healthcare provider before starting any medication.',
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.red,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
  
  Widget _buildMedicationCard(dynamic medication) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              medication.name,
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Colors.blue,
              ),
            ),
            const SizedBox(height: 8),
            _buildMedicationDetail('Dosage', medication.dosage),
            _buildMedicationDetail('Frequency', medication.frequency),
            _buildMedicationDetail('Duration', '${medication.durationDays} days'),
            if (medication.notes != null && medication.notes.isNotEmpty)
              _buildMedicationDetail('Notes', medication.notes),
          ],
        ),
      ),
    );
  }
  
  Widget _buildMedicationDetail(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 90,
            child: Text(
              '$label:',
              style: const TextStyle(
                fontWeight: FontWeight.w500,
                color: Colors.grey,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(
                color: Colors.black87,
              ),
            ),
          ),
        ],
      ),
    );
  }
  
  Widget _buildAdviceItem(String advice) {
    bool isHighlighted = advice.contains('IMPORTANT') || 
                         advice.contains('RESEARCH') || 
                         advice.contains('HEALTH FACTOR');
                         
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
      margin: const EdgeInsets.only(bottom: 8),
      decoration: BoxDecoration(
        color: isHighlighted ? Colors.amber.shade50 : Colors.grey.shade50,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(
          color: isHighlighted ? Colors.amber.shade200 : Colors.grey.shade200,
        ),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            isHighlighted ? Icons.priority_high : Icons.check_circle_outline,
            size: 16,
            color: isHighlighted ? Colors.amber.shade800 : Colors.green,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              advice,
              style: TextStyle(
                color: isHighlighted ? Colors.amber.shade900 : Colors.black87,
                fontWeight: isHighlighted ? FontWeight.w500 : FontWeight.normal,
              ),
            ),
          ),
        ],
      ),
    );
  }
  
  String _formatDateTime(DateTime dateTime) {
    final months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    return '${months[dateTime.month - 1]} ${dateTime.day}, ${dateTime.year} at ${dateTime.hour}:${dateTime.minute.toString().padLeft(2, '0')}';
  }
}