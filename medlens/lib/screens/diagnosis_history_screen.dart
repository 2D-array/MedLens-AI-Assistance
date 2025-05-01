import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:intl/intl.dart';
import 'package:medlens/models/diagnosis_history_model.dart';
import 'package:medlens/utils/app_theme.dart';
import 'package:medlens/widgets/custom_button.dart';

class DiagnosisHistoryScreen extends StatefulWidget {
  const DiagnosisHistoryScreen({Key? key}) : super(key: key);

  @override
  State<DiagnosisHistoryScreen> createState() => _DiagnosisHistoryScreenState();
}

class _DiagnosisHistoryScreenState extends State<DiagnosisHistoryScreen> {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;
  bool _isLoading = true;
  String? _errorMessage;
  List<DiagnosisHistoryModel> _diagnosisHistory = [];

  @override
  void initState() {
    super.initState();
    _fetchDiagnosisHistory();
  }

  Future<void> _fetchDiagnosisHistory() async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      print('🔍 DEBUG: Starting to fetch diagnosis history from Firestore');
      final user = _auth.currentUser;
      if (user == null) {
        throw Exception('No user logged in');
      }
      print('🔍 DEBUG: User authenticated: ${user.uid}');

      // Query the 'diagnosisHistory' collection for the current user
      final QuerySnapshot snapshot = await _firestore
          .collection('users')
          .doc(user.uid)
          .collection('diagnosisHistory')
          .orderBy('timestamp', descending: true)
          .get();

      print('🔍 DEBUG: Fetched ${snapshot.docs.length} diagnoses from Firestore');

      final List<DiagnosisHistoryModel> history = snapshot.docs
          .map((doc) => DiagnosisHistoryModel.fromFirestore(doc))
          .toList();

      setState(() {
        _diagnosisHistory = history;
        _isLoading = false;
      });

      print('🔍 DEBUG: Diagnosis history loaded successfully');
      // Print first item details for debugging
      if (_diagnosisHistory.isNotEmpty) {
        print('🔍 DEBUG: First diagnosis: ${_diagnosisHistory[0].diseaseName} (${_diagnosisHistory[0].confidence.toStringAsFixed(1)}%)');
      }

    } catch (e) {
      print('❌ ERROR: Failed to fetch diagnosis history: $e');
      setState(() {
        _errorMessage = 'Failed to load diagnosis history: $e';
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDarkMode = Theme.of(context).brightness == Brightness.dark;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Diagnosis History'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _fetchDiagnosisHistory,
            tooltip: 'Refresh',
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _errorMessage != null
              ? _buildErrorView()
              : _diagnosisHistory.isEmpty
                  ? _buildEmptyView()
                  : _buildHistoryList(isDarkMode),
    );
  }

  Widget _buildErrorView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, size: 64, color: Colors.red),
            const SizedBox(height: 16),
            Text(
              'Error',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            Text(
              _errorMessage ?? 'An unknown error occurred',
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            CustomButton(
              onPressed: _fetchDiagnosisHistory,
              text: 'Try Again',
              width: 150,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEmptyView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.history, size: 64, color: Theme.of(context).primaryColor),
            const SizedBox(height: 16),
            Text(
              'No Diagnosis History',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            const Text(
              'Your diagnosis history will appear here once you have completed a diagnosis.',
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHistoryList(bool isDarkMode) {
    return ListView.builder(
      padding: const EdgeInsets.all(16),
      itemCount: _diagnosisHistory.length,
      itemBuilder: (context, index) {
        final diagnosis = _diagnosisHistory[index];
        final date = DateFormat('MMM dd, yyyy').format(diagnosis.timestamp);
        final time = DateFormat('hh:mm a').format(diagnosis.timestamp);

        // Create a list of symptoms
        final symptomsText = diagnosis.symptoms.isEmpty
            ? 'No symptoms recorded'
            : diagnosis.symptoms.map((s) => s['name']).join(', ');

        return Card(
          margin: const EdgeInsets.only(bottom: 16),
          elevation: 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          child: ExpansionTile(
            title: Row(
              children: [
                CircleAvatar(
                  backgroundColor: AppTheme.getConfidenceColor(diagnosis.confidence),
                  child: Text(
                    '${diagnosis.confidence.toInt()}%',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        diagnosis.diseaseName,
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        '$date at $time',
                        style: TextStyle(
                          fontSize: 12,
                          color: isDarkMode ? Colors.white70 : Colors.black54,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            childrenPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            children: [
              // Symptoms
              _buildExpandedSection(
                'Symptoms',
                symptomsText,
                Icons.healing,
              ),
              const SizedBox(height: 8),

              // Medications
              _buildExpandedSection(
                'Medications',
                diagnosis.medications.isEmpty
                    ? 'No medications prescribed'
                    : diagnosis.medications.map((med) => '${med.name}: ${med.dosage}').join('\n'),
                Icons.medication,
              ),
              const SizedBox(height: 8),

              // Description
              if (diagnosis.description.isNotEmpty) ...[
                _buildExpandedSection(
                  'Description',
                  diagnosis.description,
                  Icons.description,
                ),
                const SizedBox(height: 8),
              ],

              // Specializations
              _buildExpandedSection(
                'Consult Specialists',
                diagnosis.specialisations.isEmpty
                    ? 'No specialist consultation required'
                    : diagnosis.specialisations.join(', '),
                Icons.medical_services,
              ),

              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  TextButton.icon(
                    icon: const Icon(Icons.print),
                    label: const Text('Print'),
                    onPressed: () {
                      print('🔍 DEBUG: Print report requested for diagnosis: ${diagnosis.id}');
                      // Implement PDF generation and printing
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text('Printing not implemented yet')),
                      );
                    },
                  ),
                  const SizedBox(width: 8),
                  TextButton.icon(
                    icon: const Icon(Icons.share),
                    label: const Text('Share'),
                    onPressed: () {
                      print('🔍 DEBUG: Share requested for diagnosis: ${diagnosis.id}');
                      // Implement sharing functionality
                      ScaffoldMessenger.of(context).showSnackBar(
                        const SnackBar(content: Text('Sharing not implemented yet')),
                      );
                    },
                  ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildExpandedSection(String title, String content, IconData icon) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 18, color: Theme.of(context).primaryColor),
        const SizedBox(width: 8),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 14,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                content,
                style: const TextStyle(fontSize: 14),
              ),
            ],
          ),
        ),
      ],
    );
  }
}