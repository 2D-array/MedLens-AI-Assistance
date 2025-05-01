import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:medlens/models/disease_model.dart';
import 'package:medlens/models/prescription_model.dart';
import 'package:medlens/utils/app_theme.dart';
import 'package:medlens/screens/diagnosis_result_screen.dart';
import 'package:intl/intl.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({Key? key}) : super(key: key);

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  bool _isLoading = true;
  bool _hasError = false;
  String? _errorMessage;
  List<Map<String, dynamic>> _diagnosisHistory = [];

  @override
  void initState() {
    super.initState();
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    try {
      print('🔍 DEBUG: Starting to load diagnosis history');
      setState(() {
        _isLoading = true;
        _hasError = false;
      });

      final user = FirebaseAuth.instance.currentUser;
      if (user == null) {
        print('❌ ERROR: No user logged in when trying to load history');
        throw Exception('No user logged in');
      }
      
      print('🔍 DEBUG: User authenticated: ${user.uid}');
      print('🔍 DEBUG: Attempting to fetch from Firestore path: users/${user.uid}/diagnoses');

      // Get diagnosis history from Firestore
      final snapshot = await FirebaseFirestore.instance
          .collection('users')
          .doc(user.uid)
          .collection('diagnoses')
          .orderBy('timestamp', descending: true)
          .get();

      print('🔍 DEBUG: Firestore query completed. Found ${snapshot.docs.length} diagnoses');
      
      final history = snapshot.docs.map((doc) {
        final data = doc.data();
        print('🔍 DEBUG: Processing document ${doc.id} - Disease: ${data['diseaseName']}');
        return {
          ...data,
          'id': doc.id,
        };
      }).toList();

      if (mounted) {
        setState(() {
          _diagnosisHistory = history;
          _isLoading = false;
        });
        print('🔍 DEBUG: Successfully updated UI with ${history.length} diagnoses');
      }
    } catch (e) {
      print('❌ ERROR loading diagnosis history: $e');
      if (mounted) {
        setState(() {
          _hasError = true;
          _errorMessage = 'Failed to load diagnosis history: ${e.toString()}';
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Diagnosis History'),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _hasError
              ? _buildErrorView()
              : _diagnosisHistory.isEmpty
                  ? _buildEmptyView()
                  : _buildHistoryList(),
    );
  }

  Widget _buildErrorView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(
              Icons.error_outline,
              size: 60,
              color: Colors.red,
            ),
            const SizedBox(height: 16),
            Text(
              _errorMessage ?? 'Something went wrong',
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 16),
            ),
            const SizedBox(height: 24),
            ElevatedButton(
              onPressed: _loadHistory,
              child: const Text('Try Again'),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEmptyView() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(
              Icons.history,
              size: 80,
              color: Colors.grey,
            ),
            const SizedBox(height: 24),
            const Text(
              'No Diagnosis History',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'Your previous diagnoses will appear here once you use the symptom checker.',
              textAlign: TextAlign.center,
              style: TextStyle(color: AppTheme.textSecondaryColor),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHistoryList() {
    return ListView.separated(
      padding: const EdgeInsets.all(16),
      itemCount: _diagnosisHistory.length,
      separatorBuilder: (context, index) => const SizedBox(height: 12),
      itemBuilder: (context, index) {
        final diagnosis = _diagnosisHistory[index];
        
        // Extract symptom names from the list of maps
        List<String> symptomNames = [];
        if (diagnosis['symptoms'] != null) {
          final symptomsList = diagnosis['symptoms'] as List<dynamic>;
          for (var symptom in symptomsList) {
            if (symptom is Map<String, dynamic> && symptom.containsKey('name')) {
              symptomNames.add(symptom['name'].toString());
            }
          }
        }
        
        // Create disease model with correct parameters matching the constructor
        final disease = DiseaseModel(
          id: int.tryParse(diagnosis['diseaseId']?.toString() ?? '0') ?? 0,
          name: diagnosis['diseaseName'] ?? 'Unknown Disease',
          accuracy: diagnosis['confidence'] ?? 0.0,
          specialisations: List<String>.from(diagnosis['specialisations'] ?? []),
          description: diagnosis['description'] ?? '',
          // Use the extracted symptom names instead
          possibleSymptoms: symptomNames,
          treatmentDescription: List<String>.from(diagnosis['advice'] ?? []),
        );
        
        // Create a proper PrescriptionModel
        final prescription = PrescriptionModel(
          diseaseName: disease.name,
          diseaseId: disease.id,
          accuracy: disease.accuracy,
          medications: [],  // No detailed medication info in history
          advices: List<String>.from(diagnosis['advice'] ?? []),
        );
        
        final timestamp = diagnosis['timestamp'] as Timestamp?;
        final dateString = timestamp != null
            ? DateFormat('MMM d, yyyy • h:mm a').format(timestamp.toDate())
            : 'Unknown date';

        return Card(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          child: InkWell(
            onTap: () {
              // Navigate to detailed view of this diagnosis
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => DiagnosisResultScreen(
                    diseases: [disease],
                    selectedSymptomIds: [], // Use empty list for symptomIds
                    vitalParameters: {}, // Use empty map for vitalParameters
                  ),
                ),
              );
            },
            borderRadius: BorderRadius.circular(12),
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Title and date
                  Row(
                    children: [
                      Expanded(
                        child: Text(
                          disease.name,
                          style: const TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 18,
                          ),
                        ),
                      ),
                      Text(
                        '${(disease.accuracy * 100).toStringAsFixed(0)}%',
                        style: TextStyle(
                          color: _getProbabilityColor(disease.accuracy),
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    dateString,
                    style: TextStyle(
                      color: Colors.grey[600],
                      fontSize: 12,
                    ),
                  ),
                  const SizedBox(height: 12),
                  
                  // Symptoms
                  Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    children: symptomNames.map((symptomName) {
                      return Chip(
                        materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                        label: Text(
                          symptomName,
                          style: const TextStyle(fontSize: 12),
                        ),
                        padding: EdgeInsets.zero,
                        backgroundColor: Colors.blue[50],
                        visualDensity: VisualDensity.compact,
                      );
                    }).toList(),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Color _getProbabilityColor(double accuracy) {
    if (accuracy >= 0.7) {
      return Colors.red;
    } else if (accuracy >= 0.4) {
      return Colors.orange;
    } else {
      return Colors.green;
    }
  }
}