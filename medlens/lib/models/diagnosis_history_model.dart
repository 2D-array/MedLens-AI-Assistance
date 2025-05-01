import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:medlens/models/prescription_model.dart';

class DiagnosisHistoryModel {
  final String id;
  final String diseaseName;
  final int diseaseId;
  final double confidence;
  final DateTime timestamp;
  final List<Map<String, dynamic>> symptoms;
  final List<String> specialisations;
  final String description;
  final List<MedicationModel> medications;
  final List<String> advice;

  DiagnosisHistoryModel({
    required this.id,
    required this.diseaseName,
    required this.diseaseId,
    required this.confidence,
    required this.timestamp,
    required this.symptoms,
    required this.specialisations,
    required this.description,
    required this.medications,
    required this.advice,
  });

  factory DiagnosisHistoryModel.fromFirestore(DocumentSnapshot doc) {
    final data = doc.data() as Map<String, dynamic>;
    
    // Convert Firestore timestamp to DateTime
    final Timestamp firestoreTimestamp = data['timestamp'] as Timestamp? ?? 
        Timestamp.fromDate(DateTime.now());
    
    // Parse medications
    final List<MedicationModel> medicationsList = [];
    if (data['medications'] != null) {
      for (var med in data['medications']) {
        medicationsList.add(
          MedicationModel(
            name: med['name'] ?? '',
            dosage: med['dosage'] ?? '',
            frequency: med['frequency'] ?? '',
            durationDays: med['durationDays'] ?? 0,
            notes: med['notes'] ?? '',
          ),
        );
      }
    }

    return DiagnosisHistoryModel(
      id: doc.id,
      diseaseName: data['diseaseName'] ?? 'Unknown Disease',
      diseaseId: data['diseaseId'] ?? 0,
      confidence: (data['confidence'] ?? 0.0).toDouble(),
      timestamp: firestoreTimestamp.toDate(),
      symptoms: List<Map<String, dynamic>>.from(data['symptoms'] ?? []),
      specialisations: List<String>.from(data['specialisations'] ?? []),
      description: data['description'] ?? '',
      medications: medicationsList,
      advice: List<String>.from(data['advice'] ?? []),
    );
  }
}