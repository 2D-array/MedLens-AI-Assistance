import 'package:uuid/uuid.dart';

class PrescriptionModel {
  final String id;
  final String diseaseName;
  final int diseaseId;
  final double accuracy;
  final List<Medication> medications;
  final List<String> advices;
  final DateTime createdAt;

  PrescriptionModel({
    String? id,
    required this.diseaseName,
    required this.diseaseId,
    required this.accuracy,
    required this.medications,
    required this.advices,
    DateTime? createdAt,
  })  : id = id ?? const Uuid().v4(),
        createdAt = createdAt ?? DateTime.now();

  factory PrescriptionModel.fromJson(Map<String, dynamic> json) {
    return PrescriptionModel(
      id: json['id'] ?? const Uuid().v4(),
      diseaseName: json['diseaseName'] ?? '',
      diseaseId: json['diseaseId'] ?? 0,
      accuracy: json['accuracy'] ?? 0.0,
      medications: (json['medications'] as List?)
              ?.map((med) => Medication.fromJson(med))
              .toList() ??
          [],
      advices: List<String>.from(json['advices'] ?? []),
      createdAt: json['createdAt'] != null
          ? DateTime.parse(json['createdAt'])
          : DateTime.now(),
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'diseaseName': diseaseName,
      'diseaseId': diseaseId,
      'accuracy': accuracy,
      'medications': medications.map((med) => med.toJson()).toList(),
      'advices': advices,
      'createdAt': createdAt.toIso8601String(),
    };
  }
}

class Medication {
  final String name;
  final String dosage;
  final String frequency;
  final int durationDays;
  final String? notes;

  Medication({
    required this.name,
    required this.dosage,
    required this.frequency,
    required this.durationDays,
    this.notes,
  });

  factory Medication.fromJson(Map<String, dynamic> json) {
    return Medication(
      name: json['name'] ?? '',
      dosage: json['dosage'] ?? '',
      frequency: json['frequency'] ?? '',
      durationDays: json['durationDays'] ?? 7,
      notes: json['notes'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'dosage': dosage,
      'frequency': frequency,
      'durationDays': durationDays,
      'notes': notes,
    };
  }
}