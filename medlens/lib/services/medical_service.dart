import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:medlens/models/disease_model.dart';
import 'package:medlens/models/prescription_model.dart';
import 'package:medlens/models/symptom_model.dart';
import 'package:medlens/services/medical_api_service.dart';
import 'package:medlens/utils/app_config.dart';
import 'package:medlens/utils/logger.dart';
import 'dart:math' as math;

class MedicalService {
  final Logger _logger = Logger();
  final String _baseUrl = AppConfig.medicalApiEndpoint;
  final String _apiKey = AppConfig.medicalApiKey;
  final String _researchUrl = AppConfig.researchApiEndpoint;
  final String _researchApiKey = AppConfig.researchApiKey;
  
  // Singleton implementation
  static final MedicalService _instance = MedicalService._internal();
  
  factory MedicalService() {
    return _instance;
  }
  
  MedicalService._internal();
  
  // Initialize our API service for online diagnosis
  final MedicalApiService _apiService = MedicalApiService();
  
  // Get all symptoms - using reliable mock data
  Future<List<SymptomModel>> getSymptoms() async {
    // Short delay to simulate network request
    await Future.delayed(const Duration(milliseconds: 800));
    return _getMockSymptoms();
  }
  
  // Enhanced diagnosis method with internet-based medical AI analysis
  Future<List<DiseaseModel>> getDiagnosis(
      List<int> symptomIds, 
      Map<String, dynamic> vitalParameters) async {
    try {
      // First, get advanced online medical analysis
      final medicalData = await _apiService.searchMedicalData(symptomIds, vitalParameters);
      
      // Convert the API results to disease models
      final onlineDiagnoses = _apiService.convertToDiseaseModels(medicalData);
      
      // If online diagnosis was successful, use it
      if (onlineDiagnoses.isNotEmpty) {
        // Store research data for later display
        _storeResearchData(medicalData['researchData'] ?? []);
        _storeRiskFactors(medicalData['riskFactors'] ?? []);
        
        return onlineDiagnoses;
      }
      
      // Fallback to offline diagnosis if online failed
      return _getAdvancedDiagnosis(symptomIds, vitalParameters);
    } catch (e) {
      print('Error getting diagnosis: $e');
      // Fallback to reliable offline diagnosis
      return _getAdvancedDiagnosis(symptomIds, vitalParameters);
    }
  }
  
  // Store research data for later retrieval
  List<Map<String, dynamic>> _latestResearchData = [];
  List<Map<String, dynamic>> _latestRiskFactors = [];
  
  void _storeResearchData(List<dynamic> researchData) {
    _latestResearchData = List<Map<String, dynamic>>.from(researchData);
  }
  
  void _storeRiskFactors(List<dynamic> riskFactors) {
    _latestRiskFactors = List<Map<String, dynamic>>.from(riskFactors);
  }
  
  // Get relevant medical research for current diagnosis
  List<Map<String, dynamic>> getLatestResearch() {
    return _latestResearchData;
  }
  
  // Get patient-specific risk factors
  List<Map<String, dynamic>> getRiskFactors() {
    return _latestRiskFactors;
  }
  
  // Enhanced prescription generator that takes vital parameters and research into account
  Future<PrescriptionModel> generatePrescription(
      DiseaseModel disease, 
      Map<String, dynamic> vitalParameters,
      List<SymptomModel> selectedSymptoms) async {
    
    // For demonstration, generating a detailed prescription based on the disease
    // and vital parameters like age, weight, temperature, etc.
    List<Medication> medications = [];
    List<String> advices = [];
    
    // Get age from birth year
    final int currentYear = DateTime.now().year;
    final int birthYear = vitalParameters['birthYear'] ?? (currentYear - 30);
    final int age = currentYear - birthYear;
    final String gender = vitalParameters['gender'] ?? 'male';
    final double? temperature = vitalParameters['temperature'];
    final double? weight = vitalParameters['weight'];
    final double? height = vitalParameters['height'];
    
    // Common cold and flu treatment prescriptions
    if (disease.name.toLowerCase().contains('cold') || 
        disease.name.toLowerCase().contains('flu')) {
      
      String paracetamolDosage = '500mg';
      if (age < 12) {
        paracetamolDosage = '250mg';
      } else if (weight != null) {
        if (weight < 50) {
          paracetamolDosage = '325mg';
        } else if (weight > 80) {
          paracetamolDosage = '650mg';
        }
      }
      
      bool hasFever = temperature != null && temperature > 100.4;
      bool hasProductiveCough = _hasSymptom(selectedSymptoms, ['productive cough', 'phlegm', 'mucus']);
      bool hasDryCough = _hasSymptom(selectedSymptoms, ['dry cough', 'cough']);
      bool hasSoreThroat = _hasSymptom(selectedSymptoms, ['sore throat']);
      bool hasNasalCongestion = _hasSymptom(selectedSymptoms, ['nasal congestion', 'stuffy nose']);
      
      // Core medications for cold/flu
      medications = [
        Medication(
          name: 'Paracetamol (Acetaminophen)',
          dosage: paracetamolDosage,
          frequency: 'Every 6 hours as needed for fever or pain',
          durationDays: 5,
          notes: 'Do not exceed 4g per day. Take with food to reduce stomach irritation.',
        ),
      ];
      
      // Anti-inflammatory for fever and pain
      if (hasFever || hasSoreThroat) {
        medications.add(
          Medication(
            name: 'Ibuprofen',
            dosage: weight != null && weight > 60 ? '400mg' : '200mg',
            frequency: 'Every 8 hours as needed for fever or pain',
            durationDays: 3,
            notes: 'Take with food. Do not take if you have stomach ulcers or kidney problems.',
          )
        );
      }
      
      // Expectorant for productive cough
      if (hasProductiveCough) {
        medications.add(
          Medication(
            name: 'Guaifenesin (Expectorant)',
            dosage: '200-400mg',
            frequency: 'Every 4 hours as needed',
            durationDays: 7,
            notes: 'Take with plenty of water. Helps thin and loosen mucus.',
          )
        );
      }
      
      // Cough suppressant for dry cough
      if (hasDryCough) {
        medications.add(
          Medication(
            name: 'Dextromethorphan',
            dosage: '15-30mg',
            frequency: 'Every 6-8 hours as needed for cough',
            durationDays: 5,
            notes: 'Avoid if you have high blood pressure or take MAO inhibitors.',
          )
        );
      }
      
      // Antihistamine for nasal congestion and runny nose
      if (hasNasalCongestion) {
        medications.add(
          Medication(
            name: 'Cetirizine',
            dosage: '10mg',
            frequency: 'Once daily',
            durationDays: 5,
            notes: 'May cause drowsiness. Take in the evening if it makes you sleepy.',
          )
        );
        
        medications.add(
          Medication(
            name: 'Pseudoephedrine (Decongestant)',
            dosage: '60mg',
            frequency: 'Every 6 hours as needed for nasal congestion',
            durationDays: 3,
            notes: 'Do not take if you have hypertension. May cause insomnia if taken late in the day.',
          )
        );
      }
      
      // Antibiotics only if there's a high suspicion of bacterial infection
      if (hasFever && temperature! > 101.5 && (hasProductiveCough || hasSoreThroat)) {
        medications.add(
          Medication(
            name: 'Azithromycin',
            dosage: '500mg',
            frequency: 'Once daily',
            durationDays: 3,
            notes: 'Take 1 hour before or 2 hours after meals. Complete the full course even if you feel better.',
          )
        );
      }
      
      // Zinc and vitamin C as supportive care
      medications.add(
        Medication(
          name: 'Zinc',
          dosage: '15-30mg',
          frequency: 'Once daily',
          durationDays: 5,
          notes: 'Take with food to avoid stomach upset. May help reduce duration of cold symptoms.',
        )
      );
      
      medications.add(
        Medication(
          name: 'Vitamin C',
          dosage: '500mg',
          frequency: 'Twice daily',
          durationDays: 7,
          notes: 'May help support immune function during illness.',
        )
      );
      
      advices = [
        'Rest adequately to allow your body to recover',
        'Stay hydrated by drinking at least 8-10 glasses of water daily',
        'Use a humidifier to ease congestion and sore throat',
        'Gargle with warm salt water for sore throat relief',
        'Avoid close contact with others to prevent spreading infection',
        'Consider using saline nasal spray to relieve nasal congestion'
      ];
    } 
    
    // Strep throat or bacterial pharyngitis
    else if (disease.name.toLowerCase().contains('pharyngitis') || 
             disease.name.toLowerCase().contains('tonsillitis') ||
             disease.name.toLowerCase().contains('strep throat')) {
      
      String paracetamolDosage = '500mg';
      if (age < 12) {
        paracetamolDosage = '250mg';
      } else if (weight != null) {
        if (weight < 50) {
          paracetamolDosage = '325mg';
        } else if (weight > 80) {
          paracetamolDosage = '650mg';
        }
      }
      
      medications = [
        // Antibiotics for bacterial infections
        Medication(
          name: 'Amoxicillin',
          dosage: age >= 12 ? '500mg' : '250mg',
          frequency: 'Every 8 hours',
          durationDays: 10,
          notes: 'Take for the full course even if symptoms improve. Take with food to reduce stomach upset.',
        ),
        
        // Alternative if allergic to penicillin
        Medication(
          name: 'Azithromycin (if allergic to penicillin)',
          dosage: '500mg first day, then 250mg',
          frequency: 'Once daily',
          durationDays: 5,
          notes: 'Take on an empty stomach. If you have no penicillin allergy, use amoxicillin instead.',
        ),
        
        // Pain relief
        Medication(
          name: 'Paracetamol (Acetaminophen)',
          dosage: paracetamolDosage,
          frequency: 'Every 6 hours as needed',
          durationDays: 3,
          notes: 'For throat pain and fever relief.',
        ),
        
        // Throat gargle
        Medication(
          name: 'Salt water gargle',
          dosage: '1/2 teaspoon salt in 240ml warm water',
          frequency: 'Every 2-3 hours as needed',
          durationDays: 7,
          notes: 'Do not swallow.',
        ),
      ];
      
      advices = [
        'Complete the full course of antibiotics even if you feel better',
        'Stay hydrated with warm liquids like herbal tea with honey',
        'Rest your voice as much as possible',
        'Use throat lozenges or hard candies to soothe throat',
        'Avoid irritants such as smoking or exposure to smoke',
        'Return for evaluation if symptoms worsen or do not improve within 48 hours on antibiotics'
      ];
    } 
    
    // Gastroenteritis or stomach flu prescriptions
    else if (disease.name.toLowerCase().contains('gastroenteritis') ||
             disease.name.toLowerCase().contains('stomach flu') ||
             disease.name.toLowerCase().contains('gastritis')) {
      
      medications = [
        // Rehydration
        Medication(
          name: 'Oral Rehydration Solution',
          dosage: '200-300ml',
          frequency: 'After each loose stool or vomiting episode',
          durationDays: 3,
          notes: 'Can be prepared by mixing 6 level teaspoons of sugar and 1/2 level teaspoon of salt in 1 liter of clean water.',
        ),
        
        // Anti-diarrheal (for adults only)
        Medication(
          name: 'Loperamide',
          dosage: '2mg',
          frequency: 'After each loose stool, up to 8mg per day',
          durationDays: 2,
          notes: 'Not recommended for children or if you have fever or bloody diarrhea.',
        ),
        
        // Anti-nausea
        Medication(
          name: 'Ondansetron',
          dosage: '4mg',
          frequency: 'Every 8 hours as needed for nausea',
          durationDays: 2,
          notes: 'Dissolve on tongue. Do not take more than 3 tablets per day.',
        ),
        
        // Stomach acid reducer
        Medication(
          name: 'Pantoprazole',
          dosage: '40mg',
          frequency: 'Once daily before breakfast',
          durationDays: 7,
          notes: 'Take on empty stomach. Swallow whole, do not crush or chew.',
        ),
        
        // Probiotics
        Medication(
          name: 'Probiotic supplement',
          dosage: 'As directed on package',
          frequency: 'Once or twice daily',
          durationDays: 10,
          notes: 'Helps restore gut flora.',
        )
      ];
      
      advices = [
        'Stay well-hydrated with clear fluids like water, clear broth, and electrolyte solutions',
        'Follow the BRAT diet (bananas, rice, applesauce, toast) initially',
        'Gradually reintroduce normal foods as symptoms improve',
        'Avoid dairy, caffeine, alcohol, fatty foods, and spicy foods until recovery',
        'Wash hands frequently to prevent spreading infection',
        'Seek medical attention if symptoms worsen, persist beyond 3 days, or if you notice blood in stool'
      ];
    } 
    
    // Migraine prescriptions
    else if (disease.name.toLowerCase().contains('migraine') ||
             (disease.name.toLowerCase().contains('headache') && 
              _hasSymptom(selectedSymptoms, ['sensitivity to light', 'sensitivity to sound', 'nausea']))) {
      
      medications = [
        // First line treatment
        Medication(
          name: 'Sumatriptan',
          dosage: '50mg',
          frequency: 'At onset of migraine; may repeat after 2 hours if needed',
          durationDays: 1,
          notes: 'Do not take more than 200mg in 24 hours. Not suitable for those with heart disease or uncontrolled hypertension.',
        ),
        
        // Anti-inflammatory
        Medication(
          name: 'Ibuprofen',
          dosage: '400mg',
          frequency: 'At onset of migraine, then every 6-8 hours as needed',
          durationDays: 2,
          notes: 'Take with food. May use instead of or in addition to Sumatriptan.',
        ),
        
        // Anti-nausea
        Medication(
          name: 'Metoclopramide',
          dosage: '10mg',
          frequency: 'If nausea is present, every 8 hours as needed',
          durationDays: 2,
          notes: 'Take 30 minutes before meals and at bedtime.',
        ),
        
        // Preventive if migraines are frequent
        Medication(
          name: 'Propranolol',
          dosage: '40mg',
          frequency: 'Twice daily',
          durationDays: 30,
          notes: 'For prevention if migraines occur more than 4 times monthly. Requires ongoing medical supervision.',
        )
      ];
      
      advices = [
        'Rest in a quiet, dark room during acute headache episodes',
        'Apply a cold or warm compress to your head or neck',
        'Practice stress management techniques such as deep breathing or meditation',
        'Maintain regular sleep patterns and meal times',
        'Keep a headache diary to identify and avoid triggers',
        'Consider preventive treatments if headaches occur more than once weekly'
      ];
    } 
    
    // Allergic rhinitis / seasonal allergies prescriptions
    else if (disease.name.toLowerCase().contains('allergy') ||
             disease.name.toLowerCase().contains('rhinitis')) {
      
      bool hasNasalCongestion = _hasSymptom(selectedSymptoms, ['nasal congestion', 'stuffy nose']);
      bool hasEyeSymptoms = _hasSymptom(selectedSymptoms, ['itchy eyes', 'watery eyes']);
      
      medications = [
        // Non-sedating antihistamine
        Medication(
          name: 'Loratadine',
          dosage: '10mg',
          frequency: 'Once daily',
          durationDays: 14,
          notes: 'Non-drowsy antihistamine. Take in the morning.',
        ),
        
        // Alternative antihistamine option
        Medication(
          name: 'Cetirizine',
          dosage: '10mg',
          frequency: 'Once daily, preferably in the evening',
          durationDays: 14,
          notes: 'May cause mild drowsiness in some people.',
        )
      ];
      
      // Add nasal spray for congestion
      if (hasNasalCongestion) {
        medications.add(
          Medication(
            name: 'Fluticasone (Nasal corticosteroid spray)',
            dosage: '1-2 sprays per nostril',
            frequency: 'Once daily',
            durationDays: 14,
            notes: 'May take several days to reach full effect. Most effective when used regularly.',
          )
        );
        
        medications.add(
          Medication(
            name: 'Saline nasal spray',
            dosage: '1-2 sprays per nostril',
            frequency: '3-4 times daily as needed',
            durationDays: 14,
            notes: 'Safe for continuous use. Use before medicated nasal spray.',
          )
        );
      }
      
      // Add eye drops for eye symptoms
      if (hasEyeSymptoms) {
        medications.add(
          Medication(
            name: 'Ketotifen (Antihistamine eye drops)',
            dosage: '1 drop per eye',
            frequency: 'Twice daily',
            durationDays: 14,
            notes: 'For itchy, watery eyes due to allergies.',
          )
        );
      }
      
      advices = [
        'Identify and avoid exposure to your allergy triggers',
        'Keep windows closed during high pollen seasons',
        'Use air purifiers with HEPA filters in your home',
        'Wash bedding weekly in hot water to reduce allergens',
        'Shower and change clothes after outdoor activities during allergy season',
        'Consider allergen immunotherapy (allergy shots) for long-term management'
      ];
    } 
    
    // Bronchitis prescriptions
    else if (disease.name.toLowerCase().contains('bronchitis')) {
      
      bool hasProductiveCough = _hasSymptom(selectedSymptoms, ['productive cough', 'phlegm', 'mucus']);
      bool hasDryCough = _hasSymptom(selectedSymptoms, ['dry cough', 'cough']);
      bool hasFever = temperature != null && temperature > 100.4;
      
      medications = [
        // Cough suppressant for dry cough
        Medication(
          name: 'Dextromethorphan',
          dosage: '15-30mg',
          frequency: 'Every 6-8 hours as needed',
          durationDays: 7,
          notes: 'For dry, non-productive cough. Avoid if you have asthma.',
        )
      ];
      
      // Add expectorant for productive cough
      if (hasProductiveCough) {
        medications.add(
          Medication(
            name: 'Guaifenesin (Expectorant)',
            dosage: '400mg',
            frequency: 'Every 4 hours as needed',
            durationDays: 7,
            notes: 'Helps thin and loosen mucus. Drink plenty of water for best effect.',
          )
        );
      }
      
      // Add bronchodilator if there's wheezing or shortness of breath
      if (_hasSymptom(selectedSymptoms, ['wheezing', 'shortness of breath'])) {
        medications.add(
          Medication(
            name: 'Salbutamol (Albuterol) inhaler',
            dosage: '1-2 puffs',
            frequency: 'Every 4-6 hours as needed for wheezing',
            durationDays: 7,
            notes: 'Shake well before use. May cause tremors or increased heart rate.',
          )
        );
      }
      
      // Add antibiotics only if bacterial bronchitis is suspected
      if (hasFever && hasProductiveCough && _hasSymptom(selectedSymptoms, ['yellow phlegm', 'green phlegm', 'colored phlegm'])) {
        medications.add(
          Medication(
            name: 'Azithromycin',
            dosage: '500mg first day, 250mg following days',
            frequency: 'Once daily',
            durationDays: 5,
            notes: 'Take for the full course even if symptoms improve.',
          )
        );
      }
      
      // Add pain/fever reliever
      medications.add(
        Medication(
          name: 'Paracetamol (Acetaminophen)',
          dosage: '500mg',
          frequency: 'Every 6 hours as needed for fever or pain',
          durationDays: 5,
          notes: 'Do not exceed 4g per day.',
        )
      );
      
      advices = [
        'Rest and stay hydrated to help thin mucus secretions',
        'Use a humidifier or take steamy showers to ease breathing',
        'Avoid smoking and secondhand smoke',
        'Prop yourself up with pillows when sleeping to reduce coughing',
        'Complete the full course of antibiotics if prescribed',
        'Seek medical attention if breathing becomes difficult or symptoms worsen'
      ];
    } 
    
    // Default case for other conditions
    else {
      // Generic medications based on symptoms
      if (_hasSymptom(selectedSymptoms, ['fever', 'pain', 'headache', 'sore throat'])) {
        medications.add(
          Medication(
            name: 'Paracetamol (Acetaminophen)',
            dosage: '500mg',
            frequency: 'Every 6 hours as needed',
            durationDays: 3,
            notes: 'For fever and pain relief. Do not exceed 4g per day.',
          )
        );
      }
      
      if (_hasSymptom(selectedSymptoms, ['cough'])) {
        medications.add(
          Medication(
            name: 'Dextromethorphan',
            dosage: '15-30mg',
            frequency: 'Every 6-8 hours as needed',
            durationDays: 5,
            notes: 'For cough suppression.',
          )
        );
      }
      
      if (_hasSymptom(selectedSymptoms, ['nasal congestion', 'runny nose'])) {
        medications.add(
          Medication(
            name: 'Cetirizine',
            dosage: '10mg',
            frequency: 'Once daily',
            durationDays: 5,
            notes: 'For allergy symptoms.',
          )
        );
      }
      
      if (_hasSymptom(selectedSymptoms, ['diarrhea', 'loose stool'])) {
        medications.add(
          Medication(
            name: 'Loperamide',
            dosage: '2mg',
            frequency: 'After each loose stool, up to 8mg per day',
            durationDays: 2,
            notes: 'Not for use with fever or blood in stool.',
          )
        );
      }
      
      if (_hasSymptom(selectedSymptoms, ['nausea', 'vomiting'])) {
        medications.add(
          Medication(
            name: 'Ondansetron',
            dosage: '4mg',
            frequency: 'Every 8 hours as needed',
            durationDays: 2,
            notes: 'For nausea and vomiting.',
          )
        );
      }
      
      // Add stomach protector if GI symptoms or using pain medications
      if (_hasSymptom(selectedSymptoms, ['heartburn', 'indigestion', 'stomach pain']) || 
          !medications.isEmpty) {
        medications.add(
          Medication(
            name: 'Pantoprazole',
            dosage: '40mg',
            frequency: 'Once daily before breakfast',
            durationDays: 7,
            notes: 'Take on empty stomach to protect against stomach acid.',
          )
        );
      }
      
      advices = [
        'Rest and allow your body time to recover',
        'Stay hydrated by drinking plenty of fluids',
        'Monitor your symptoms and note any changes',
        'Seek medical attention if symptoms worsen or do not improve within 3-5 days',
        'Consider telehealth consultation for professional medical advice'
      ];
    }
    
    // Add age-specific advice
    if (age > 60) {
      advices.add('As an older adult, you may be at higher risk for complications. Consider consulting with your healthcare provider even for seemingly minor symptoms.');
    } else if (age < 18) {
      advices.add('For pediatric patients, medication dosages should be confirmed by a healthcare provider based on exact weight and age.');
    }
    
    // Add BMI-related advice if height and weight are available
    if (height != null && weight != null) {
      double heightInMeters = height / 100;
      double bmi = weight / (heightInMeters * heightInMeters);
      
      if (bmi < 18.5) {
        advices.add('Your BMI indicates you may be underweight. Consider discussing nutrition with your healthcare provider.');
      } else if (bmi >= 25 && bmi < 30) {
        advices.add('Your BMI indicates you may be overweight. Maintaining a healthy weight can help prevent or manage many health conditions.');
      } else if (bmi >= 30) {
        advices.add('Your BMI indicates obesity, which may impact your health. Consider discussing weight management strategies with your healthcare provider.');
      }
    }
    
    // Add research-informed advice based on stored research data
    if (_latestResearchData.isNotEmpty) {
      String researchAdvice = 'RESEARCH INSIGHT: ';
      // Get the most relevant research
      final mostRelevant = _latestResearchData.firstWhere(
        (research) => research['relevance'] == 'high',
        orElse: () => _latestResearchData.first
      );
      
      researchAdvice += mostRelevant['summary'];
      advices.add(researchAdvice);
    }
    
    // Add risk factor information to advice
    if (_latestRiskFactors.isNotEmpty) {
      // Get high impact risk factors first
      final highImpactFactors = _latestRiskFactors.where((factor) => factor['impact'] == 'high').toList();
      if (highImpactFactors.isNotEmpty) {
        String riskAdvice = 'IMPORTANT HEALTH FACTOR: ${highImpactFactors.first['factor']} - ${highImpactFactors.first['description']}';
        advices.add(riskAdvice);
      }
    }
    
    // Professional disclaimer for all prescriptions
    advices.add('IMPORTANT: This AI-generated prescription is based on the information provided and is not a substitute for professional medical advice. Please consult with a qualified healthcare provider before starting any new medication or treatment regimen.');
    
    // Simulate processing time for comprehensive analysis
    await Future.delayed(const Duration(seconds: 1));
    
    return PrescriptionModel(
      diseaseName: disease.name,
      diseaseId: disease.id,
      accuracy: disease.accuracy,
      medications: medications,
      advices: advices,
      createdAt: DateTime.now(),
    );
  }
  
  // Helper method to check if any of the symptom names are present
  bool _hasSymptom(List<SymptomModel> symptoms, List<String> keywords) {
    for (var symptom in symptoms) {
      for (var keyword in keywords) {
        if (symptom.name.toLowerCase().contains(keyword.toLowerCase())) {
          return true;
        }
      }
    }
    return false;
  }
  
  // Mock data methods for reliable application functionality
  List<SymptomModel> _getMockSymptoms() {
    return [
      SymptomModel(id: 10, name: 'Abdominal pain'),
      SymptomModel(id: 238, name: 'Anxiety'),
      SymptomModel(id: 104, name: 'Back pain'),
      SymptomModel(id: 75, name: 'Cough'),
      SymptomModel(id: 16, name: 'Diarrhea'),
      SymptomModel(id: 95, name: 'Dizziness'),
      SymptomModel(id: 11, name: 'Fatigue'),
      SymptomModel(id: 57, name: 'Fever'),
      SymptomModel(id: 9, name: 'Headache'),
      SymptomModel(id: 45, name: 'Nausea'),
      SymptomModel(id: 29, name: 'Shortness of breath'),
      SymptomModel(id: 13, name: 'Sore throat'),
      SymptomModel(id: 14, name: 'Vomiting'),
      SymptomModel(id: 52, name: 'Sneezing'),
      SymptomModel(id: 12, name: 'Runny nose'),
      SymptomModel(id: 15, name: 'Chest pain'),
      SymptomModel(id: 22, name: 'Blurry vision'),
      SymptomModel(id: 31, name: 'Earache'),
      SymptomModel(id: 37, name: 'Skin rash'),
      SymptomModel(id: 46, name: 'Joint pain'),
      SymptomModel(id: 50, name: 'Difficulty swallowing'),
      SymptomModel(id: 60, name: 'Swollen lymph nodes'),
      SymptomModel(id: 65, name: 'Muscle weakness'),
      SymptomModel(id: 70, name: 'Loss of appetite'),
      SymptomModel(id: 78, name: 'Insomnia'),
      SymptomModel(id: 85, name: 'Night sweats'),
      SymptomModel(id: 88, name: 'Nasal congestion'),
      SymptomModel(id: 91, name: 'Dry cough'),
      SymptomModel(id: 92, name: 'Productive cough'),
      SymptomModel(id: 95, name: 'Wheezing'),
      SymptomModel(id: 96, name: 'Sensitivity to light'),
      SymptomModel(id: 98, name: 'Sensitivity to sound'),
      SymptomModel(id: 102, name: 'Palpitations'),
      SymptomModel(id: 104, name: 'Stuffy nose'),
      SymptomModel(id: 108, name: 'Panic attack'),
      SymptomModel(id: 112, name: 'Severe anxiety'),
      SymptomModel(id: 118, name: 'Blood in stool'),
      SymptomModel(id: 120, name: 'Weight loss'),
      SymptomModel(id: 121, name: 'Weight gain'),
      SymptomModel(id: 125, name: 'Excessive thirst'),
      SymptomModel(id: 131, name: 'Frequent urination'),
    ];
  }
  
  // Enhanced diagnosis method with more comprehensive analysis
  List<DiseaseModel> _getAdvancedDiagnosis(List<int> symptomIds, Map<String, dynamic> vitalParameters) {
    // Extract vital signs for better diagnosis
    final gender = vitalParameters['gender'] ?? 'male';
    final birthYear = vitalParameters['birthYear'] ?? 2000;
    final int age = DateTime.now().year - (birthYear as int);
    final double? temperature = vitalParameters['temperature'];
    final double? weight = vitalParameters['weight'];
    final double? height = vitalParameters['height'];
    final int? heartRate = vitalParameters['heartRate'];
    final int? spo2 = vitalParameters['spo2'];
    
    // Calculate BMI if height and weight are available
    double? bmi;
    if (height != null && weight != null) {
      double heightInMeters = height / 100;
      bmi = weight / (heightInMeters * heightInMeters);
    }
    
    // Setup symptom sets for pattern matching
    final bool hasFever = temperature != null && temperature >= 100.4;
    final bool hasHighFever = temperature != null && temperature >= 102.0;
    final bool hasCough = symptomIds.contains(75) || symptomIds.contains(91) || symptomIds.contains(92);
    final bool hasRunnyNose = symptomIds.contains(12);
    final bool hasNasalCongestion = symptomIds.contains(88) || symptomIds.contains(104);
    final bool hasSoreThroat = symptomIds.contains(13);
    final bool hasHeadache = symptomIds.contains(9);
    final bool hasFatigue = symptomIds.contains(11);
    final bool hasBodyPain = symptomIds.contains(46) || symptomIds.contains(104);
    final bool hasBreathingIssues = symptomIds.contains(29) || symptomIds.contains(95);
    final bool hasGISymptoms = symptomIds.contains(16) || symptomIds.contains(45) || symptomIds.contains(14);
    
    // Advanced diagnosis based on symptom patterns and vital signs
    
    // COVID-19 pattern: fever, cough, fatigue, with potential SpO2 reduction
    if (hasFever && hasCough && (hasFatigue || hasBreathingIssues) && (spo2 != null && spo2 < 95)) {
      return [
        DiseaseModel(
          id: 201,
          name: 'Possible COVID-19',
          accuracy: 0.85,
          specialisations: ['Infectious Disease', 'Pulmonology'],
          description: 'COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. Symptoms typically include fever, cough, and fatigue, with possible breathing difficulties. Your reduced oxygen saturation (SpO2) is concerning and requires immediate medical attention.',
          possibleSymptoms: ['Fever', 'Cough', 'Fatigue', 'Shortness of breath', 'Loss of taste or smell'],
          treatmentDescription: ['Testing recommended', 'Isolation until confirmed', 'Supportive care', 'Oxygen therapy if needed'],
        ),
      ];
    }
    
    // Common cold pattern
    else if (hasRunnyNose && (hasCough || hasNasalCongestion) && !hasHighFever) {
      return [
        DiseaseModel(
          id: 15,
          name: 'Common Cold',
          accuracy: 0.92,
          specialisations: ['General Practice', 'ENT'],
          description: "The common cold is a viral infection of your upper respiratory tract. It's usually harmless, although it might not feel that way. Many types of viruses can cause a common cold, with rhinoviruses being the most common.",
          possibleSymptoms: ['Runny nose', 'Nasal congestion', 'Sore throat', 'Cough', 'Sneezing'],
          treatmentDescription: ['Rest', 'Hydration', 'Over-the-counter symptom relief'],
        ),
      ];
    }
    
    // Influenza pattern: high fever, body aches, extreme fatigue
    else if ((hasHighFever || (hasFever && hasFatigue)) && (hasCough || hasBodyPain) && !hasGISymptoms) {
      return [
        DiseaseModel(
          id: 55,
          name: 'Influenza (Flu)',
          accuracy: 0.88,
          specialisations: ['Infectious Disease', 'General Practice'],
          description: "Influenza is a viral infection that attacks your respiratory system. Unlike the common cold, flu symptoms come on suddenly and are typically more severe. Annual vaccination is the best prevention method.",
          possibleSymptoms: ['High fever', 'Body aches', 'Fatigue', 'Cough', 'Headache'],
          treatmentDescription: ['Rest', 'Hydration', 'Antiviral medications if started early', 'Fever reducers'],
        ),
      ];
    }
    
    // Migraine pattern
    else if (hasHeadache && (symptomIds.contains(96) || symptomIds.contains(98)) && !hasFever) {
      return [
        DiseaseModel(
          id: 121,
          name: 'Migraine',
          accuracy: 0.78,
          specialisations: ['Neurology'],
          description: 'Migraine is a neurological condition characterized by intense, debilitating headaches. Symptoms may include throbbing pain, nausea, vomiting, and extreme sensitivity to light and sound.',
          possibleSymptoms: ['Throbbing headache', 'Sensitivity to light', 'Sensitivity to sound', 'Nausea', 'Visual disturbances'],
          treatmentDescription: ['Rest in a dark, quiet room', 'Prescription medications', 'Preventive therapies'],
        ),
      ];
    }
    
    // Gastroenteritis pattern
    else if (hasGISymptoms && (hasFever || symptomIds.contains(70))) {
      return [
        DiseaseModel(
          id: 60,
          name: 'Gastroenteritis',
          accuracy: 0.82,
          specialisations: ['Gastroenterology', 'Internal Medicine'],
          description: "Gastroenteritis is an inflammation of the digestive tract, particularly the stomach and intestines. It's typically caused by a viral or bacterial infection and often results in diarrhea and vomiting.",
          possibleSymptoms: ['Diarrhea', 'Nausea', 'Vomiting', 'Abdominal cramps', 'Low-grade fever'],
          treatmentDescription: ['Hydration', 'Rest', 'Gradual reintroduction of foods', 'Electrolyte replacement'],
        ),
      ];
    }
    
    // Strep throat pattern
    else if (hasSoreThroat && hasHighFever && (symptomIds.contains(60) || !hasRunnyNose)) {
      return [
        DiseaseModel(
          id: 202,
          name: 'Strep Throat',
          accuracy: 0.75,
          specialisations: ['ENT', 'Infectious Disease'],
          description: 'Strep throat is a bacterial infection caused by group A Streptococcus bacteria. It causes inflammation and pain in the throat, often with white patches on the tonsils. Unlike viral sore throats, strep requires antibiotic treatment.',
          possibleSymptoms: ['Severe sore throat', 'Difficulty swallowing', 'Fever', 'Swollen lymph nodes', 'White patches on tonsils'],
          treatmentDescription: ['Antibiotics', 'Pain relievers', 'Rest', 'Warm salt water gargles'],
        ),
      ];
    }
    
    // Anxiety disorder pattern
    else if (symptomIds.contains(238) || symptomIds.contains(108) || symptomIds.contains(112)) {
      return [
        DiseaseModel(
          id: 102,
          name: 'Anxiety Disorder',
          accuracy: 0.80,
          specialisations: ['Psychiatry', 'Psychology'],
          description: 'Anxiety disorders are characterized by persistent, excessive worry or fear that interferes with daily activities. This is more than just temporary worry or fear related to stressful events.',
          possibleSymptoms: ['Excessive worry', 'Restlessness', 'Fatigue', 'Difficulty concentrating', 'Irritability'],
          treatmentDescription: ['Cognitive behavioral therapy', 'Stress management techniques', 'Meditation', 'Possible medication'],
        ),
      ];
    }
    
    // Hypertension pattern - based on self-reported symptoms and risk factors
    else if (symptomIds.contains(15) || symptomIds.contains(9)) {
      // Check risk factors like age and BMI
      bool hasRiskFactors = (age > 50) || (bmi != null && bmi > 30);
      
      if (hasRiskFactors || heartRate != null && heartRate > 100) {
        return [
          DiseaseModel(
            id: 180,
            name: 'Possible Hypertension',
            accuracy: 0.65,
            specialisations: ['Cardiology', 'Internal Medicine'],
            description: 'Hypertension, or high blood pressure, is a common condition where the force of blood against artery walls is consistently too high. It typically develops over years and can lead to serious health problems if untreated.',
            possibleSymptoms: ['Usually asymptomatic', 'Headache in severe cases', 'Chest pain', 'Shortness of breath', 'Dizziness'],
            treatmentDescription: ['Blood pressure measurement', 'Lifestyle modifications', 'Possible medication'],
          ),
        ];
      }
    }
    
    // Allergic rhinitis pattern
    else if ((hasRunnyNose || hasNasalCongestion) && symptomIds.contains(52) && !hasFever) {
      return [
        DiseaseModel(
          id: 130,
          name: 'Allergic Rhinitis',
          accuracy: 0.88,
          specialisations: ['Allergy & Immunology', 'ENT'],
          description: 'Allergic rhinitis is an inflammatory reaction of the nasal mucosa to airborne allergens like pollen, causing inflammation in the nasal passages. It presents with sneezing, nasal congestion, and runny nose.',
          possibleSymptoms: ['Sneezing', 'Runny nose', 'Nasal congestion', 'Itchy eyes', 'Postnasal drip'],
          treatmentDescription: ['Allergen avoidance', 'Antihistamines', 'Nasal corticosteroids', 'Immunotherapy for severe cases'],
        ),
      ];
    }
    
    // Diabetes pattern - look for classic symptoms
    else if ((symptomIds.contains(125) && symptomIds.contains(131)) || 
            (symptomIds.contains(125) && symptomIds.contains(120)) || 
            (symptomIds.contains(131) && symptomIds.contains(120))) {
      return [
        DiseaseModel(
          id: 190,
          name: 'Possible Diabetes',
          accuracy: 0.70,
          specialisations: ['Endocrinology', 'Internal Medicine'],
          description: 'Diabetes is a chronic condition affecting how your body turns food into energy. It occurs when your body doesn\'t make enough insulin or can\'t use insulin effectively, resulting in high blood sugar levels.',
          possibleSymptoms: ['Excessive thirst', 'Frequent urination', 'Unexplained weight loss', 'Fatigue', 'Blurred vision'],
          treatmentDescription: ['Blood glucose testing', 'Dietary changes', 'Regular exercise', 'Possible medication or insulin therapy'],
        ),
      ];
    }
    
    // Generic response for all other cases (default fallback)
    // Adjust confidence based on the number of symptoms provided
    double confidence = math.min(0.5 + (symptomIds.length / 20), 0.7);
    
    return [
      DiseaseModel(
        id: 999,
        name: 'Unspecified Condition',
        accuracy: confidence,
        specialisations: ['General Practice'],
        description: "Based on the combination of symptoms you've provided, a specific diagnosis cannot be determined with confidence. The symptoms may be related to various conditions, and professional medical evaluation is recommended.",
        possibleSymptoms: ['Various symptoms depending on underlying cause'],
        treatmentDescription: ['Medical consultation recommended', 'Symptom-specific management'],
      ),
    ];
  }
  
  /// Analyzes symptoms using AI models and returns a list of possible diagnoses
  Future<List<DiseaseModel>> analyzeSymptomsWithAI(
    List<int> symptomIds, 
    Map<String, dynamic> vitalParameters
  ) async {
    try {
      _logger.info('Analyzing symptoms with AI', tag: 'MedicalService');
      
      // Instead of using mock data, use the advanced diagnosis algorithm
      // which actually analyzes the symptoms provided
      if (!AppConfig.enableRealTimeAnalysis) {
        return _getAdvancedDiagnosis(symptomIds, vitalParameters);
      }
      
      final response = await http.post(
        Uri.parse('$_baseUrl/analyze'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $_apiKey',
        },
        body: jsonEncode({
          'symptomIds': symptomIds,
          'vitalParameters': vitalParameters,
        }),
      );
      
      if (response.statusCode == 200) {
        final jsonResponse = jsonDecode(response.body);
        final List<dynamic> diseases = jsonResponse['diseases'];
        
        return diseases.map((disease) => DiseaseModel.fromJson(disease)).toList();
      } else {
        _logger.error(
          'Failed to analyze symptoms', 
          tag: 'MedicalService',
          error: 'Status code: ${response.statusCode}, Body: ${response.body}'
        );
        throw Exception('Failed to analyze symptoms');
      }
    } catch (e, stackTrace) {
      _logger.error('Error analyzing symptoms', tag: 'MedicalService', error: e, stackTrace: stackTrace);
      // Here too, fall back to advanced diagnosis instead of mock data
      return _getAdvancedDiagnosis(symptomIds, vitalParameters);
    }
  }
  
  /// Gets statistics for a specific disease
  Future<Map<String, dynamic>> getDiseaseStatistics(String diseaseName) async {
    try {
      _logger.info('Getting statistics for disease: $diseaseName', tag: 'MedicalService');
      
      // In a real app, this would make an API call to get real statistics
      // For now, returning mock data
      return {
        'prevalence': '4.5% of global population',
        'annualCases': 'Approximately 350 million new cases annually',
        'mortalityRate': '0.8% in developed countries, 2.3% globally',
        'treatmentSuccess': '85% with early detection and intervention',
        'sourceInfo': 'World Health Organization, 2025'
      };
    } catch (e, stackTrace) {
      _logger.error('Error getting disease statistics', tag: 'MedicalService', error: e, stackTrace: stackTrace);
      throw Exception('Error getting disease statistics: ${e.toString()}');
    }
  }
  
  /// Private method to get mock diagnoses for testing
  List<DiseaseModel> _getMockDiagnoses() {
    return [
      DiseaseModel(
        id: 1,
        name: 'Upper Respiratory Infection',
        accuracy: 0.89,
        description: 'A viral infection affecting the upper respiratory tract, including the nose, throat, and airways.',
        treatmentDescription: [
          'Rest and adequate hydration',
          'Over-the-counter pain relievers for fever and pain',
          'Saline nasal spray to relieve congestion',
          'Humidifier to ease breathing'
        ],
        specialisations: ['General Practice', 'ENT']
      ),
      DiseaseModel(
        id: 2,
        name: 'Seasonal Allergies',
        accuracy: 0.75,
        description: 'An immune system response to environmental triggers such as pollen, causing inflammation in the nasal passages.',
        treatmentDescription: [
          'Antihistamines to reduce allergic reactions',
          'Nasal corticosteroids to reduce inflammation',
          'Avoiding known allergens when possible',
          'Regular cleaning to reduce indoor allergens'
        ],
        specialisations: ['Allergy & Immunology', 'General Practice']
      ),
      DiseaseModel(
        id: 3,
        name: 'Gastroenteritis',
        accuracy: 0.62,
        description: 'Inflammation of the stomach and intestines, typically resulting from a viral or bacterial infection.',
        treatmentDescription: [
          'Oral rehydration solutions to prevent dehydration',
          'Clear liquids until symptoms improve',
          'Gradual reintroduction of bland foods',
          'Probiotics to restore gut flora'
        ],
        specialisations: ['Gastroenterology', 'Internal Medicine']
      )
    ];
  }
}