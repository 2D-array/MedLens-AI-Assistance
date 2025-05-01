import 'package:flutter/foundation.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// AppConfig manages application configuration and API keys
class AppConfig {
  // API keys and endpoints
  static const String medicalApiKey = 'YOUR_MEDICAL_API_KEY';
  static const String medicalApiEndpoint = 'https://api.medicaldata.org/v1';
  static const String researchApiKey = 'YOUR_RESEARCH_API_KEY'; 
  static const String researchApiEndpoint = 'https://api.medicalresearch.org/v1';
  
  // Feature flags
  static const bool enableMedicalResearch = true;
  static const bool enableRealTimeAnalysis = true;
  static const bool enableChatWithDoctor = false; // Premium feature
  
  // App settings
  static const int maxSymptomsSelection = 10;
  static const int maxDiagnosisResults = 5;
  static const int cacheExpiryMinutes = 60;
  
  // UI settings
  static const double cardElevation = 4.0;
  static const double borderRadius = 12.0;
  
  // Notification settings
  static const bool enablePushNotifications = true;
  static const bool enableEmailNotifications = false;
  
  // Logging settings
  static const bool enableDebugLogs = true;
  static const bool enableAnalytics = true;
}