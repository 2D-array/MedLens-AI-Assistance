class DiseaseModel {
  final int id;
  final String name;
  final double accuracy;
  final List<String> specialisations;
  final String description;
  final List<String> possibleSymptoms;
  final List<String> treatmentDescription;
  final String? severity;
  final Map<String, dynamic>? riskFactors;

  DiseaseModel({
    required this.id,
    required this.name,
    required this.accuracy,
    required this.specialisations,
    this.description = '',
    this.possibleSymptoms = const [],
    this.treatmentDescription = const [],
    this.severity,
    this.riskFactors,
  });

  factory DiseaseModel.fromJson(Map<String, dynamic> json) {
    return DiseaseModel(
      id: json['Issue']['ID'] ?? 0,
      name: json['Issue']['Name'] ?? '',
      accuracy: json['Issue']['Accuracy'] ?? 0.0,
      specialisations: (json['Specialisation'] as List?)
          ?.map((spec) => spec['Name'].toString())
          .toList() ??
          [],
      description: json['Description'] ?? '', 
      possibleSymptoms: json['PossibleSymptoms'] != null 
          ? List<String>.from(json['PossibleSymptoms'])
          : [],
      treatmentDescription: json['TreatmentDescription'] != null
          ? List<String>.from(json['TreatmentDescription'])
          : [],
      severity: json['Severity'],
      riskFactors: json['RiskFactors'] as Map<String, dynamic>?,
    );
  }
}