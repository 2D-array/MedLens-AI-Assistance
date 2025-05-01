class SymptomModel {
  final int id;
  final String name;
  final bool selected;

  SymptomModel({
    required this.id,
    required this.name,
    this.selected = false,
  });

  factory SymptomModel.fromJson(Map<String, dynamic> json) {
    return SymptomModel(
      id: json['ID'] ?? 0,
      name: json['Name'] ?? '',
    );
  }

  SymptomModel copyWith({
    int? id,
    String? name, 
    bool? selected,
  }) {
    return SymptomModel(
      id: id ?? this.id,
      name: name ?? this.name,
      selected: selected ?? this.selected,
    );
  }
}