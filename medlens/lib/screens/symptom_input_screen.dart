import 'package:flutter/material.dart';
import 'package:medlens/models/symptom_model.dart';
import 'package:medlens/services/medical_service.dart';
import 'package:medlens/utils/app_theme.dart';
import 'package:medlens/widgets/custom_button.dart';
import 'package:medlens/widgets/custom_input_field.dart';
import 'package:medlens/screens/diagnosis_result_screen.dart';

class SymptomInputScreen extends StatefulWidget {
  const SymptomInputScreen({Key? key}) : super(key: key);

  @override
  State<SymptomInputScreen> createState() => _SymptomInputScreenState();
}

class _SymptomInputScreenState extends State<SymptomInputScreen> {
  final _medicalService = MedicalService();
  final _formKey = GlobalKey<FormState>();
  final _searchController = TextEditingController();
  final _yearOfBirthController = TextEditingController();
  // New controllers for vital parameters
  final _temperatureController = TextEditingController();
  final _heightController = TextEditingController();
  final _weightController = TextEditingController();
  final _heartRateController = TextEditingController();
  final _spo2Controller = TextEditingController();
  
  List<SymptomModel> _allSymptoms = [];
  List<SymptomModel> _filteredSymptoms = [];
  List<SymptomModel> _selectedSymptoms = [];
  String _selectedGender = 'male';
  
  bool _isLoading = true;
  bool _isProcessing = false;
  String? _errorMessage;
  bool _showAdvancedParameters = false;

  @override
  void initState() {
    super.initState();
    _loadSymptoms();
  }

  @override
  void dispose() {
    _searchController.dispose();
    _yearOfBirthController.dispose();
    _temperatureController.dispose();
    _heightController.dispose();
    _weightController.dispose();
    _heartRateController.dispose();
    _spo2Controller.dispose();
    super.dispose();
  }

  Future<void> _loadSymptoms() async {
    try {
      setState(() {
        _isLoading = true;
        _errorMessage = null;
      });

      final symptoms = await _medicalService.getSymptoms();
      
      setState(() {
        _allSymptoms = symptoms;
        _filteredSymptoms = symptoms;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to load symptoms. Please try again.';
        _isLoading = false;
      });
    }
  }

  void _filterSymptoms(String query) {
    setState(() {
      if (query.isEmpty) {
        _filteredSymptoms = _allSymptoms;
      } else {
        _filteredSymptoms = _allSymptoms
            .where((symptom) => symptom.name.toLowerCase().contains(query.toLowerCase()))
            .toList();
      }
    });
  }

  void _toggleSymptom(SymptomModel symptom) {
    setState(() {
      final isAlreadySelected = _selectedSymptoms.any((s) => s.id == symptom.id);
      
      if (isAlreadySelected) {
        _selectedSymptoms.removeWhere((s) => s.id == symptom.id);
      } else {
        _selectedSymptoms.add(symptom);
      }
    });
  }

  bool _validateForm() {
    if (!_formKey.currentState!.validate()) {
      return false;
    }
    
    if (_selectedSymptoms.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select at least one symptom'),
          backgroundColor: Colors.red,
        ),
      );
      return false;
    }

    return true;
  }

  // Parse vital parameters with validation
  Map<String, dynamic> _getVitalParameters() {
    final Map<String, dynamic> vitals = {
      'birthYear': int.parse(_yearOfBirthController.text),
      'gender': _selectedGender,
    };
    
    // Add temperature if provided
    if (_temperatureController.text.isNotEmpty) {
      final temp = double.tryParse(_temperatureController.text);
      if (temp != null) vitals['temperature'] = temp;
    }
    
    // Add height if provided
    if (_heightController.text.isNotEmpty) {
      final height = double.tryParse(_heightController.text);
      if (height != null) vitals['height'] = height;
    }
    
    // Add weight if provided
    if (_weightController.text.isNotEmpty) {
      final weight = double.tryParse(_weightController.text);
      if (weight != null) vitals['weight'] = weight;
    }
    
    // Add heart rate if provided
    if (_heartRateController.text.isNotEmpty) {
      final heartRate = int.tryParse(_heartRateController.text);
      if (heartRate != null) vitals['heartRate'] = heartRate;
    }
    
    // Add SPO2 if provided
    if (_spo2Controller.text.isNotEmpty) {
      final spo2 = int.tryParse(_spo2Controller.text);
      if (spo2 != null) vitals['spo2'] = spo2;
    }
    
    return vitals;
  }

  Future<void> _analyzeSymptoms() async {
    if (!_validateForm()) return;

    setState(() {
      _isProcessing = true;
      _errorMessage = null;
    });

    try {
      final symptomIds = _selectedSymptoms.map((s) => s.id).toList();
      final vitalParameters = _getVitalParameters();
      
      final diagnoses = await _medicalService.getDiagnosis(
        symptomIds,
        vitalParameters,
      );
      
      if (mounted) {
        if (diagnoses.isNotEmpty) {
          final disease = diagnoses[0]; // Get the most likely disease
          final prescription = await _medicalService.generatePrescription(
            disease,
            vitalParameters,
            _selectedSymptoms,
          );
          
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => DiagnosisResultScreen(
                diseases: [disease],
                selectedSymptomIds: symptomIds,
                vitalParameters: vitalParameters,
                selectedSymptoms: _selectedSymptoms,
                prescription: prescription, // Pass the prescription to the result screen
              ),
            ),
          );
        } else {
          setState(() {
            _errorMessage = 'Could not determine a diagnosis based on the provided symptoms. Please try again or consult a healthcare professional.';
          });
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'An error occurred during analysis. Please try again.';
        });
      }
    } finally {
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Symptom Checker'),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : SafeArea(
              child: Column(
                children: [
                  // Main content area (scrollable)
                  Expanded(
                    child: CustomScrollView(
                      slivers: [
                        // Header and search
                        SliverToBoxAdapter(
                          child: Padding(
                            padding: const EdgeInsets.fromLTRB(16.0, 16.0, 16.0, 8.0),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Enter Your Symptoms',
                                  style: Theme.of(context).textTheme.titleLarge?.copyWith(
                                        fontWeight: FontWeight.bold,
                                      ),
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  'Select all symptoms you are experiencing',
                                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                                        color: AppTheme.textSecondaryColor,
                                      ),
                                ),
                                const SizedBox(height: 16),
                                
                                // Search bar
                                TextField(
                                  controller: _searchController,
                                  decoration: InputDecoration(
                                    hintText: 'Search symptoms...',
                                    prefixIcon: const Icon(Icons.search),
                                    border: OutlineInputBorder(
                                      borderRadius: BorderRadius.circular(12.0),
                                    ),
                                    contentPadding: const EdgeInsets.symmetric(
                                      vertical: 12.0, 
                                      horizontal: 16.0,
                                    ),
                                    filled: true,
                                    fillColor: Colors.grey.shade50,
                                  ),
                                  onChanged: _filterSymptoms,
                                ),
                              ],
                            ),
                          ),
                        ),
                        
                        // Filtered symptoms list
                        SliverPadding(
                          padding: const EdgeInsets.symmetric(horizontal: 16.0),
                          sliver: _filteredSymptoms.isEmpty
                              ? SliverFillRemaining(
                                  hasScrollBody: false,
                                  child: Center(
                                    child: Text(
                                      'No symptoms found. Try a different search term.',
                                      style: TextStyle(color: AppTheme.textSecondaryColor),
                                    ),
                                  ),
                                )
                              : SliverList(
                                  delegate: SliverChildBuilderDelegate(
                                    (context, index) {
                                      final symptom = _filteredSymptoms[index];
                                      final isSelected = _selectedSymptoms.any((s) => s.id == symptom.id);
                                      
                                      return Card(
                                        elevation: 0,
                                        color: isSelected ? Colors.blue[50] : Colors.grey[50],
                                        margin: const EdgeInsets.only(bottom: 8),
                                        shape: RoundedRectangleBorder(
                                          borderRadius: BorderRadius.circular(10),
                                          side: BorderSide(
                                            color: isSelected ? Colors.blue.shade200 : Colors.grey.shade200,
                                            width: 1,
                                          ),
                                        ),
                                        child: ListTile(
                                          contentPadding: const EdgeInsets.symmetric(
                                            horizontal: 16.0,
                                            vertical: 4.0,
                                          ),
                                          title: Text(
                                            symptom.name,
                                            style: TextStyle(
                                              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                                            ),
                                          ),
                                          leading: CircleAvatar(
                                            backgroundColor: isSelected
                                                ? AppTheme.primaryColor
                                                : Colors.grey[200],
                                            child: Icon(
                                              isSelected ? Icons.check : Icons.add,
                                              color: isSelected ? Colors.white : Colors.grey,
                                              size: 20,
                                            ),
                                          ),
                                          onTap: () => _toggleSymptom(symptom),
                                        ),
                                      );
                                    },
                                    childCount: _filteredSymptoms.length,
                                  ),
                                ),
                        ),
                      ],
                    ),
                  ),
                  
                  // Selected symptoms display area (horizontally scrollable)
                  if (_selectedSymptoms.isNotEmpty)
                    Container(
                      padding: const EdgeInsets.symmetric(vertical: 8.0),
                      decoration: BoxDecoration(
                        color: Colors.grey[50],
                        border: Border(
                          top: BorderSide(color: Colors.grey.shade200),
                        ),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Padding(
                            padding: const EdgeInsets.only(left: 16.0, top: 4.0),
                            child: Text(
                              'Selected Symptoms:',
                              style: Theme.of(context).textTheme.titleSmall?.copyWith(
                                    fontWeight: FontWeight.bold,
                                    color: AppTheme.textSecondaryColor,
                                  ),
                            ),
                          ),
                          const SizedBox(height: 4),
                          SizedBox(
                            height: 48,
                            child: ListView.builder(
                              scrollDirection: Axis.horizontal,
                              padding: const EdgeInsets.symmetric(horizontal: 12.0),
                              itemCount: _selectedSymptoms.length,
                              itemBuilder: (context, index) {
                                final symptom = _selectedSymptoms[index];
                                return Padding(
                                  padding: const EdgeInsets.symmetric(horizontal: 4.0),
                                  child: Chip(
                                    label: Text(symptom.name),
                                    deleteIcon: const Icon(Icons.close, size: 16),
                                    onDeleted: () => _toggleSymptom(symptom),
                                    backgroundColor: Colors.blue[100],
                                    padding: const EdgeInsets.all(4),
                                    labelPadding: const EdgeInsets.symmetric(horizontal: 4),
                                  ),
                                );
                              },
                            ),
                          ),
                        ],
                      ),
                    ),
                  
                  // Patient details form in bottom sheet
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      boxShadow: [
                        BoxShadow(
                          color: Colors.grey.withOpacity(0.2),
                          spreadRadius: 1,
                          blurRadius: 4,
                          offset: const Offset(0, -1),
                        ),
                      ],
                    ),
                    child: ExpansionTile(
                      title: Text(
                        'Patient Details',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: AppTheme.primaryColor,
                        ),
                      ),
                      leading: Icon(Icons.person, color: AppTheme.primaryColor),
                      trailing: Icon(Icons.expand_more, color: AppTheme.primaryColor),
                      initiallyExpanded: _selectedSymptoms.isNotEmpty,
                      children: [
                        Padding(
                          padding: const EdgeInsets.fromLTRB(16.0, 0.0, 16.0, 16.0),
                          child: Form(
                            key: _formKey,
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                // Gender selection
                                Row(
                                  children: [
                                    const Text(
                                      'Gender:',
                                      style: TextStyle(fontWeight: FontWeight.bold),
                                    ),
                                    const SizedBox(width: 16),
                                    Expanded(
                                      child: Row(
                                        children: [
                                          Radio(
                                            value: 'male',
                                            groupValue: _selectedGender,
                                            onChanged: (value) {
                                              setState(() {
                                                _selectedGender = value.toString();
                                              });
                                            },
                                            materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                                          ),
                                          const Text('Male'),
                                        ],
                                      ),
                                    ),
                                    Expanded(
                                      child: Row(
                                        children: [
                                          Radio(
                                            value: 'female',
                                            groupValue: _selectedGender,
                                            onChanged: (value) {
                                              setState(() {
                                                _selectedGender = value.toString();
                                              });
                                            },
                                            materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                                          ),
                                          const Text('Female'),
                                        ],
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 12),
                                
                                // Basic vital parameters
                                Row(
                                  children: [
                                    // Year of birth
                                    Expanded(
                                      child: CustomInputField(
                                        hintText: 'Year of Birth',
                                        controller: _yearOfBirthController,
                                        keyboardType: TextInputType.number,
                                        validator: (value) {
                                          if (value == null || value.isEmpty) {
                                            return 'Required';
                                          }
                                          final birthYear = int.tryParse(value);
                                          if (birthYear == null || 
                                              birthYear < 1900 || 
                                              birthYear > DateTime.now().year) {
                                            return 'Invalid year';
                                          }
                                          return null;
                                        },
                                      ),
                                    ),
                                    const SizedBox(width: 12),
                                    
                                    // Body temperature
                                    Expanded(
                                      child: CustomInputField(
                                        hintText: 'Temperature (°F)',
                                        controller: _temperatureController,
                                        keyboardType: const TextInputType.numberWithOptions(decimal: true),
                                        validator: (value) {
                                          if (value != null && value.isNotEmpty) {
                                            final temp = double.tryParse(value);
                                            if (temp == null || temp < 95 || temp > 108) {
                                              return 'Invalid temp';
                                            }
                                          }
                                          return null;
                                        },
                                      ),
                                    ),
                                  ],
                                ),
                                
                                // Advanced parameters toggle
                                const SizedBox(height: 16),
                                InkWell(
                                  onTap: () {
                                    setState(() {
                                      _showAdvancedParameters = !_showAdvancedParameters;
                                    });
                                  },
                                  child: Row(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      Text(
                                        _showAdvancedParameters
                                            ? 'Hide Advanced Parameters'
                                            : 'Show Advanced Parameters',
                                        style: TextStyle(
                                          color: AppTheme.primaryColor,
                                          fontWeight: FontWeight.bold,
                                        ),
                                      ),
                                      const SizedBox(width: 4),
                                      Icon(
                                        _showAdvancedParameters
                                            ? Icons.keyboard_arrow_up
                                            : Icons.keyboard_arrow_down,
                                        color: AppTheme.primaryColor,
                                        size: 18,
                                      ),
                                    ],
                                  ),
                                ),
                                
                                // Advanced parameters section
                                if (_showAdvancedParameters) ...[
                                  const SizedBox(height: 16),
                                  Row(
                                    children: [
                                      Expanded(
                                        child: CustomInputField(
                                          hintText: 'Height (cm)',
                                          controller: _heightController,
                                          keyboardType: const TextInputType.numberWithOptions(decimal: true),
                                          validator: (value) {
                                            if (value != null && value.isNotEmpty) {
                                              final height = double.tryParse(value);
                                              if (height == null || height < 50 || height > 250) {
                                                return 'Invalid';
                                              }
                                            }
                                            return null;
                                          },
                                        ),
                                      ),
                                      const SizedBox(width: 12),
                                      Expanded(
                                        child: CustomInputField(
                                          hintText: 'Weight (kg)',
                                          controller: _weightController,
                                          keyboardType: const TextInputType.numberWithOptions(decimal: true),
                                          validator: (value) {
                                            if (value != null && value.isNotEmpty) {
                                              final weight = double.tryParse(value);
                                              if (weight == null || weight < 2 || weight > 500) {
                                                return 'Invalid';
                                              }
                                            }
                                            return null;
                                          },
                                        ),
                                      ),
                                    ],
                                  ),
                                  const SizedBox(height: 12),
                                  Row(
                                    children: [
                                      Expanded(
                                        child: CustomInputField(
                                          hintText: 'Heart Rate (bpm)',
                                          controller: _heartRateController,
                                          keyboardType: TextInputType.number,
                                          validator: (value) {
                                            if (value != null && value.isNotEmpty) {
                                              final heartRate = int.tryParse(value);
                                              if (heartRate == null || heartRate < 30 || heartRate > 220) {
                                                return 'Invalid';
                                              }
                                            }
                                            return null;
                                          },
                                        ),
                                      ),
                                      const SizedBox(width: 12),
                                      Expanded(
                                        child: CustomInputField(
                                          hintText: 'SpO2 (%)',
                                          controller: _spo2Controller,
                                          keyboardType: TextInputType.number,
                                          validator: (value) {
                                            if (value != null && value.isNotEmpty) {
                                              final spo2 = int.tryParse(value);
                                              if (spo2 == null || spo2 < 70 || spo2 > 100) {
                                                return 'Invalid';
                                              }
                                            }
                                            return null;
                                          },
                                        ),
                                      ),
                                    ],
                                  ),
                                ],
                                
                                // Error message
                                if (_errorMessage != null) ...[
                                  const SizedBox(height: 16),
                                  Container(
                                    padding: const EdgeInsets.all(10),
                                    decoration: BoxDecoration(
                                      color: Colors.red[50],
                                      borderRadius: BorderRadius.circular(10),
                                    ),
                                    child: Row(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        const Icon(Icons.error_outline, color: Colors.red),
                                        const SizedBox(width: 10),
                                        Expanded(
                                          child: Text(
                                            _errorMessage!,
                                            style: const TextStyle(color: Colors.red),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  
                  // Bottom analyze button
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(16.0),
                    child: CustomButton(
                      text: 'Analyze Symptoms',
                      onPressed: _selectedSymptoms.isEmpty ? null : _analyzeSymptoms,
                      isLoading: _isProcessing,
                      icon: Icons.health_and_safety_outlined,
                    ),
                  ),
                ],
              ),
            ),
    );
  }
}