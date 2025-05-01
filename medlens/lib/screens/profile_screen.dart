import 'dart:io';
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:image_picker/image_picker.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:medlens/utils/app_theme.dart';
import 'package:medlens/widgets/custom_button.dart';
import 'package:medlens/widgets/custom_input_field.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({Key? key}) : super(key: key);

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  final _formKey = GlobalKey<FormState>();
  final _nameController = TextEditingController();
  final _phoneController = TextEditingController();
  final _ageController = TextEditingController();
  final _genderController = TextEditingController();
  final _heightController = TextEditingController();
  final _weightController = TextEditingController();
  final _bloodGroupController = TextEditingController();
  final _allergiesController = TextEditingController();
  final _medicalConditionsController = TextEditingController();

  bool _isLoading = true;
  bool _isSaving = false;
  bool _hasError = false;
  String? _errorMessage;
  String? _profileImageUrl;
  File? _profileImageFile;
  bool _isRetrying = false;

  @override
  void initState() {
    super.initState();
    _loadUserProfile();
  }

  @override
  void dispose() {
    _nameController.dispose();
    _phoneController.dispose();
    _ageController.dispose();
    _genderController.dispose();
    _heightController.dispose();
    _weightController.dispose();
    _bloodGroupController.dispose();
    _allergiesController.dispose();
    _medicalConditionsController.dispose();
    super.dispose();
  }

  Future<void> _loadUserProfile() async {
    try {
      setState(() {
        _isLoading = true;
        _hasError = false;
        _isRetrying = false;
      });

      final user = FirebaseAuth.instance.currentUser;
      if (user == null) {
        throw Exception('No user logged in');
      }

      // Initialize with basic info from Firebase Auth
      _nameController.text = user.displayName ?? '';
      _profileImageUrl = user.photoURL;

      // Add error handling with a timeout
      try {
        // Get user profile from Firestore
        final doc = await FirebaseFirestore.instance
            .collection('users')
            .doc(user.uid)
            .get()
            .timeout(const Duration(seconds: 10));

        if (doc.exists) {
          final data = doc.data()!;
          setState(() {
            _nameController.text = user.displayName ?? data['displayName'] ?? '';
            _profileImageUrl = user.photoURL ?? data['photoURL'];
            _phoneController.text = data['phone'] ?? '';
            _ageController.text = data['age']?.toString() ?? '';
            _genderController.text = data['gender'] ?? '';
            _heightController.text = data['height']?.toString() ?? '';
            _weightController.text = data['weight']?.toString() ?? '';
            _bloodGroupController.text = data['bloodGroup'] ?? '';
            _allergiesController.text = data['allergies'] ?? '';
            _medicalConditionsController.text = data['medicalConditions'] ?? '';
          });
        } else {
          // Create a basic user document if it doesn't exist yet
          await FirebaseFirestore.instance
              .collection('users')
              .doc(user.uid)
              .set({
                'displayName': user.displayName ?? '',
                'email': user.email ?? '',
                'photoURL': user.photoURL,
                'createdAt': FieldValue.serverTimestamp(),
                'updatedAt': FieldValue.serverTimestamp(),
              }, SetOptions(merge: true));
        }
      } catch (e) {
        // If Firestore is unavailable, continue with basic Auth info
        print('Firestore error: $e - Using basic Auth profile info');
      }

      setState(() {
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _hasError = true;
        _errorMessage = 'Failed to load profile: ${e.toString()}';
      });
    }
  }

  Future<void> _saveProfile() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    try {
      setState(() {
        _isSaving = true;
        _hasError = false;
      });

      final user = FirebaseAuth.instance.currentUser;
      if (user == null) {
        throw Exception('No user logged in');
      }

      String? photoURL = _profileImageUrl;

      // Upload profile image if a new one was selected
      if (_profileImageFile != null) {
        try {
          final storageRef = FirebaseStorage.instance
              .ref()
              .child('profile_images')
              .child('${user.uid}.jpg');
          
          await storageRef.putFile(_profileImageFile!);
          photoURL = await storageRef.getDownloadURL();
          
          // Update Firebase Auth profile with new photo URL
          await user.updatePhotoURL(photoURL);
        } catch (e) {
          print('Failed to upload profile image: $e');
          // Continue with profile saving even if image upload fails
        }
      }

      // Update name in Firebase Auth if it changed
      if (user.displayName != _nameController.text) {
        try {
          await user.updateDisplayName(_nameController.text);
        } catch (e) {
          print('Failed to update display name in Firebase Auth: $e');
          // Continue with Firestore update even if Auth update fails
        }
      }

      // Create user profile data
      final userData = {
        'displayName': _nameController.text,
        'photoURL': photoURL,
        'phone': _phoneController.text,
        'email': user.email,
        'age': _ageController.text.isNotEmpty ? int.tryParse(_ageController.text) : null,
        'gender': _genderController.text,
        'height': _heightController.text.isNotEmpty ? double.tryParse(_heightController.text) : null,
        'weight': _weightController.text.isNotEmpty ? double.tryParse(_weightController.text) : null,
        'bloodGroup': _bloodGroupController.text,
        'allergies': _allergiesController.text,
        'medicalConditions': _medicalConditionsController.text,
        'updatedAt': FieldValue.serverTimestamp(),
      };

      // Remove any null values to prevent overwriting existing data with nulls
      userData.removeWhere((key, value) => value == null);

      try {
        // Use a transaction for atomicity
        await FirebaseFirestore.instance.runTransaction((transaction) async {
          final docRef = FirebaseFirestore.instance.collection('users').doc(user.uid);
          transaction.set(docRef, userData, SetOptions(merge: true));
        }).timeout(const Duration(seconds: 15));

        if (mounted) {
          setState(() {
            _isSaving = false;
          });
          
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Profile updated successfully'),
              backgroundColor: Colors.green,
              duration: Duration(seconds: 2),
            ),
          );
        }
      } catch (firestoreError) {
        print('Failed to save profile to Firestore: $firestoreError');
        // Try one more direct set operation as fallback
        try {
          await FirebaseFirestore.instance
              .collection('users')
              .doc(user.uid)
              .set(userData, SetOptions(merge: true))
              .timeout(const Duration(seconds: 10));
              
          if (mounted) {
            setState(() {
              _isSaving = false;
            });
            
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('Profile updated successfully'),
                backgroundColor: Colors.green,
              ),
            );
          }
        } catch (retryError) {
          if (mounted) {
            setState(() {
              _isSaving = false;
              _hasError = true;
            });
            
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text('Failed to save profile details. Please check your internet connection.'),
                backgroundColor: Colors.red,
                action: SnackBarAction(
                  label: 'RETRY',
                  onPressed: _saveProfile,
                ),
              ),
            );
          }
        }
      }
    } catch (e) {
      print('Profile save error: $e');
      if (mounted) {
        setState(() {
          _isSaving = false;
          _hasError = true;
        });
        
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to update profile. Please try again.'),
            backgroundColor: Colors.red,
            action: SnackBarAction(
              label: 'RETRY',
              onPressed: _saveProfile,
            ),
          ),
        );
      }
    }
  }

  Future<void> _pickImage() async {
    try {
      final picker = ImagePicker();
      final pickedFile = await picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 512,
        maxHeight: 512,
        imageQuality: 75,
      );

      if (pickedFile != null) {
        setState(() {
          _profileImageFile = File(pickedFile.path);
        });
      }
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Failed to pick image: ${e.toString()}'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('My Profile'),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _hasError
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text(_errorMessage ?? 'Failed to load profile.'),
                      const SizedBox(height: 16),
                      ElevatedButton(
                        onPressed: _isRetrying ? null : () {
                          setState(() {
                            _isRetrying = true;
                          });
                          _loadUserProfile();
                        },
                        child: _isRetrying 
                            ? const SizedBox(
                                height: 20,
                                width: 20,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                ),
                              )
                            : const Text('Retry'),
                      ),
                    ],
                  ),
                )
              : _buildProfileForm(),
    );
  }

  Widget _buildProfileForm() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20.0),
      child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            // Profile Image
            GestureDetector(
              onTap: _pickImage,
              child: Stack(
                children: [
                  CircleAvatar(
                    radius: 60,
                    backgroundColor: Colors.grey[200],
                    backgroundImage: _profileImageFile != null
                        ? FileImage(_profileImageFile!)
                        : _profileImageUrl != null
                            ? NetworkImage(_profileImageUrl!)
                            : null,
                    child: (_profileImageUrl == null && _profileImageFile == null)
                        ? const Icon(
                            Icons.person,
                            size: 60,
                            color: Colors.grey,
                          )
                        : null,
                  ),
                  Positioned(
                    bottom: 0,
                    right: 0,
                    child: Container(
                      padding: const EdgeInsets.all(4),
                      decoration: const BoxDecoration(
                        shape: BoxShape.circle,
                        color: AppTheme.primaryColor,
                      ),
                      child: const Icon(
                        Icons.camera_alt,
                        color: Colors.white,
                        size: 20,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // Basic Information
            const Text(
              'Basic Information',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            CustomInputField(
              hintText: 'Full Name',
              controller: _nameController,
              prefixIcon: Icons.person_outline,
              validator: (value) {
                if (value == null || value.trim().isEmpty) {
                  return 'Please enter your name';
                }
                return null;
              },
            ),
            const SizedBox(height: 16),
            CustomInputField(
              hintText: 'Phone Number',
              controller: _phoneController,
              prefixIcon: Icons.phone_outlined,
              keyboardType: TextInputType.phone,
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: CustomInputField(
                    hintText: 'Age',
                    controller: _ageController,
                    prefixIcon: Icons.calendar_today_outlined,
                    keyboardType: TextInputType.number,
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: CustomInputField(
                    hintText: 'Gender',
                    controller: _genderController,
                    prefixIcon: Icons.person_outline,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),

            // Health Information
            const Text(
              'Health Information',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: CustomInputField(
                    hintText: 'Height (cm)',
                    controller: _heightController,
                    prefixIcon: Icons.height_outlined,
                    keyboardType: TextInputType.number,
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: CustomInputField(
                    hintText: 'Weight (kg)',
                    controller: _weightController,
                    prefixIcon: Icons.monitor_weight_outlined,
                    keyboardType: TextInputType.number,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            CustomInputField(
              hintText: 'Blood Group',
              controller: _bloodGroupController,
              prefixIcon: Icons.bloodtype_outlined,
            ),
            const SizedBox(height: 16),
            CustomInputField(
              hintText: 'Allergies',
              controller: _allergiesController,
              maxLines: 3,
              contentPadding: const EdgeInsets.symmetric(
                vertical: 16,
                horizontal: 20,
              ),
            ),
            const SizedBox(height: 16),
            CustomInputField(
              hintText: 'Medical Conditions',
              controller: _medicalConditionsController,
              maxLines: 3,
              contentPadding: const EdgeInsets.symmetric(
                vertical: 16,
                horizontal: 20,
              ),
            ),
            const SizedBox(height: 32),

            // Save Button
            SizedBox(
              width: double.infinity,
              child: CustomButton(
                text: 'Save Profile',
                onPressed: _saveProfile,
                isLoading: _isSaving,
              ),
            ),
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }
}