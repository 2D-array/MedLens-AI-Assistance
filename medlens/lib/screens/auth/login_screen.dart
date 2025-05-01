import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:medlens/services/auth_service.dart';
import 'package:medlens/utils/app_theme.dart';
import 'package:medlens/widgets/custom_button.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({Key? key}) : super(key: key);

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  final _nameController = TextEditingController();
  bool _isLoading = false;
  bool _isSignUp = false;
  String? _errorMessage;
  String? _successMessage;

  @override
  void initState() {
    super.initState();
    // Check if user has just registered
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final authService = Provider.of<AuthService>(context, listen: false);
      if (authService.justRegistered) {
        setState(() {
          _successMessage = 'Registration successful! Please check your email to verify your account.';
          authService.clearRegistrationStatus();
        });
      }
    });
  }

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    _nameController.dispose();
    super.dispose();
  }

  // Changed to void return type
  void _submit() {
    if (!_formKey.currentState!.validate()) {
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
      _successMessage = null;
    });

    final email = _emailController.text.trim();
    final password = _passwordController.text.trim();

    try {
      final authService = Provider.of<AuthService>(context, listen: false);
      
      if (_isSignUp) {
        authService.registerWithEmailAndPassword(
          email,
          password,
          _nameController.text.trim(),
        ).then((_) {
          setState(() {
            _successMessage = 'Registration successful! Please check your email to verify your account.';
            _isSignUp = false; // Switch back to login view
          });
        }).catchError((e) {
          setState(() {
            _errorMessage = e.toString().contains('firebase')
                ? 'Registration failed. Please check your information and try again.'
                : e.toString();
          });
        }).whenComplete(() {
          if (mounted) {
            setState(() {
              _isLoading = false;
            });
          }
        });
      } else {
        authService.signInWithEmailAndPassword(
          email,
          password,
        ).catchError((e) {
          setState(() {
            if (e.code == 'email-not-verified') {
              _errorMessage = 'Please verify your email before signing in.';
              // Save credentials temporarily for resend verification
              _showResendVerificationDialog(email, password);
            } else {
              _errorMessage = e.toString().contains('firebase')
                  ? 'Authentication failed. Please check your credentials.'
                  : e.toString();
            }
          });
        }).whenComplete(() {
          if (mounted) {
            setState(() {
              _isLoading = false;
            });
          }
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = e.toString();
          _isLoading = false;
        });
      }
    }
  }

  void _showResendVerificationDialog(String email, String password) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Email Not Verified'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Your email address has not been verified.'),
            const SizedBox(height: 8),
            const Text('Do you want to resend the verification email?'),
            const SizedBox(height: 8),
            Text(
              'Email: $email',
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              final authService = Provider.of<AuthService>(context, listen: false);
              Navigator.pop(context); // Close dialog first
              
              setState(() {
                _isLoading = true;
              });
              
              authService.sendEmailVerificationAgain(email, password).then((_) {
                setState(() {
                  _successMessage = 'Verification email sent. Please check your inbox.';
                  _isLoading = false;
                });
              }).catchError((error) {
                setState(() {
                  _errorMessage = 'Error sending verification email: ${error.toString()}';
                  _isLoading = false;
                });
              });
            },
            child: const Text('Resend'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          // Background gradient
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  AppTheme.primaryColor.withOpacity(0.8),
                  AppTheme.primaryColor.withOpacity(0.3),
                ],
              ),
            ),
          ),
          
          // Content
          SafeArea(
            child: Center(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(24.0),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    // Logo and title
                    const Icon(
                      Icons.medical_services_outlined,
                      size: 80,
                      color: Colors.white,
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'MedLens',
                      style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _isSignUp ? 'Create an account' : 'Sign in to continue',
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: 40),
                    
                    // Success message
                    if (_successMessage != null) ...[
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.green.shade50,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(color: Colors.green.shade200),
                        ),
                        child: Row(
                          children: [
                            const Icon(Icons.check_circle, color: Colors.green),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                _successMessage!,
                                style: TextStyle(color: Colors.green.shade700),
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 20),
                    ],
                    
                    // Form card
                    Card(
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                      elevation: 4,
                      child: Padding(
                        padding: const EdgeInsets.all(20.0),
                        child: Form(
                          key: _formKey,
                          child: Column(
                            children: [
                              // Name field (only for signup)
                              if (_isSignUp) ...[
                                TextFormField(
                                  controller: _nameController,
                                  decoration: const InputDecoration(
                                    labelText: 'Name',
                                    prefixIcon: Icon(Icons.person_outline),
                                  ),
                                  validator: (value) {
                                    if (value == null || value.trim().isEmpty) {
                                      return 'Please enter your name';
                                    }
                                    return null;
                                  },
                                ),
                                const SizedBox(height: 16),
                              ],
                              
                              // Email field
                              TextFormField(
                                controller: _emailController,
                                decoration: const InputDecoration(
                                  labelText: 'Email',
                                  prefixIcon: Icon(Icons.email_outlined),
                                ),
                                keyboardType: TextInputType.emailAddress,
                                validator: (value) {
                                  if (value == null || value.trim().isEmpty) {
                                    return 'Please enter your email';
                                  }
                                  if (!value.contains('@') || !value.contains('.')) {
                                    return 'Please enter a valid email';
                                  }
                                  return null;
                                },
                              ),
                              const SizedBox(height: 16),
                              
                              // Password field
                              TextFormField(
                                controller: _passwordController,
                                decoration: const InputDecoration(
                                  labelText: 'Password',
                                  prefixIcon: Icon(Icons.lock_outline),
                                ),
                                obscureText: true,
                                validator: (value) {
                                  if (value == null || value.isEmpty) {
                                    return 'Please enter your password';
                                  }
                                  if (_isSignUp && value.length < 6) {
                                    return 'Password must be at least 6 characters';
                                  }
                                  return null;
                                },
                              ),
                              
                              // Error message
                              if (_errorMessage != null) ...[
                                const SizedBox(height: 16),
                                Container(
                                  padding: const EdgeInsets.all(8),
                                  decoration: BoxDecoration(
                                    color: Colors.red.shade50,
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Row(
                                    children: [
                                      Icon(Icons.error_outline, color: Colors.red.shade700, size: 18),
                                      const SizedBox(width: 8),
                                      Expanded(
                                        child: Text(
                                          _errorMessage!,
                                          style: TextStyle(
                                            color: Colors.red.shade700,
                                            fontSize: 14,
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ],
                              
                              const SizedBox(height: 24),
                              
                              // Submit button - fixed onPressed to not use async directly
                              SizedBox(
                                width: double.infinity,
                                child: CustomButton(
                                  text: _isSignUp ? 'Sign Up' : 'Sign In',
                                  onPressed: _isLoading ? () {} : _submit,
                                  isLoading: _isLoading,
                                ),
                              ),
                              
                              // Toggle sign in/sign up
                              const SizedBox(height: 16),
                              TextButton(
                                onPressed: () {
                                  setState(() {
                                    _isSignUp = !_isSignUp;
                                    _errorMessage = null;
                                    _successMessage = null;
                                  });
                                },
                                child: Text(
                                  _isSignUp
                                      ? 'Already have an account? Sign In'
                                      : 'Don\'t have an account? Sign Up',
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 24),
                    
                    // Forgot password link
                    if (!_isSignUp)
                      TextButton(
                        onPressed: () {
                          // Implement forgot password
                          if (_emailController.text.trim().isNotEmpty &&
                              _emailController.text.contains('@')) {
                            final authService =
                                Provider.of<AuthService>(context, listen: false);
                            authService
                                .resetPassword(_emailController.text.trim())
                                .then((_) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                const SnackBar(
                                  content: Text(
                                    'Password reset email sent. Check your inbox.',
                                  ),
                                ),
                              );
                            }).catchError((error) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(
                                  content: Text(
                                    'Error: ${error.toString()}',
                                  ),
                                  backgroundColor: Colors.red,
                                ),
                              );
                            });
                          } else {
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                content: Text(
                                  'Please enter a valid email first',
                                ),
                              ),
                            );
                          }
                        },
                        child: const Text('Forgot Password?'),
                      ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}