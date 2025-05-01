import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:medlens/utils/app_theme.dart';
import 'package:provider/provider.dart';
import 'package:medlens/services/auth_service.dart';
import 'package:medlens/services/theme_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:url_launcher/url_launcher.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({Key? key}) : super(key: key);

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool _notificationsEnabled = true;
  bool _isLoadingAppInfo = true;
  String _appVersion = '';
  bool _isLoadingSignOut = false;

  @override
  void initState() {
    super.initState();
    _loadSettings();
    _loadAppInfo();
  }

  Future<void> _loadSettings() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      setState(() {
        _notificationsEnabled = prefs.getBool('notifications_enabled') ?? true;
      });
    } catch (e) {
      // Fallback to defaults if preferences can't be loaded
    }
  }

  Future<void> _loadAppInfo() async {
    try {
      final packageInfo = await PackageInfo.fromPlatform();
      setState(() {
        _appVersion = 'v${packageInfo.version} (${packageInfo.buildNumber})';
        _isLoadingAppInfo = false;
      });
    } catch (e) {
      setState(() {
        _appVersion = 'v1.0.0';
        _isLoadingAppInfo = false;
      });
    }
  }

  Future<void> _toggleNotifications(bool value) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setBool('notifications_enabled', value);
      setState(() {
        _notificationsEnabled = value;
      });
      
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(value 
              ? 'Notifications enabled' 
              : 'Notifications disabled'
            ),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Failed to update notification settings'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _toggleDarkMode(bool value) async {
    try {
      final themeProvider = Provider.of<ThemeProvider>(context, listen: false);
      
      // Toggle theme using the provider
      await themeProvider.setThemeMode(value ? ThemeMode.dark : ThemeMode.light);
      
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(value 
              ? 'Dark mode enabled' 
              : 'Light mode enabled'
            ),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Failed to update theme settings'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _signOut() async {
    try {
      setState(() {
        _isLoadingSignOut = true;
      });
      
      final authService = Provider.of<AuthService>(context, listen: false);
      await authService.signOut();
      
      // Navigator.pop should happen automatically due to the AuthWrapper
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoadingSignOut = false;
        });
        
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to sign out: ${e.toString()}'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _clearData() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      
      // Save the theme setting before clearing
      final themeMode = Provider.of<ThemeProvider>(context, listen: false).themeMode;
      
      await prefs.clear();

      // Restore theme setting
      final themeProvider = Provider.of<ThemeProvider>(context, listen: false);
      await themeProvider.setThemeMode(themeMode);

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('App data cleared'),
          ),
        );
        
        // Reload settings after clearing
        _loadSettings();
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Failed to clear app data'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _resetPassword() async {
    try {
      final user = FirebaseAuth.instance.currentUser;
      if (user != null) {
        await FirebaseAuth.instance.sendPasswordResetEmail(email: user.email!);
        
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Password reset email sent. Please check your inbox.'),
            ),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to send password reset email: ${e.toString()}'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  Future<void> _launchURL(String url) async {
    final uri = Uri.parse(url);
    if (!await launchUrl(uri, mode: LaunchMode.externalApplication)) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Could not launch $url'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  void _showDeleteAccountDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Account'),
        content: const Text(
          'Are you sure you want to permanently delete your account? This action cannot be undone and all your data will be lost.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('CANCEL'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              _deleteAccount();
            },
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('DELETE'),
          ),
        ],
      ),
    );
  }

  Future<void> _deleteAccount() async {
    try {
      setState(() {
        _isLoadingSignOut = true;
      });
      
      final user = FirebaseAuth.instance.currentUser;
      if (user != null) {
        // Delete user data in Firestore
        await FirebaseFirestore.instance.collection('users').doc(user.uid).delete();
        
        // Delete user authentication account
        await user.delete();
        
        // Sign out and return to login screen will happen automatically due to AuthWrapper
      }
    } catch (e) {
      setState(() {
        _isLoadingSignOut = false;
      });
      
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to delete account: ${e.toString()}'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    // Get theme provider to check current theme mode
    final themeProvider = Provider.of<ThemeProvider>(context);
    final isDarkMode = themeProvider.isDarkMode;
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
      ),
      body: _isLoadingSignOut
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text('Signing out...'),
                ],
              ),
            )
          : ListView(
              children: [
                // App Preferences
                _buildSectionHeader('App Preferences'),
                SwitchListTile(
                  title: const Text('Enable Notifications'),
                  subtitle: const Text('Receive alerts for important updates'),
                  value: _notificationsEnabled,
                  onChanged: _toggleNotifications,
                  activeColor: AppTheme.primaryColor,
                ),
                SwitchListTile(
                  title: const Text('Dark Mode'),
                  subtitle: const Text('Use dark theme throughout the app'),
                  value: isDarkMode,
                  onChanged: _toggleDarkMode,
                  activeColor: AppTheme.primaryColor,
                ),
                const Divider(),
                
                // Account Settings
                _buildSectionHeader('Account Settings'),
                ListTile(
                  leading: const Icon(Icons.lock_outline),
                  title: const Text('Reset Password'),
                  subtitle: const Text('Send password reset email'),
                  onTap: _resetPassword,
                ),
                ListTile(
                  leading: const Icon(Icons.logout),
                  title: const Text('Sign Out'),
                  onTap: _signOut,
                ),
                ListTile(
                  leading: const Icon(Icons.delete_outline, color: Colors.red),
                  title: const Text('Delete Account', 
                    style: TextStyle(color: Colors.red)),
                  subtitle: const Text('Permanently remove your account'),
                  onTap: _showDeleteAccountDialog,
                ),
                const Divider(),
                
                // About & Support
                _buildSectionHeader('About & Support'),
                ListTile(
                  leading: const Icon(Icons.help_outline),
                  title: const Text('Help Center'),
                  onTap: () => _launchURL('https://medlens.example.com/help'),
                ),
                ListTile(
                  leading: const Icon(Icons.privacy_tip_outlined),
                  title: const Text('Privacy Policy'),
                  onTap: () => _launchURL('https://medlens.example.com/privacy'),
                ),
                ListTile(
                  leading: const Icon(Icons.description_outlined),
                  title: const Text('Terms of Service'),
                  onTap: () => _launchURL('https://medlens.example.com/terms'),
                ),
                ListTile(
                  leading: const Icon(Icons.info_outline),
                  title: const Text('About MedLens'),
                  subtitle: Text(_isLoadingAppInfo ? 'Loading...' : _appVersion),
                ),
                const Divider(),
                
                // Danger Zone
                _buildSectionHeader('Danger Zone', color: Colors.red),
                ListTile(
                  leading: const Icon(Icons.delete_forever, color: Colors.red),
                  title: const Text('Clear App Data',
                    style: TextStyle(color: Colors.red)),
                  subtitle: const Text(
                    'Reset all app settings (doesn\'t delete your account)',
                    style: TextStyle(color: Colors.red),
                  ),
                  onTap: _clearData,
                ),
                const SizedBox(height: 40),
              ],
            ),
    );
  }

  Widget _buildSectionHeader(String title, {Color? color}) {
    // Use current theme's text color if no specific color is provided
    final textColor = color ?? Theme.of(context).textTheme.bodyLarge?.color ?? Colors.black87;
    
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      child: Text(
        title.toUpperCase(),
        style: TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.bold,
          color: textColor,
          letterSpacing: 0.5,
        ),
      ),
    );
  }
}