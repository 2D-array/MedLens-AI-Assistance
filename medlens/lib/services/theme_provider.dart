import 'package:flutter/material.dart';
import 'package:medlens/utils/app_theme.dart';
import 'package:shared_preferences/shared_preferences.dart';

class ThemeProvider with ChangeNotifier {
  // Theme key in SharedPreferences
  static const String _themeKey = 'theme_mode';
  
  // Default to system theme
  ThemeMode _themeMode = ThemeMode.light;
  
  // Getters
  ThemeMode get themeMode => _themeMode;
  bool get isDarkMode => _themeMode == ThemeMode.dark;
  
  // Constructor - load theme from SharedPreferences
  ThemeProvider() {
    _loadTheme();
  }
  
  // Load saved theme
  Future<void> _loadTheme() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final savedTheme = prefs.getString(_themeKey);
      
      if (savedTheme != null) {
        _themeMode = _getThemeModeFromString(savedTheme);
        notifyListeners();
      }
    } catch (e) {
      // Default to light theme if there's an error
      _themeMode = ThemeMode.light;
    }
  }
  
  // Convert string to ThemeMode
  ThemeMode _getThemeModeFromString(String themeString) {
    switch (themeString) {
      case 'dark':
        return ThemeMode.dark;
      case 'light':
        return ThemeMode.light;
      case 'system':
        return ThemeMode.system;
      default:
        return ThemeMode.light;
    }
  }
  
  // Toggle between light and dark themes
  Future<void> toggleTheme() async {
    _themeMode = (_themeMode == ThemeMode.light) ? ThemeMode.dark : ThemeMode.light;
    
    // Save the new theme preference
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_themeKey, _themeMode == ThemeMode.dark ? 'dark' : 'light');
    
    // Notify listeners to rebuild the UI with the new theme
    notifyListeners();
  }
  
  // Set specific theme mode
  Future<void> setThemeMode(ThemeMode mode) async {
    if (_themeMode == mode) return;
    
    _themeMode = mode;
    
    // Save the new theme preference
    final prefs = await SharedPreferences.getInstance();
    String themeModeString;
    
    switch (mode) {
      case ThemeMode.dark:
        themeModeString = 'dark';
        break;
      case ThemeMode.light:
        themeModeString = 'light';
        break;
      case ThemeMode.system:
        themeModeString = 'system';
        break;
    }
    
    await prefs.setString(_themeKey, themeModeString);
    
    // Notify listeners to rebuild the UI with the new theme
    notifyListeners();
  }
}