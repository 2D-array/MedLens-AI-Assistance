import 'package:flutter/foundation.dart';
import 'package:medlens/utils/app_config.dart';

enum LogLevel {
  debug,
  info,
  warning,
  error,
}

class Logger {
  static final Logger _instance = Logger._internal();
  
  factory Logger() {
    return _instance;
  }
  
  Logger._internal();
  
  void log(String message, {LogLevel level = LogLevel.info, String? tag}) {
    if (!AppConfig.enableDebugLogs && level == LogLevel.debug) {
      return;
    }
    
    final timestamp = DateTime.now().toString();
    final logTag = tag ?? 'MedLens';
    
    String formattedMessage = '[$timestamp] [$logTag] [${_levelToString(level)}]: $message';
    
    // In debug mode, print to console
    if (kDebugMode) {
      debugPrint(formattedMessage);
    }
    
    // Could extend this to write to file or send to a remote logging service
    _logToAnalytics(level, message, tag);
  }
  
  void debug(String message, {String? tag}) {
    log(message, level: LogLevel.debug, tag: tag);
  }
  
  void info(String message, {String? tag}) {
    log(message, level: LogLevel.info, tag: tag);
  }
  
  void warning(String message, {String? tag}) {
    log(message, level: LogLevel.warning, tag: tag);
  }
  
  void error(String message, {String? tag, Object? error, StackTrace? stackTrace}) {
    String fullMessage = message;
    if (error != null) {
      fullMessage += '\nError: $error';
    }
    
    if (stackTrace != null && kDebugMode) {
      fullMessage += '\nStackTrace: $stackTrace';
    }
    
    log(fullMessage, level: LogLevel.error, tag: tag);
  }
  
  String _levelToString(LogLevel level) {
    switch (level) {
      case LogLevel.debug:
        return 'DEBUG';
      case LogLevel.info:
        return 'INFO';
      case LogLevel.warning:
        return 'WARNING';
      case LogLevel.error:
        return 'ERROR';
    }
  }
  
  void _logToAnalytics(LogLevel level, String message, String? tag) {
    // Only log non-debug messages to analytics in release mode
    if (level != LogLevel.debug && AppConfig.enableAnalytics && !kDebugMode) {
      // Analytics integration would go here
      // Example: FirebaseAnalytics.instance.logEvent(...);
    }
  }
}