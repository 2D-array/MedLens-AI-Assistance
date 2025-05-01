import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';
import 'package:medlens/models/user_model.dart';

class AuthService extends ChangeNotifier {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  
  // Explicit flag to track registration status
  bool _justRegistered = false;
  bool get justRegistered => _justRegistered;
  
  // Auth state changes
  Stream<User?> get authStateChanges => _auth.authStateChanges();
  
  // Get current user
  User? get currentUser => _auth.currentUser;
  
  // Check if email is verified
  bool get isEmailVerified {
    return currentUser?.emailVerified ?? false;
  }
  
  // Get current user as UserModel
  Future<UserModel?> get currentUserModel async {
    final user = currentUser;
    if (user != null) {
      // Reload user to get the latest verification status
      await user.reload();
      
      final doc = await _firestore.collection('users').doc(user.uid).get();
      if (doc.exists) {
        return UserModel.fromJson(doc.data()!);
      }
      return UserModel(
        uid: user.uid, 
        email: user.email,
        emailVerified: user.emailVerified,
      );
    }
    return null;
  }
  
  // Register with email and password
  Future<UserModel?> registerWithEmailAndPassword(
      String email, String password, String name) async {
    try {
      _justRegistered = false;
      
      final result = await _auth.createUserWithEmailAndPassword(
        email: email,
        password: password,
      );
      
      User? user = result.user;
      
      if (user != null) {
        // Update display name
        await user.updateDisplayName(name);
        
        // Send email verification immediately
        await user.sendEmailVerification();
        
        // Create user document in Firestore
        final userData = {
          'uid': user.uid,
          'email': user.email,
          'displayName': name,
          'photoURL': null,
          'emailVerified': false,
          'createdAt': FieldValue.serverTimestamp(),
        };
        
        await _firestore.collection('users').doc(user.uid).set(userData);
        
        // Set registration flag and create user model before signing out
        _justRegistered = true;
        final userModel = UserModel(
          uid: user.uid,
          email: user.email,
          displayName: name,
          emailVerified: false,
        );
        
        // Force sign out the user after registration
        await _auth.signOut();
        
        notifyListeners();
        return userModel;
      }
      return null;
    } catch (e) {
      _justRegistered = false;
      rethrow;
    }
  }
  
  // Sign in with email and password
  Future<UserModel?> signInWithEmailAndPassword(
      String email, String password) async {
    try {
      // Reset registration flag
      _justRegistered = false;
      
      final result = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );
      
      User? user = result.user;
      
      if (user != null) {
        // Reload user to get latest status
        await user.reload();
        
        // Check if email is verified
        if (!user.emailVerified) {
          // If not verified, sign out and throw an error
          await _auth.signOut();
          throw FirebaseAuthException(
            code: 'email-not-verified',
            message: 'Please verify your email before signing in.',
          );
        }
        
        // Update emailVerified status in Firestore
        await _firestore.collection('users').doc(user.uid).update({
          'emailVerified': true,
        });
        
        notifyListeners();
        return UserModel(
          uid: user.uid,
          email: user.email,
          displayName: user.displayName,
          photoURL: user.photoURL,
          emailVerified: user.emailVerified,
        );
      }
      return null;
    } catch (e) {
      rethrow;
    }
  }
  
  // Send email verification again
  Future<void> sendEmailVerificationAgain(String email, String password) async {
    try {
      // Sign in to send verification email
      final credential = EmailAuthProvider.credential(email: email, password: password);
      await _auth.signInWithCredential(credential);
      
      final user = _auth.currentUser;
      if (user != null && !user.emailVerified) {
        await user.sendEmailVerification();
        // Sign out after sending verification
        await _auth.signOut();
      }
    } catch (e) {
      rethrow;
    }
  }
  
  // Check if email is verified (reloads user data from server)
  Future<bool> checkEmailVerified() async {
    try {
      final user = currentUser;
      if (user != null) {
        await user.reload();
        return user.emailVerified;
      }
      return false;
    } catch (e) {
      return false;
    }
  }
  
  // Clear registration status
  void clearRegistrationStatus() {
    _justRegistered = false;
    notifyListeners();
  }
  
  // Sign out
  Future<void> signOut() async {
    await _auth.signOut();
    _justRegistered = false;
    notifyListeners();
  }
  
  // Reset password
  Future<void> resetPassword(String email) async {
    await _auth.sendPasswordResetEmail(email: email);
  }
}