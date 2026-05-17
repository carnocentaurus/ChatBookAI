// main.dart

import 'dart:io'; // Gives access to platform information (e.g., Windows, Android) and file I/O operations.
import 'dart:convert'; // Allows JSON encoding/decoding (used to convert data to/from the backend API).
import 'package:flutter/material.dart'; // Core Flutter UI library — provides widgets, layouts, and themes.
import 'package:http/http.dart' as http; // Enables HTTP requests to communicate with the FastAPI backend.
import 'package:window_size/window_size.dart'; // Lets you control the window's size, title, and position (desktop only).
import 'package:flutter/foundation.dart' show kIsWeb; // Added to safely check for Web platform.

// Import pages
import 'chat.dart';

void main() { // The entry point of every Flutter app. Execution starts here.
  WidgetsFlutterBinding.ensureInitialized(); // Initializes Flutter engine bindings required for platform channel and native desktop method calls (like setWindowTitle).

  if (!kIsWeb && (Platform.isWindows || Platform.isLinux || Platform.isMacOS)) {
    setWindowTitle('ChatBook AI'); // Only visible on desktop title bars.
    //setWindowMinSize(const Size(500, 1000)); // Defines the smallest possible window size (width: 500px, height: 1000px).  
    //setWindowMaxSize(const Size(500, 1000)); // Defines the largest possible window size (same as min size).
    //setWindowFrame(const Rect.fromLTWH(50, 50, 500, 1000)); // Positions the window at (50, 50) with a size of 500x1000 pixels upon opening.
  }

  runApp(const SplashWrapper()); // Launches the root Flutter widget (`MyApp`) which builds the entire UI. 
}

const bool useProduction = false;  // true = Render, false = local

String getBaseUrl() {
  if (useProduction) {
    return "https://chatbookai-render-2.onrender.com";  // Production
  } 
  else {
    // Local testing
    if (kIsWeb) {
      return "http://localhost:8000"; 
    }
    if (Platform.isWindows || Platform.isLinux || Platform.isMacOS) {
      return "http://127.0.0.1:8000";
    } 
    else if (Platform.isAndroid || Platform.isIOS) {
      return "http://10.226.63.213:8000";
    }
  }
  throw UnsupportedError("Unsupported platform");
}

Future<bool> isServerReady() async {
  try {
    final response = await http.get(
      Uri.parse("${getBaseUrl()}/health")
    );
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['status'] == 'healthy';
    }
  } 
  catch (e) {
    return false;
  }
  return false;
}


// Creates a unique identifier for each app session by using the current timestamp in milliseconds.  
String _sessionId = DateTime.now().millisecondsSinceEpoch.toString();

// Getter function to access session ID from other files
String getSessionId() => _sessionId;


// This function sends the question to the chatbot backend, waits for a response, and returns the chatbot's answer
Future<String> queryHandbook(String question) async {
  final url = Uri.parse("${getBaseUrl()}/chat");
  final logUrl = Uri.parse("${getBaseUrl()}/log_to_csv");

  try {
    // 1. Get the answer from Gemini
    final response = await http.post(
      url,
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"query": question, "session_id": _sessionId}),
    );

    if (response.statusCode == 200) {
      final responseData = jsonDecode(response.body);
      String answer = responseData["answer"] ?? "No response received";

      // 2. Log to server
      // remove .catchError and wrapped it in a separate try/catch 
      // to keep it clean and avoid the type mismatch error.
      _sendAndForgetLog(logUrl, question, answer);

      return answer;
    } else {
      return "Server error: ${response.statusCode}";
    }
  } catch (e) {
    return "⚠️ Cannot connect to server. Error: $e";
  }
}


// Separate helper to keep queryHandbook clean and fix the 'void' return error
void _sendAndForgetLog(Uri url, String query, String answer) async {
  try {
    await http.post(
      url,
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({
        "query": query,
        "answer": answer,
        "session_id": _sessionId,
      }),
    );
  } catch (e) {
    // debugPrint is preferred over print for production code
    debugPrint("Background logging failed: $e");
  }
}


// Fetch reports API
Future<Map<String, dynamic>> fetchReports() async {
  final url = Uri.parse("${getBaseUrl()}/report"); // Link to the report page in the backend
  try {
    final response = await http.get(url); // Get data from the backend
    if (response.statusCode == 200) { // If the server reply is OK
      final data = jsonDecode(response.body) as Map<String, dynamic>; // Turn the reply into a readable format
      return {
        'total_queries': data['total_queries'] ?? 0,
        'answered_queries': data['answered_queries'] ?? 0,
        'not_found_queries': data['not_found_queries'] ?? 0,
        'failed_queries': data['failed_queries'] ?? 0,
        'accuracy_rate': data['accuracy_rate'] ?? 0.0, 
        'most_frequent_questions': data['most_frequent_questions'] ?? [],
      };
    }
  } 
  catch (_) {} // If something goes wrong, skip it
  return {
    'total_queries': 0, // Default values when no data is found
    'answered_queries': 0,
    'not_found_queries': 0,
    'failed_queries': 0,
    'accuracy_rate': 0.0,
    'most_frequent_questions': [],
  };
}


// Submit feedback API
Future<Map<String, dynamic>> submitFeedback(
    String feedbackText, int rating, String userType, String sessionId) async {
  final url = Uri.parse("${getBaseUrl()}/feedback"); // Link to the feedback page in the backend
  try {
    final response = await http.post( // Send data to the backend
      url,
      headers: {"Content-Type": "application/json"}, // Tell the server we're sending JSON data
      body: jsonEncode({
        "feedback_text": feedbackText,
        "rating": rating, 
        "user_type": userType,
        "session_id": sessionId 
      }),
    );

    if (response.statusCode == 200) { // If the server reply is OK
      final data = jsonDecode(response.body); // Turn the reply into a readable format
      return {
        "success": true, // Mark as success
        "message": data["message"] ?? "Feedback submitted successfully" // Show message from server
      };
    } 
    else {
      return {
        "success": false, // Mark as failed
        "message": "Server error: ${response.statusCode}" // Show server error message
      };
    }
  } 
  catch (e) {
    return {"success": false, "message": "Cannot connect to server: $e"}; // Show error if the server can't be reached
  }
}


class SplashWrapper extends StatefulWidget { // A widget that can update while the app is running
  const SplashWrapper({super.key}); // gives Flutter a small label it can use to keep track of this widget

  @override
  // createState() tells Flutter which "helper object" should control this widget, and it creates an _SplashWrapperState to do that
  State<SplashWrapper> createState() => _SplashWrapperState(); 
}


class _SplashWrapperState extends State<SplashWrapper> { // This class controls what happens during the splash screen
  bool _showMainApp = false; // Starts as false because we want to show the splash first

  @override
  void initState() { // Runs automatically when this screen first appears
    super.initState(); // Keeps Flutter's setup working properly
    // Wait 4 seconds before showing the main app
    Future.delayed(const Duration(seconds: 4), () { 
      setState(() { // Updates the screen
        _showMainApp = true; // After waiting, switch to the main app
      });
    });
  }

  // context - how a widget knows where it is in the app and what it has access to

  @override
  Widget build(BuildContext context) { // Decides what to show on the screen
    if (_showMainApp) { // If true, show the main app
      return MyApp(); // Open the main app
    } 
    else { // If still false, keep showing the splash
      return MaterialApp( // Basic app setup
        debugShowCheckedModeBanner: false, // Hides the "debug" label
        home: Scaffold( // The main layout for this screen
          backgroundColor: const Color(0xFF1976d2), // Set background color to blue
          body: Center( // Put things in the middle of the screen
            child: Image.asset(
              'assets/images/ChatBookAILogoWhite.png',
              width: 170,
            ),
          ),
        ),
      );
    }
  }
}


// the overall app container
class MyApp extends StatefulWidget { // Main app widget that can change while running
  @override
  _MyAppState createState() => _MyAppState(); // Creates the app's state
}

class _MyAppState extends State<MyApp> { // Holds data and behavior for MyApp
  @override
  Widget build(BuildContext context) { // Builds the main structure of the app
    return MaterialApp( // Main app container
      debugShowCheckedModeBanner: false,
      title: 'ChatBook AI',
      theme: ThemeData( // App theme settings
        fontFamily: 'Poppins',
        primaryColor: const Color(0xFF1976d2), // Main blue color
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1976d2)),  // Creates color shades from blue
      ),
      // UPDATED: Directly load ChatPage which now has built-in sidebar navigation
      home: ChatPage(), // Opens ChatPage when app starts
    );
  }
}