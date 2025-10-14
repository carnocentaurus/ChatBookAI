// main.dart

import 'dart:io'; // Gives access to platform information (e.g., Windows, Android) and file I/O operations.
import 'dart:convert'; // Allows JSON encoding/decoding (used to convert data to/from the backend API).
import 'package:flutter/material.dart'; // Core Flutter UI library — provides widgets, layouts, and themes.
import 'package:http/http.dart' as http; // Enables HTTP requests to communicate with the FastAPI backend.
import 'package:window_size/window_size.dart'; // Lets you control the window’s size, title, and position (desktop only).

// Import pages
import 'chat.dart';
import 'faq.dart';
import 'feedback.dart';
import 'about.dart';

void main() { // The entry point of every Flutter app. Execution starts here.
  WidgetsFlutterBinding.ensureInitialized(); // Initializes Flutter engine bindings required for platform channel and native desktop method calls (like setWindowTitle).

  if (Platform.isWindows || Platform.isLinux || Platform.isMacOS) { // Checks if the app is running on a desktop platform (Windows, Linux, or macOS).  
    setWindowTitle('ChatBook AI'); // Only visible on desktop title bars.
    setWindowMinSize(const Size(500, 1000)); // Defines the smallest possible window size (width: 500px, height: 1000px).  
    setWindowMaxSize(const Size(500, 1000)); // Defines the largest possible window size (same as min size).
    setWindowFrame(const Rect.fromLTWH(50, 50, 500, 1000)); // Positions the window at (50, 50) with a size of 500x1000 pixels upon opening.
  }

  runApp(const SplashWrapper()); // Launches the root Flutter widget (`MyApp`) which builds the entire UI. 
}


String getBaseUrl() {  // This function determines which server address to use depending on the platform (desktop or mobile).
  if (Platform.isWindows || Platform.isLinux || Platform.isMacOS) {
    return "http://127.0.0.1:8000"; // Returns the localhost URL — typically used when developing and testing on desktop.
  } 
  else if (Platform.isAndroid || Platform.isIOS) {
    return "http://10.0.15.188:8000"; // Returns the LAN (Local Area Network) IP address of the PC where FastAPI is running.  
  } 
  else {
    throw UnsupportedError("Unsupported platform");
  }
}


// Creates a unique identifier for each app session by using the current timestamp in milliseconds.  
String _sessionId = DateTime.now().millisecondsSinceEpoch.toString();


Future<String> queryHandbook(String question) async {  // Defines an asynchronous function that takes the user's question as input
  final url = Uri.parse("${getBaseUrl()}/chat"); // Constructs the full backend API endpoint URL by appending '/chat' to the base URL. 
  try {
    // Sends an HTTP POST request to the backend asynchronously.
    final response = await http.post(
      url,
      headers: {"Content-Type": "application/json"},
      // The unique session ID ensures the backend maintains session context per user.
      body: jsonEncode({"query": question, "session_id": _sessionId}),
    );

    if (response.statusCode == 200) { // Checks if the HTTP status code indicates success (200 OK).
      final responseData = jsonDecode(response.body);  // Decodes the JSON response body from the backend into a Dart Map.
      return responseData["answer"] ?? "No response received"; // Returns the chatbot’s answer if it exists.  
    } 
    else { // Executes if the server responds but with an error (e.g., 404, 500).
      return "Server error: ${response.statusCode}";
    }
  } 
  catch (e) {
    return "⚠️ Cannot connect to server. Error: $e";
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
        'total_queries': data['total_queries'] ?? 0, // Total number of questions asked
        'answered_queries': data['answered_queries'] ?? 0, // Questions the chatbot answered
        'not_found_queries': data['not_found_queries'] ?? 0, // Questions not found in the handbook
        'failed_queries': data['failed_queries'] ?? 0, // Questions that failed to process
        'accuracy_rate': data['accuracy_rate'] ?? 0.0, // Chatbot’s accuracy rate
        'most_frequent_questions': data['most_frequent_questions'] ?? [], // Common questions asked
      };
    }
  } catch (_) {} // If something goes wrong, skip it
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
      headers: {"Content-Type": "application/json"}, // Tell the server we’re sending JSON data
      body: jsonEncode({
        "feedback_text": feedbackText, // The message written by the user
        "rating": rating, // The user’s rating (like 1–5 stars)
        "user_type": userType, // Type of user (ex. student, admin, etc.)
        "session_id": sessionId // Unique ID for this session
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
    return {"success": false, "message": "Cannot connect to server: $e"}; // Show error if the server can’t be reached
  }
}


class SplashWrapper extends StatefulWidget {
  const SplashWrapper({super.key});

  @override
  State<SplashWrapper> createState() => _SplashWrapperState();
}

class _SplashWrapperState extends State<SplashWrapper> {
  bool _showMainApp = false;

  @override
  void initState() {
    super.initState();
    // Show splash for 2 seconds before showing main app
    Future.delayed(const Duration(seconds: 4), () {
      setState(() {
        _showMainApp = true;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_showMainApp) {
      return MyApp(); // Load your main app after splash
    } else {
      return MaterialApp(
        debugShowCheckedModeBanner: false,
        home: Scaffold(
          backgroundColor: const Color(0xFF1976d2), // Blue background
          body: Center(
            child: Image.asset(
              'assets/images/ChatBookAILogoAppIcon.png',
              width: 180,
            ),
          ),
        ),
      );
    }
  }
}


class MyApp extends StatefulWidget { // Main app widget that can change while running
  @override
  _MyAppState createState() => _MyAppState(); // Creates the app’s state
}

class _MyAppState extends State<MyApp> { // Holds data and behavior for MyApp
  @override
  Widget build(BuildContext context) { // Builds the main structure of the app
    return MaterialApp( // Main app container
      title: 'ChatBook AI',
      theme: ThemeData( // App theme settings
        fontFamily: 'Poppins',
        primaryColor: const Color(0xFF1976d2), // Main blue color
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1976d2)),  // Creates color shades from blue
      ),
      home: MainScreen(), // Opens MainScreen when app starts
    );
  }
}

class MainScreen extends StatefulWidget { // The main screen of the app
  @override 
  _MainScreenState createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> { // Holds data and actions for MainScreen
  final GlobalKey<ChatPageState> _chatPageKey = GlobalKey<ChatPageState>(); // Key to access the chat page’s state

  void _onFaqQuestionTap(String question) { // Runs when an FAQ question is tapped
    if (_chatPageKey.currentState != null) { // Checks if chat page is ready
      _chatPageKey.currentState!.autoQuery(question); // Sends the question to the chat automatically
    }
    Navigator.of(context).pop(); // Closes the FAQ or side menu
  }


   // ---------- FAQ ----------  
  void _showFAQPage() { // Opens the "Frequently Asked Questions" page  
    showModalBottomSheet( // Shows a slide-up window from the bottom of the screen  
      context: context, // Uses the current page’s context to display it  
      isScrollControlled: true, // Allows the sheet to take up more space if needed  
      backgroundColor: Colors.transparent, // Makes the background see-through  
      builder: (context) { // Builds what the bottom sheet will show  
        return _buildSheet( // Uses a helper function to build the sheet layout  
          title: "FAQs", // Title shown at the top of the sheet  
          child: FaqPage(onQuestionTap: _onFaqQuestionTap), // Shows the FAQ page and handles question clicks  
        );  
      },  
    );  
  }  

  // ---------- ABOUT ----------  
  void _showAboutPage() { // Opens the "About" page  
    showModalBottomSheet( // Shows a slide-up window from the bottom of the screen  
      context: context, // Uses the current page’s context to display it  
      isScrollControlled: true, // Allows the sheet to take up more space if needed  
      backgroundColor: Colors.transparent, // Makes the background see-through  
      builder: (context) { // Builds what the bottom sheet will show  
        return _buildSheet( // Uses a helper function to build the sheet  
          title: "About ChatBook AI", // Title shown at the top of the sheet  
          child: AboutPage(), // Shows the AboutPage content inside the sheet  
        );  
      },  
    );  
  }  

  // ---------- FEEDBACK ----------  
  void _showFeedbackPage() { // Opens the "Feedback" page  
    showModalBottomSheet( // Shows a slide-up window from the bottom of the screen  
      context: context, // Uses the current page’s context to display it  
      isScrollControlled: true, // Allows the sheet to take up more space if needed  
      backgroundColor: Colors.transparent, // Makes the background see-through  
      builder: (context) { // Builds what the bottom sheet will show  
        return _buildSheet( // Uses a helper function to build the sheet  
          title: "Send Feedback", // Title shown at the top of the sheet  
          child: FeedbackPage(sessionId: _sessionId), // Shows the feedback form with the current session ID  
        );  
      },  
    );  
  }  


    // ---------- REUSABLE SHEET BUILDER ----------  
  Widget _buildSheet({required String title, required Widget child}) { // Makes a slide-up window that can be reused  
    return DraggableScrollableSheet( // A bottom sheet that can be dragged up or down  
      initialChildSize: 0.8, // Starts at 80% of the screen height  
      minChildSize: 0.5, // Can shrink to 50% of the screen height  
      maxChildSize: 0.95, // Can stretch up to 95% of the screen height  
      builder: (context, scrollController) { // Builds what will appear inside the sheet  
        return Container( // The main box holding everything inside  
          decoration: const BoxDecoration( // Adds style to the container  
            color: Colors.white, // Sets the background color to white  
            borderRadius: BorderRadius.only( // Rounds the top corners  
              topLeft: Radius.circular(20),  
              topRight: Radius.circular(20),  
            ),  
          ),  
          child: Column( // Arranges things from top to bottom  
            children: [  
              Container( // The top header area of the sheet  
                color: const Color(0xFF1976d2), // Sets the header color (blue)  
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14), // Adds space inside the header  
                child: Row( // Puts the title and close button side by side  
                  mainAxisAlignment: MainAxisAlignment.spaceBetween, // Spreads them apart  
                  children: [  
                    Text(title, // Shows the title text  
                        style: const TextStyle(  
                            color: Colors.white, // White text color  
                            fontSize: 18, // Text size  
                            fontWeight: FontWeight.bold)), // Makes the title bold  
                    IconButton( // A button with an icon  
                      icon: const Icon(Icons.close, color: Colors.white), // “X” close icon in white  
                      onPressed: () => Navigator.of(context).pop(), // Closes the sheet when pressed  
                    ),  
                  ],  
                ),  
              ),  
              Expanded(child: child), // Fills the rest of the space with the given content  
            ],  
          ),  
        );  
      },  
    );  
  }  


@override
Widget build(BuildContext context) {
  return Scaffold(
    resizeToAvoidBottomInset: true, // Moves up chat box when keyboard appears
    backgroundColor: Colors.white, // Background color
    body: Column(
      children: [
        // 🔹 Top Header Bar
        Container(
          color: const Color(0xFF1976d2), // Blue header color
          padding: EdgeInsets.only(
            top: MediaQuery.of(context).padding.top + 10,
            left: 12,
            right: 8,
            bottom: 10,
          ),
          child: Row(
            children: [
              // Left side: Icon + Title
              Expanded(
                child: Row(
                  children: [
                    // App logo (small icon)
                    Image.asset(
                      'assets/images/ChatBookAILogoAppIcon.png',
                      height: 28,
                      width: 28,
                      fit: BoxFit.contain,
                    ),
                    const SizedBox(width: 10),

                    // App title text
                    Flexible(
                      child: Text(
                        "ChatBook AI",
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          fontFamily: 'Poppins', // Uses your Poppins font
                        ),
                        overflow: TextOverflow.ellipsis, // Prevents overflow
                        maxLines: 1,
                        softWrap: false,
                      ),
                    ),
                  ],
                ),
              ),

              // Right side: Menu button
              PopupMenuButton<String>(
                icon: const Icon(Icons.more_vert, color: Colors.white),
                onSelected: (String result) {
                  if (result == 'faq') {
                    _showFAQPage();
                  } else if (result == 'feedback') {
                    _showFeedbackPage();
                  } else if (result == 'about') {
                    _showAboutPage();
                  }
                },
                itemBuilder: (context) => [
                  const PopupMenuItem(
                    value: 'faq',
                    child: ListTile(
                      leading: Icon(Icons.help_outline),
                      title: Text('FAQ'),
                      dense: true,
                    ),
                  ),
                  const PopupMenuItem(
                    value: 'feedback',
                    child: ListTile(
                      leading: Icon(Icons.feedback_outlined),
                      title: Text('Feedback'),
                      dense: true,
                    ),
                  ),
                  const PopupMenuItem(
                    value: 'about',
                    child: ListTile(
                      leading: Icon(Icons.info_outline),
                      title: Text('About'),
                      dense: true,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),

        // 🔹 Chat Body Area
        Expanded(
          child: Container(
            color: const Color(0xFFF4F6F9), // Light gray background
            child: ChatPage(key: _chatPageKey), // Main chat screen
          ),
        ),
      ],
    ),
  );
}}