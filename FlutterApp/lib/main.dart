import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:window_size/window_size.dart';

// Import pages
import 'chat.dart';
import 'faq.dart';
import 'feedback.dart';
import 'about.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  if (Platform.isWindows || Platform.isLinux || Platform.isMacOS) {
    setWindowTitle('ChatBook AI');
    setWindowMinSize(const Size(500, 1000));
    setWindowMaxSize(const Size(500, 1000));
    setWindowFrame(const Rect.fromLTWH(50, 50, 500, 1000));
  }

  runApp(MyApp());
}

// Auto-detect platform and return base server URL
String getBaseUrl() {
  if (Platform.isWindows || Platform.isLinux || Platform.isMacOS) {
    return "http://127.0.0.1:8000"; // Desktop
  } else if (Platform.isAndroid || Platform.isIOS) {
    return "http://192.168.1.100:8000"; // Mobile: replace with your PC LAN IP
  } else {
    throw UnsupportedError("Unsupported platform");
  }
}

// Generate unique session ID
String _sessionId = DateTime.now().millisecondsSinceEpoch.toString();

// Query API
Future<String> queryHandbook(String question) async {
  final url = Uri.parse("${getBaseUrl()}/chat");
  try {
    final response = await http.post(
      url,
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"query": question, "session_id": _sessionId}),
    );

    if (response.statusCode == 200) {
      final responseData = jsonDecode(response.body);
      return responseData["answer"] ?? "No response received";
    } else {
      return "Server error: ${response.statusCode}";
    }
  } catch (e) {
    return "⚠️ Cannot connect to server. Make sure FastAPI is running.";
  }
}

// Fetch reports API
Future<Map<String, dynamic>> fetchReports() async {
  final url = Uri.parse("${getBaseUrl()}/report");
  try {
    final response = await http.get(url);
    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      return {
        'total_queries': data['total_queries'] ?? 0,
        'answered_queries': data['answered_queries'] ?? 0,
        'not_found_queries': data['not_found_queries'] ?? 0,
        'failed_queries': data['failed_queries'] ?? 0,
        'accuracy_rate': data['accuracy_rate'] ?? 0.0,
        'most_frequent_questions': data['most_frequent_questions'] ?? [],
      };
    }
  } catch (_) {}
  return {
    'total_queries': 0,
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
  final url = Uri.parse("${getBaseUrl()}/feedback");
  try {
    final response = await http.post(
      url,
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({
        "feedback_text": feedbackText,
        "rating": rating,
        "user_type": userType,
        "session_id": sessionId  // Add this
      }),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return {
        "success": true,
        "message": data["message"] ?? "Feedback submitted successfully"
      };
    } else {
      return {"success": false, "message": "Server error: ${response.statusCode}"};
    }
  } catch (e) {
    return {"success": false, "message": "Cannot connect to server"};
  }
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ChatBook AI',
      theme: ThemeData(
        primaryColor: Color(0xFF1976d2),
        colorScheme: ColorScheme.fromSeed(seedColor: Color(0xFF1976d2)),
      ),
      home: MainScreen(),
    );
  }
}

class MainScreen extends StatefulWidget {
  @override
  _MainScreenState createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  final GlobalKey<ChatPageState> _chatPageKey = GlobalKey<ChatPageState>();

  void _onFaqQuestionTap(String question) {
    if (_chatPageKey.currentState != null) {
      _chatPageKey.currentState!.autoQuery(question);
    }
    Navigator.of(context).pop();
  }

  // ---------- FAQ ----------
  void _showFAQPage() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) {
        return _buildSheet(
          title: "Frequently Asked Questions",
          child: FaqPage(onQuestionTap: _onFaqQuestionTap),
        );
      },
    );
  }

  // ---------- ABOUT ----------
  void _showAboutPage() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) {
        return _buildSheet(
          title: "About ChatBook AI",
          child: AboutPage(),
        );
      },
    );
  }

  // ---------- FEEDBACK ----------
  void _showFeedbackPage() {
  showModalBottomSheet(
    context: context,
    isScrollControlled: true,
    backgroundColor: Colors.transparent,
    builder: (context) {
      return _buildSheet(
        title: "Send Feedback",
        child: FeedbackPage(sessionId: _sessionId),  // Pass session ID here
      );
    },
  );
}

  // ---------- REUSABLE SHEET BUILDER ----------
  Widget _buildSheet({required String title, required Widget child}) {
    return DraggableScrollableSheet(
      initialChildSize: 0.8,
      minChildSize: 0.5,
      maxChildSize: 0.95,
      builder: (context, scrollController) {
        return Container(
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.only(
              topLeft: Radius.circular(20),
              topRight: Radius.circular(20),
            ),
          ),
          child: Column(
            children: [
              Container(
                color: Color(0xFF1976d2),
                padding: EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(title,
                        style: TextStyle(
                            color: Colors.white,
                            fontSize: 18,
                            fontWeight: FontWeight.bold)),
                    IconButton(
                      icon: Icon(Icons.close, color: Colors.white),
                      onPressed: () => Navigator.of(context).pop(),
                    ),
                  ],
                ),
              ),
              Expanded(
                child: child,
              ),
            ],
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Column(
        children: [
          // Top AppBar
          Container(
            color: Color(0xFF1976d2),
            padding: EdgeInsets.only(
              top: MediaQuery.of(context).padding.top + 10,
              left: 16,
              right: 16,
              bottom: 10,
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text("ChatBook AI",
                    style: TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold)),
                PopupMenuButton<String>(
                  icon: Icon(Icons.more_vert, color: Colors.white),
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
                    PopupMenuItem(
                      value: 'faq',
                      child: ListTile(
                        leading: Icon(Icons.help_outline),
                        title: Text('FAQ'),
                        dense: true,
                      ),
                    ),
                    PopupMenuItem(
                      value: 'feedback',
                      child: ListTile(
                        leading: Icon(Icons.feedback_outlined),
                        title: Text('Feedback'),
                        dense: true,
                      ),
                    ),
                    PopupMenuItem(
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
          // Chat content
          Expanded(
            child: Container(
              color: Color(0xFFF4F6F9),
              child: ChatPage(key: _chatPageKey),
            ),
          ),
        ],
      ),
    );
  }
}