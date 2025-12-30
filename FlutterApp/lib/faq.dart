import 'package:flutter/material.dart'; // gives access to UI widgets (buttons, text, etc.)
import 'main.dart'; // lets this file use functions from main.dart (like fetchReports)
import 'chat.dart'; // REQUIRED: Import this to access the ResponsiveSize class

class FaqPage extends StatefulWidget { // a screen that can update itself
  final Function(String)? onQuestionTap; // lets main.dart know if a question is tapped
  final ResponsiveSize responsive; // FIXED: Changed from dynamic to ResponsiveSize

  // calls the parent class’s constructor
  const FaqPage({Key? key, this.onQuestionTap, required this.responsive}) : super(key: key);

  @override
  State<FaqPage> createState() => _FaqPageState(); // connects logic part
}

class _FaqPageState extends State<FaqPage> { // this handles the logic of FAQ page
  Map<String, dynamic>? _reportData; // stores data fetched from the server
  bool _loading = true; // shows loading spinner
  String? _error; // shows error message if connection fails

  @override
  void initState() { // runs automatically when the page opens
    super.initState(); // Runs Flutter’s built-in setup code from the parent State class
    _fetchReport(); // gets FAQ data from the backend
  }

  // fetch data from backend
  Future<void> _fetchReport() async {
    setState(() { // Tells Flutter that something in the UI changed, so it should rebuild
      _loading = true; // show loading spinner
      _error = null; // reset any old errors
    });

    try {
      final data = await fetchReports(); // asks the backend for FAQ data
      setState(() {
        _reportData = data; // store data for display
        _loading = false; // stop showing spinner
      });
    } 
    catch (e) {
      setState(() {
        _error = "⚠️ Cannot connect to server."; // show error message
        _loading = false; // stop spinner
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    // We use widget.responsive to keep sizes consistent across the app
    final res = widget.responsive;

    return Container(
      color: Colors.white,
      child: Column( // lays everything vertically
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ---------- HEADER SECTION ----------
          Padding(
            padding: EdgeInsets.fromLTRB(
              res.paddingMedium(24.0), // FIXED: 24 -> 24.0
              res.paddingMedium(24.0), // FIXED: 24 -> 24.0
              res.paddingMedium(24.0), // FIXED: 24 -> 24.0
              res.paddingSmall(8.0)    // FIXED: 8 -> 8.0
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "Frequently Asked Questions",
                  style: TextStyle(
                    fontSize: res.fontTitle(),
                    fontWeight: FontWeight.bold,
                    color: const Color(0xFF1976d2),
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  "Common topics students are asking about.",
                  style: TextStyle(
                    fontSize: res.fontSmall(),
                    color: Colors.black54,
                  ),
                ),
              ],
            ),
          ),

          // ---------- MAIN CONTENT ----------
          Expanded( 
            child: _loading
                ? const Center(child: CircularProgressIndicator(color: Color(0xFF1976d2))) 
                : _error != null 
                    ? Center( 
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.cloud_off, size: 64, color: Colors.grey.shade300),
                            const SizedBox(height: 16),
                            Text(
                              _error!,
                              style: const TextStyle(color: Colors.redAccent),
                            ),
                            const SizedBox(height: 16),
                            ElevatedButton.icon(
                              onPressed: _fetchReport, 
                              icon: const Icon(Icons.refresh),
                              label: const Text("Retry"),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: const Color(0xFF1976d2),
                                foregroundColor: Colors.white,
                              ),
                            ),
                          ],
                        ),
                      )
                    : _reportData == null || (_reportData?['most_frequent_questions'] as List).isEmpty
                        ? Center(
                            child: Text(
                              "No FAQ data available yet.",
                              style: TextStyle(color: Colors.grey.shade400),
                            ),
                          )
                        : ListView.builder( 
                            padding: EdgeInsets.all(res.paddingSmall(16.0)), // FIXED: 16 -> 16.0
                            itemCount: (_reportData?['most_frequent_questions'] as List).length,
                            itemBuilder: (context, index) {
                              final faqList = _reportData?['most_frequent_questions'] as List;
                              final faqData = faqList[index];
                              final question = faqData['question']?.toString() ?? 'Unknown question';
                              final count = faqData['count']?.toString() ?? '0';
                              return _buildFaqCard(question, count, index + 1, res);
                            },
                          ),
          ),
        ],
      ),
    );
  }

  // ---------- BUILDS EACH FAQ CARD ----------
  Widget _buildFaqCard(String question, String count, int rank, ResponsiveSize res) { // FIXED: dynamic -> ResponsiveSize
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.grey.shade100),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.03),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: ListTile( 
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        leading: Container( 
          width: 36,
          height: 36,
          decoration: BoxDecoration(
            color: const Color(0xFF1976d2).withOpacity(0.1),
            shape: BoxShape.circle,
          ),
          child: Center(
            child: Text(
              rank.toString(),
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                color: Color(0xFF1976d2),
              ),
            ),
          ),
        ),
        title: Text(
          question,
          maxLines: 2,
          overflow: TextOverflow.ellipsis,
          style: TextStyle(
            fontSize: res.fontSmall(),
            fontWeight: FontWeight.w600,
            color: Colors.black87,
          ),
        ),
        subtitle: Padding(
          padding: const EdgeInsets.only(top: 4.0),
          child: Row(
            children: [
              Icon(Icons.chat_bubble_outline, size: 12, color: Colors.grey.shade500),
              const SizedBox(width: 4),
              Text(
                "Tap to ask chatbot", 
                style: TextStyle(fontSize: 11, color: Colors.grey.shade500),
              ),
            ],
          ),
        ),
        trailing: Container( 
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration( 
            color: Colors.grey.shade100,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                count, 
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  color: Colors.black54,
                ),
              ),
              const Text(
                "Asks",
                style: TextStyle(fontSize: 8, color: Colors.black38),
              ),
            ],
          ),
        ),
        onTap: () {
          if (widget.onQuestionTap != null) {
            widget.onQuestionTap!(question); 
            
            ScaffoldMessenger.of(context).showSnackBar( 
              SnackBar( 
                content: Text("Asking: $question"),
                behavior: SnackBarBehavior.floating,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                duration: const Duration(seconds: 1),
                backgroundColor: const Color(0xFF1976d2),
              ),
            );
          }
        },
      ),
    );
  }
}