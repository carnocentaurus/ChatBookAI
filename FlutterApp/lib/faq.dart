// faq.dart — shows frequently asked questions from the chatbot data.

import 'package:flutter/material.dart'; // gives access to UI widgets (buttons, text, etc.)
import 'main.dart'; // lets this file use functions from main.dart (like fetchReports)


class FaqPage extends StatefulWidget { // a screen that can update itself
  final Function(String)? onQuestionTap; // lets main.dart know if a question is tapped

  // calls the parent class’s constructor
  const FaqPage({Key? key, this.onQuestionTap}) : super(key: key);

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
  // build() - this is where you describe what the page looks like
  Widget build(BuildContext context) {
    return Column( // lays everything vertically
      children: [
        // ---------- MAIN CONTENT ----------
        Expanded( // fills the available space
          child: _loading
              ? const Center(child: CircularProgressIndicator()) // shows spinner while loading
              : _error != null // if there’s an error message
                  ? Center( // shows error and retry button
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Text(
                            _error!,
                            style: const TextStyle(color: Colors.red),
                          ),
                          const SizedBox(height: 16),
                          ElevatedButton(
                            onPressed: _fetchReport, // reload data
                            child: const Text("Retry"),
                          ),
                        ],
                      ),
                    )
                  : _reportData == null
                      ? const Center(
                          child: Text("No FAQ data available.")) // if no data
                      : ListView( // creates a scrollable column (you can scroll through questions)
                          padding: const EdgeInsets.all(16),
                          children: [
                            const SizedBox(height: 8),
                            const SizedBox(height: 12),
                            ..._buildFaqList(), // builds cards for each FAQ
                          ],
                        ),
        ),
      ],
    );
  }

  // ---------- BUILDS LIST OF FAQ CARDS ----------
  List<Widget> _buildFaqList() {
    final faqList =
        (_reportData?['most_frequent_questions'] as List<dynamic>?) ?? [];
    // gets list of top questions; empty if none found

    if (faqList.isEmpty) {
      return [
        Card(
          shape:
              RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          child: const Padding(
            padding: EdgeInsets.all(16.0),
            child: Text(
              "No frequently asked questions yet.\nStart chatting to see popular questions here!",
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.grey,
                fontStyle: FontStyle.italic,
              ),
            ),
          ),
        ),
      ];
    }

    // builds each FAQ card with a number (rank)
    return faqList.asMap().entries.map((entry) { // turns the list into a map with key-value pairs
      final int index = entry.key; // card number
      final dynamic faqData = entry.value; // each question
      final int rank = index + 1; // makes numbering start at 1 (instead of 0)
      final question = faqData['question']?.toString() ?? 'Unknown question';
      final count = faqData['count']?.toString() ?? '0';
      return _buildFaqCard(question, count, rank); // make a card for each
    }).toList();
  }

  // ---------- BUILDS EACH FAQ CARD ----------
  Widget _buildFaqCard(String question, String count, int rank) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      elevation: 2, // small shadow
      margin: const EdgeInsets.symmetric(vertical: 6), // space between cards
      child: ListTile( // Convenient layout for showing icons or widgets on the left/right with text in the middle
        leading: Container( // round number circle on the left
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            color: Colors.blueAccent.withOpacity(0.1),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.blueAccent, width: 2),
          ),
          child: Center(
            child: Text(
              rank.toString(), // shows 1, 2, 3 to indicate rank
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Colors.blueAccent,
              ),
            ),
          ),
        ),
        title: Text(
          question.length > 80
              ? "${question.substring(0, 80)}..." // If the text is longer than 80 characters, it trims it and adds “...”
              : question,
          style: const TextStyle(fontWeight: FontWeight.w500), // medium bold
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start, // aligns the children to the left
          children: [
            Text(
              "Tap to ask chatbot", // hint for the user
              style: TextStyle(fontSize: 12, color: Colors.grey[600]),
            ),
          ],
        ),
        trailing: Container( // shows count bubble on the right
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4), // Defines the space inside or around widgets
          decoration: BoxDecoration( // Styles a box
            color: Colors.blueAccent.withOpacity(0.15),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Text(
            count, // number of times question was asked
            style: const TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.bold,
                color: Colors.blue),
          ),
        ),
        onTap: () {
          // when the user taps a question card
          if (widget.onQuestionTap != null) {
            widget.onQuestionTap!(question); // sends the question to chat.dart

            // small notification popup
            ScaffoldMessenger.of(context).showSnackBar( // Shows brief messages (SnackBar) over the app screen
              SnackBar( // A small popup that appears temporarily to give feedback (e.g., “Message sent”, “Item deleted”)
                content: Text(
                  "Asking: ${question.length > 50 ? question.substring(0, 50) + '...' : question}",
                ),
                duration: Duration(seconds: 2),
                backgroundColor: Colors.green,
              ),
            );
          }
        },
      ),
    );
  }
}