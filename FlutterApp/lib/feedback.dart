// feedback.dart

import 'package:flutter/material.dart';
import 'main.dart';

class FeedbackPage extends StatefulWidget {
  // This stores the chat session ID passed from another page.
  // If removed: You’ll get errors when trying to send feedback linked to a session.
  final String sessionId;

  const FeedbackPage({Key? key, required this.sessionId}) : super(key: key);

  @override
  State<FeedbackPage> createState() => _FeedbackPageState();
}

class _FeedbackPageState extends State<FeedbackPage> {
  // This controller manages the text inside the feedback box.
  // If removed: You can’t read or clear the feedback input.
  final TextEditingController _feedbackController = TextEditingController();

  // Default rating and user type
  int _rating = 5;
  String _userType = 'student';

  @override
  void dispose() {
    // Cleans up the text controller when leaving the page.
    // If removed: The app may slowly use more memory over time.
    _feedbackController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      // Allows scrolling if the keyboard covers part of the form.
      // If removed: The keyboard might block the input or submit button.
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).viewInsets.bottom, // Keeps space for keyboard
        left: 16,
        right: 16,
        top: 16,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ⭐ Rating Section
          Text("How would you rate your experience?"),
          SizedBox(height: 10),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: List.generate(5, (index) {
              return GestureDetector(
                onTap: () {
                  // Changes the star rating when tapped
                  setState(() {
                    _rating = index + 1;
                  });
                },
                child: Padding(
                  padding: EdgeInsets.symmetric(horizontal: 2),
                  child: Icon(
                    index < _rating ? Icons.star : Icons.star_border, // Filled or empty star
                    color: Colors.amber,
                    size: 30,
                  ),
                ),
              );
            }),
          ),
          SizedBox(height: 15),

          // 📝 Feedback Text Input
          Text("Your feedback:"),
          SizedBox(height: 8),
          TextField(
            controller: _feedbackController, // Connects the text field to the controller
            decoration: InputDecoration(
              hintText: "Tell us about your experience...",
              border: OutlineInputBorder(),
              contentPadding: EdgeInsets.all(12),
            ),
            maxLines: 3, // Allows multi-line feedback
          ),
          SizedBox(height: 15),

          // 👤 User Type Dropdown
          Text("You are a:"),
          SizedBox(height: 8),
          DropdownButton<String>(
            value: _userType,
            isExpanded: true,
            items: [
              DropdownMenuItem(value: 'student', child: Text('Student')),
              DropdownMenuItem(value: 'faculty', child: Text('Faculty')),
              DropdownMenuItem(value: 'staff', child: Text('Staff')),
              DropdownMenuItem(value: 'visitor', child: Text('Visitor')),
            ],
            onChanged: (val) => setState(() => _userType = val!), // Updates selected role
          ),
          SizedBox(height: 20),

          // 📤 Submit Button
          Align(
            alignment: Alignment.centerRight,
            child: ElevatedButton(
              child: Text("Submit"),
              onPressed: () async {
                // Prevents sending empty feedback
                if (_feedbackController.text.trim().isEmpty) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text("Please enter feedback")),
                  );
                  return;
                }

                // Closes the feedback page before sending data
                Navigator.of(context).pop();

                // Sends feedback to backend (session ID is used here)
                final result = await submitFeedback(
                  _feedbackController.text.trim(),
                  _rating,
                  _userType,
                  widget.sessionId,
                );

                // ✅ Shows success or error message at bottom
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text(result["message"]),
                    backgroundColor:
                        result["success"] ? Colors.green : Colors.red,
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}