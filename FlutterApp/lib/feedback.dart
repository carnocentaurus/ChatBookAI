import 'package:flutter/material.dart';
import 'main.dart';

class FeedbackPage extends StatefulWidget {
  final String sessionId;  // Add this parameter
  
  const FeedbackPage({Key? key, required this.sessionId}) : super(key: key);

  @override
  State<FeedbackPage> createState() => _FeedbackPageState();
}

class _FeedbackPageState extends State<FeedbackPage> {
  final TextEditingController _feedbackController = TextEditingController();
  int _rating = 5;
  String _userType = 'student';

  @override
  void dispose() {
    _feedbackController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: EdgeInsets.only(
        bottom: MediaQuery.of(context).viewInsets.bottom,
        left: 16,
        right: 16,
        top: 16,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text("How would you rate your experience?"),
          SizedBox(height: 10),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: List.generate(5, (index) {
              return GestureDetector(
                onTap: () {
                  setState(() {
                    _rating = index + 1;
                  });
                },
                child: Padding(
                  padding: EdgeInsets.symmetric(horizontal: 2),
                  child: Icon(
                    index < _rating ? Icons.star : Icons.star_border,
                    color: Colors.amber,
                    size: 30,
                  ),
                ),
              );
            }),
          ),
          SizedBox(height: 15),
          Text("Your feedback:"),
          SizedBox(height: 8),
          TextField(
            controller: _feedbackController,
            decoration: InputDecoration(
              hintText: "Tell us about your experience...",
              border: OutlineInputBorder(),
              contentPadding: EdgeInsets.all(12),
            ),
            maxLines: 3,
          ),
          SizedBox(height: 15),
          Text("You are a:"),
          SizedBox(height: 8),
          DropdownButton<String>(
            value: _userType,
            isExpanded: true,
            items: [
              DropdownMenuItem(
                  value: 'student', child: Text('Student')),
              DropdownMenuItem(
                  value: 'faculty', child: Text('Faculty')),
              DropdownMenuItem(value: 'staff', child: Text('Staff')),
              DropdownMenuItem(
                  value: 'visitor', child: Text('Visitor')),
            ],
            onChanged: (val) => setState(() => _userType = val!),
          ),
          SizedBox(height: 20),
          Align(
            alignment: Alignment.centerRight,
            child: ElevatedButton(
              child: Text("Submit"),
              onPressed: () async {
                if (_feedbackController.text.trim().isEmpty) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(content: Text("Please enter feedback")),
                  );
                  return;
                }

                Navigator.of(context).pop();

                final result = await submitFeedback(
                  _feedbackController.text.trim(),
                  _rating,
                  _userType,
                  widget.sessionId,  // Use the passed session ID
                );

                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text(result["message"]),
                    backgroundColor: result["success"]
                        ? Colors.green
                        : Colors.red,
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