// feedback.dart

import 'package:flutter/material.dart';
import 'main.dart'; 
import 'chat.dart'; // REQUIRED: Import this to access the ResponsiveSize class

class FeedbackPage extends StatefulWidget {
  final String sessionId;
  final ResponsiveSize responsive; // FIXED: Changed from dynamic to ResponsiveSize

  const FeedbackPage({
    Key? key, 
    required this.sessionId, 
    required this.responsive
  }) : super(key: key);

  @override
  _FeedbackPageState createState() => _FeedbackPageState();
}

class _FeedbackPageState extends State<FeedbackPage> {
  int _selectedRating = 5; 
  String _userType = 'student';
  final TextEditingController _feedbackController = TextEditingController();

  @override
  void dispose() {
    _feedbackController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // We use widget.responsive to keep sizes consistent across the app
    final res = widget.responsive;

    return GestureDetector(
      onTap: () => FocusScope.of(context).unfocus(),
      child: Container(
        color: Colors.white,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start, 
          children: [
            // ---------- HEADER SECTION ----------
            Padding(
              padding: EdgeInsets.fromLTRB(
                res.paddingMedium(24.0), // FIXED: 24 -> 24.0
                res.paddingSmall(16.0),  // FIXED: 16 -> 16.0
                res.paddingMedium(24.0), // FIXED: 24 -> 24.0
                res.paddingSmall(4.0)    // FIXED: 4 -> 4.0
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "Feedback",
                    style: TextStyle(
                      fontSize: res.fontTitle(),
                      fontWeight: FontWeight.bold,
                      color: const Color(0xFF1976d2),
                    ),
                  ),
                  const SizedBox(height: 2), 
                  Text(
                    "Tell us how we can improve the handbook experience.",
                    style: TextStyle(
                      fontSize: res.fontSmall(),
                      color: Colors.black54,
                    ),
                  ),
                ],
              ),
            ),

            // ---------- SCROLLABLE FORM CONTENT ----------
            Expanded(
              child: SingleChildScrollView(
                padding: EdgeInsets.only(
                  left: res.paddingMedium(24.0),  // FIXED: 24 -> 24.0
                  right: res.paddingMedium(24.0), // FIXED: 24 -> 24.0
                  top: 0,
                  bottom: MediaQuery.of(context).viewInsets.bottom + 16,
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Center(
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: List.generate(5, (index) {
                          return IconButton(
                            padding: const EdgeInsets.symmetric(horizontal: 4),
                            constraints: const BoxConstraints(),
                            icon: Icon(
                              index < _selectedRating ? Icons.star : Icons.star_border,
                              color: Colors.amber,
                              size: 32, 
                            ),
                            onPressed: () {
                              setState(() {
                                _selectedRating = index + 1;
                              });
                            },
                          );
                        }),
                      ),
                    ),
                    const SizedBox(height: 12),

                    Text(
                      "You are a:",
                      style: TextStyle(
                        fontSize: res.fontSmall(),
                        fontWeight: FontWeight.bold,
                        color: Colors.black87,
                      ),
                    ),
                    const SizedBox(height: 4),
                    // Dropdown for user type
                    SizedBox(
                      height: 45, 
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12),
                        decoration: BoxDecoration(
                          color: Colors.grey.shade50,
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: Colors.grey.shade300),
                        ),
                        child: DropdownButtonHideUnderline(
                          child: DropdownButton<String>(
                            value: _userType,
                            isExpanded: true,
                            style: const TextStyle(fontSize: 14, color: Colors.black),
                            items: const [
                              DropdownMenuItem(value: 'student', child: Text('Student')),
                              DropdownMenuItem(value: 'faculty', child: Text('Faculty')),
                              DropdownMenuItem(value: 'staff', child: Text('Staff')),
                              DropdownMenuItem(value: 'visitor', child: Text('Visitor')),
                            ],
                            onChanged: (val) => setState(() => _userType = val!),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),

                    Text(
                      "Comments:",
                      style: TextStyle(
                        fontSize: res.fontSmall(),
                        fontWeight: FontWeight.bold,
                        color: Colors.black87,
                      ),
                    ),
                    const SizedBox(height: 4),
                    TextField(
                      controller: _feedbackController,
                      maxLines: 12, // FIX: Reduced from 12 to stop it from pushing off screen
                      style: const TextStyle(fontSize: 14),
                      decoration: InputDecoration(
                        hintText: "What could be better?",
                        isDense: true, 
                        filled: true,
                        fillColor: Colors.grey.shade50,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(8),
                          borderSide: BorderSide(color: Colors.grey.shade300),
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),

                    // Submit button
                    SizedBox(
                      width: double.infinity,
                      height: 44, 
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF1976d2),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8),
                          ),
                          elevation: 0,
                        ),
                        onPressed: () async {
                          if (_feedbackController.text.trim().isEmpty) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(content: Text("Please enter feedback")),
                            );
                            return;
                          }

                          final result = await submitFeedback(
                            _feedbackController.text.trim(),
                            _selectedRating,
                            _userType,
                            widget.sessionId,
                          );

                          if (mounted) {
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(
                                content: Text(result["message"]),
                                behavior: SnackBarBehavior.floating,
                                backgroundColor: result["success"] ? Colors.green : Colors.red,
                              ),
                            );

                            if (result["success"]) {
                              FocusScope.of(context).unfocus();
                              // FIX: Use a more robust way to close the bottom sheet
                              if (Navigator.canPop(context)) {
                                Navigator.pop(context);
                              }
                            }
                          }
                        },
                        child: const Text(
                          "Submit Feedback",
                          style: TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.bold),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}