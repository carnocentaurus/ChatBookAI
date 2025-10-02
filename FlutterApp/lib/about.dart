import 'package:flutter/material.dart';

class AboutPage extends StatelessWidget {
  const AboutPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Text(
        "ChatBook AI is your GSU student handbook assistant. "
        "Ask questions about admissions, campus life, academics, and more.",
        style: TextStyle(fontSize: 15, height: 1.4),
      ),
    );
  }
}