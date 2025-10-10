// about.dart

import 'package:flutter/material.dart';

// This page shows a short description about ChatBook AI.
// It’s a simple static info page (no interactivity or data fetching).
class AboutPage extends StatelessWidget {
  const AboutPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Padding(
      // Adds spacing around the text so it doesn’t touch the screen edges.
      // If removed: the text would look cramped against the borders.
      padding: const EdgeInsets.all(16),

      // Displays the text content for the About section.
      child: Text(
        "ChatBook AI is your GSU student handbook assistant. "
        "Ask questions about admissions, campus life, academics, and more.",

        // Text styling: adjusts size and spacing between lines.
        // If removed: text becomes smaller and tightly packed.
        style: TextStyle(fontSize: 15, height: 1.4),
      ),
    );
  }
}