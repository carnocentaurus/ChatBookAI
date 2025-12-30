// about.dart

import 'package:flutter/material.dart';
import 'chat.dart'; // REQUIRED: Import to access ResponsiveSize class

class AboutPage extends StatelessWidget {
  final ResponsiveSize responsive; // FIXED: Changed from dynamic to ResponsiveSize

  const AboutPage({Key? key, required this.responsive}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final res = responsive;

    return Container(
      color: Colors.white,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ---------- HEADER SECTION (MATCHES FAQ) ----------
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
                  "About ChatBook AI",
                  style: TextStyle(
                    fontSize: res.fontTitle(),
                    fontWeight: FontWeight.bold,
                    color: const Color(0xFF1976d2),
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  "Your intelligent guide to the GSU Student Handbook.",
                  style: TextStyle(
                    fontSize: res.fontSmall(),
                    color: Colors.black54,
                  ),
                ),
              ],
            ),
          ),

          // ---------- CONTENT SECTION ----------
          Expanded(
            child: SingleChildScrollView(
              padding: EdgeInsets.symmetric(horizontal: res.paddingMedium(24.0)), // FIXED: 24 -> 24.0
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(height: res.paddingSmall(16.0)), // FIXED: 16 -> 16.0
                  Text(
                    "ChatBook AI is designed to help Guimaras State University students navigate the complexities of campus life. Whether you have questions about admissions, academic policies, or student organizations, our AI-powered assistant provides instant answers directly from the official handbook.",
                    style: TextStyle(
                      fontSize: res.fontSmall(),
                      color: Colors.black87,
                      height: 1.5,
                    ),
                  ),
                  SizedBox(height: res.paddingLarge(40.0)), // FIXED: 40 -> 40.0
                  Center(
                    child: Opacity(
                      opacity: 0.5,
                      child: Image.asset(
                        'assets/images/ChatBookAILogoBlue.png',
                        width: 80,
                        height: 80,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}