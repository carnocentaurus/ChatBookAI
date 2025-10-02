import 'package:flutter/material.dart';
import 'main.dart';

class FaqPage extends StatefulWidget {
  final Function(String)? onQuestionTap;
  
  const FaqPage({Key? key, this.onQuestionTap}) : super(key: key);

  @override
  State<FaqPage> createState() => _FaqPageState();
}

class _FaqPageState extends State<FaqPage> {
  Map<String, dynamic>? _reportData;
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _fetchReport();
  }

  Future<void> _fetchReport() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final data = await fetchReports();
      setState(() {
        _reportData = data;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = "⚠️ Cannot connect to server.";
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // MAIN CONTENT
        Expanded(
          child: _loading
              ? const Center(child: CircularProgressIndicator())
              : _error != null
                  ? Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Text(
                            _error!,
                            style: const TextStyle(color: Colors.red),
                          ),
                          const SizedBox(height: 16),
                          ElevatedButton(
                            onPressed: _fetchReport,
                            child: const Text("Retry"),
                          ),
                        ],
                      ),
                    )
                  : _reportData == null
                      ? const Center(child: Text("No FAQ data available."))
                      : ListView(
                          padding: const EdgeInsets.all(16),
                          children: [
                            const SizedBox(height: 8),
                            const SizedBox(height: 12),
                            ..._buildFaqList(),
                          ],
                        ),
        ),
      ],
    );
  }

  List<Widget> _buildFaqList() {
    final faqList = (_reportData?['most_frequent_questions'] as List<dynamic>?) ?? [];
    
    if (faqList.isEmpty) {
      return [
        Card(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
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

    // Build FAQ cards with rank numbers
    return faqList.asMap().entries.map((entry) {
      final int index = entry.key;
      final dynamic faqData = entry.value;
      final int rank = index + 1;
      final question = faqData['question']?.toString() ?? 'Unknown question';
      final count = faqData['count']?.toString() ?? '0';
      return _buildFaqCard(question, count, rank);
    }).toList();
  }

  Widget _buildFaqCard(String question, String count, int rank) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      elevation: 2,
      margin: const EdgeInsets.symmetric(vertical: 6),
      child: ListTile(
        leading: Container(
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            color: Colors.blueAccent.withOpacity(0.1),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.blueAccent, width: 2),
          ),
          child: Center(
            child: Text(
              rank.toString(),
              style: const TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Colors.blueAccent,
              ),
            ),
          ),
        ),
        title: Text(
          question.length > 80 ? "${question.substring(0, 80)}..." : question,
          style: const TextStyle(fontWeight: FontWeight.w500),
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              "Tap to ask chatbot",
              style: TextStyle(fontSize: 12, color: Colors.grey[600]),
            ),
          ],
        ),
        trailing: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
          decoration: BoxDecoration(
            color: Colors.blueAccent.withOpacity(0.15),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Text(
            count,
            style: const TextStyle(
                fontSize: 14, fontWeight: FontWeight.bold, color: Colors.blue),
          ),
        ),
        onTap: () {
          // Navigate to chat and auto-query the question
          if (widget.onQuestionTap != null) {
            widget.onQuestionTap!(question);
            
            // Show brief feedback
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text("Asking: ${question.length > 50 ? question.substring(0, 50) + '...' : question}"),
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