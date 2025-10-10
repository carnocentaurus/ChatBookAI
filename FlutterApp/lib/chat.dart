// chat.dart

import 'package:flutter/material.dart'; // For building the app UI
import 'main.dart'; // Imports functions like queryHandbook from main.dart

class ChatPage extends StatefulWidget { // Chat screen that can change while running
  const ChatPage({Key? key}) : super(key: key); // Constructor

  @override
  ChatPageState createState() => ChatPageState(); // Creates the state for this page
}

class ChatPageState extends State<ChatPage> { // Main chat logic and layout
  final List<_Message> _messages = []; // Stores all chat messages
  final TextEditingController _controller = TextEditingController(); // Controls the text box
  final ScrollController _scrollController = ScrollController(); // Controls chat scrolling
  bool _isTyping = false; // Shows when the bot is typing

  @override
  void dispose() { // Runs when leaving the screen
    _scrollController.dispose(); // Frees scroll memory
    super.dispose(); // Calls parent cleanup
  }

  void _scrollToBottom() { // Moves chat to the latest message
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) { // Checks if chat can scroll
        _scrollController.animateTo( // Smoothly scrolls down
          _scrollController.position.maxScrollExtent, // Go to bottom
          duration: const Duration(milliseconds: 300), // Takes 0.3 seconds
          curve: Curves.easeOut, // Smooth ending motion
        );
      }
    });
  }

  // Auto-query method that can be called from main.dart
  void autoQuery(String question) { // Allows FAQ questions to go straight to chat
    _controller.text = question; // Puts question in input box
    _sendMessage(question); // Sends the question automatically
  }

  Future<void> _sendMessage(String text) async { // Sends user message and gets bot reply
    if (text.trim().isEmpty) return; // Does nothing if input is empty

    setState(() {
      _messages.add(_Message(text: text, isUser: true)); // Adds user message to chat
      _isTyping = true; // Shows "Bot is typing..."
    });

    _scrollToBottom(); // Scrolls chat to bottom
    _controller.clear(); // Clears input box

    try {
      final answer = await queryHandbook(text); // Sends text to backend and waits for reply

      setState(() {
        _messages.add(_Message(text: answer, isUser: false)); // Adds bot reply to chat
        _isTyping = false; // Stops typing message
      });

      _scrollToBottom(); // Scroll again to latest reply
    } catch (e) {
      setState(() {
        _messages.add( // Shows error if backend fails
          _Message(text: "⚠️ Cannot connect to server.", isUser: false),
        );
        _isTyping = false; // Stops typing message
      });

      _scrollToBottom(); // Scroll again to bottom
    }
  }

  @override
  Widget build(BuildContext context) { // Builds the whole chat layout
    return Column( // Puts everything in vertical order
      children: [
        Expanded( // Fills available space
          child: Column(
            children: [
              Expanded(
                child: ListView.builder( // Shows all messages
                  controller: _scrollController, // Makes it scrollable
                  padding: const EdgeInsets.all(12), // Adds space around messages
                  itemCount: _messages.length, // Number of chat messages
                  itemBuilder: (context, index) { // Builds each message bubble
                    final message = _messages[index]; // Gets the message
                    return Align( // Aligns message left or right
                      alignment: message.isUser
                          ? Alignment.centerRight // User messages on right
                          : Alignment.centerLeft, // Bot messages on left
                      child: Container( // Message bubble box
                        margin: const EdgeInsets.symmetric(vertical: 4), // Space between bubbles
                        padding: const EdgeInsets.symmetric(
                          vertical: 12, // Space above/below text
                          horizontal: 16, // Space left/right of text
                        ),
                        decoration: BoxDecoration( // Styles the bubble
                          color: message.isUser
                              ? Colors.blue.shade700 // Blue for user
                              : Colors.green.shade200, // Green for bot
                          borderRadius: BorderRadius.circular(20), // Rounded corners
                          boxShadow: [ // Adds soft shadow
                            BoxShadow(
                              color: Colors.black.withOpacity(0.05),
                              blurRadius: 4, // Shadow softness
                              offset: const Offset(2, 2), // Shadow position
                            ),
                          ],
                        ),
                        child: SelectableText( // Lets user copy text
                          message.text, // Shows the actual message
                          style: TextStyle(
                            fontSize: 15, // Text size
                            color: message.isUser
                                ? Colors.white // White for user
                                : Colors.black87, // Black for bot
                          ),
                        ),
                      ),
                    );
                  },
                ),
              ),
              if (_isTyping) // Shows this only when bot is typing
                const Padding(
                  padding: EdgeInsets.all(8.0), // Adds space around text
                  child: Align(
                    alignment: Alignment.centerLeft, // Aligns to left
                    child: Text(
                      "Bot is typing...", // Typing message
                      style: TextStyle(fontStyle: FontStyle.italic), // Italic style
                    ),
                  ),
                ),

              // Input bar
              Container( // The bottom message input area
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6), // Inner spacing
                decoration: BoxDecoration(
                  color: Colors.white, // White background
                  border: Border( // Adds a thin line on top
                    top: BorderSide(color: Colors.grey.shade300),
                  ),
                ),
                child: Row( // Places input and send button side by side
                  children: [
                    Expanded(
                      child: TextField( // Input box for typing messages
                        controller: _controller, // Connects to text controller
                        onSubmitted: (text) => _sendMessage(text), // Sends when pressing enter
                        decoration: InputDecoration(
                          hintText: "Message", // Placeholder text
                          hintStyle: TextStyle(
                            color: Colors.grey.shade600, // Faint grey color
                            fontWeight: FontWeight.w400, // Regular weight
                          ),
                          filled: true, // Fills with color
                          fillColor: Colors.grey.shade100, // Light grey background
                          contentPadding: const EdgeInsets.symmetric(
                            vertical: 10, // Space above and below text
                            horizontal: 14, // Space inside box sides
                          ),
                          border: InputBorder.none, // No outline border
                          enabledBorder: InputBorder.none, // No border when enabled
                          focusedBorder: InputBorder.none, // No border when focused
                        ),
                      ),
                    ),
                    const SizedBox(width: 6), // Space before send button
                    CircleAvatar( // Round send button
                      backgroundColor: Colors.blue.shade700, // Blue background
                      child: IconButton(
                        icon: const Icon(Icons.send, // Send icon
                            color: Colors.white, size: 18),
                        onPressed: () => _sendMessage(_controller.text), // Sends message when tapped
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _Message { // Holds one chat message
  final String text; // The message text
  final bool isUser; // True if from user, false if from bot
  _Message({required this.text, required this.isUser}); // Constructor
}