// chat.dart

import 'package:flutter/material.dart'; // gives access to all Flutter UI tools (widgets, colors, layouts, etc.).
import 'main.dart'; // imports functions or classes (like queryHandbook()) that the chat might use later.

// ============ RESPONSIVE HELPER CLASS ============
class ResponsiveSize {
  // This allows it to use MediaQuery to get the screen size of the current device.
  final BuildContext context;
  ResponsiveSize(this.context);

  double get screenWidth => MediaQuery.of(context).size.width;
  double get screenHeight => MediaQuery.of(context).size.height;

  bool get isSmallPhone => screenWidth < 360;
  bool get isPhone => screenWidth < 600;
  bool get isTablet => screenWidth >= 600;

  // These functions scale padding proportionally to screen width.
  double paddingXSmall(double baseValue) => baseValue * (screenWidth / 360);
  double paddingSmall(double baseValue) => baseValue * (screenWidth / 400);
  double paddingMedium(double baseValue) => baseValue * (screenWidth / 500);
  double paddingLarge(double baseValue) => baseValue * (screenWidth / 600);
 
  // Each method returns a font size based on the screen type:
  double fontXSmall() => isSmallPhone ? 10 : 12;
  double fontSmall() => isSmallPhone ? 12 : 14;
  double fontMedium() => isSmallPhone ? 14 : 16;
  double fontLarge() => isSmallPhone ? 16 : 18;
  double fontXLarge() => isSmallPhone ? 18 : 20;
  double fontTitle() => isSmallPhone ? 20 : 24;
}


class ChatPage extends StatefulWidget {
  // Key? key is an optional identifier used internally by Flutter to optimize widget rebuilding.
  // super(key: key) passes that key to the parent class (StatefulWidget).
  const ChatPage({Key? key}) : super(key: key);

  @override
  // createState() tells Flutter which State class should manage this widget's behavior.
  ChatPageState createState() => ChatPageState();
}

// This defines the state (the dynamic part) for the ChatPage
class ChatPageState extends State<ChatPage> {
  final List<_Message> _messages = []; // A list holding all chat messages.
  final TextEditingController _controller = TextEditingController(); // This controller manages the TextField (the input box).
  final ScrollController _scrollController = ScrollController(); // Used to automatically scroll to the bottom when new messages appear.
  final FocusNode _focusNode = FocusNode(); // Manages keyboard focus on the input field.
  bool _isTyping = false; // tracks whether the AI is currently "typing."

  
@override
void initState() {
  super.initState();

  // Request focus immediately after the first frame is rendered
  WidgetsBinding.instance.addPostFrameCallback((_) {
    if (mounted) {
      _focusNode.requestFocus();
    }
  });
}


  // what happens when the page is about to be removed or closed.
  @override
  void dispose() { // This runs when the chat page is closing or being removed from memory.
    _scrollController.dispose(); // Stops and deletes the scroll controller
    _controller.dispose(); // Closes the text controller — the one that handled the text box input.
    _focusNode.dispose(); // Releases the focus control (keyboard management)
    super.dispose(); // this runs Flutter's default cleanup too, to make sure nothing is left behind.
  }

  // automatically scrolls the chat window down to show the latest message
  void _scrollToBottom() {
    // Wait until the screen finishes updating, then run this code
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) { // checks if the chat list is actually on screen and ready to scroll.
        _scrollController.animateTo( // Scroll down smoothly to the bottom of the chat messages.
          _scrollController.position.maxScrollExtent, // Go to the very bottom of the scrollable area (where the latest messages are)
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }
  
  // automatically fills in the chat box with a question and sends it
  void autoQuery(String question) {
    _controller.text = question; // Put the given text (question) inside the chat input box.
    _sendMessage(question);
  }


  // heart of your chat system
  Future<void> _sendMessage(String text) async {
    final trimmedText = text.trim(); // removes leading and trailing spaces from the message
    if (trimmedText.isEmpty) return; // checks if the user typed nothing or only spaces.

    setState(() { // updates what's shown on the screen.
      _messages.add(_Message(text: trimmedText, isUser: true)); // adds the trimmed user message to the chat list
      _isTyping = true;
    });

    _scrollToBottom();
    _controller.clear(); // clears the message input box

    try {
      // sends the message (text) to the AI backend and waits for its answer.
      final answer = await queryHandbook(trimmedText); // sends the cleaned message without extra spaces

      setState(() {
        _messages.add(_Message(text: answer, isUser: false));
        _isTyping = false; // hides the "Bot is typing..." message.
      });

      _scrollToBottom();
    } 
    catch (e) {
      setState(() {
        _messages.add(
          _Message(text: "⚠️ Cannot connect to server.", isUser: false),
        );
        _isTyping = false;
      });

      _scrollToBottom();
    }
  }

  @override
  Widget build(BuildContext context) {
    final bool hasMessages = _messages.isNotEmpty; // checks if there are any chat messages.
    final responsive = ResponsiveSize(context); // creates a helper tool that adjusts the layout depending on screen size
 
    // tarts building the vertical layout of the page — everything stacked top to bottom.
    return Column(
      children: [
        Expanded( // This part should take up all remaining screen space
          child: Column(
            children: [
              Expanded( // the main chat area.
                child: hasMessages
                    ? ListView.builder( // This creates the scrollable list of messages.
                        controller: _scrollController,
                        padding: EdgeInsets.all(responsive.paddingSmall(12)),
                        itemCount: _messages.length,
                        itemBuilder: (context, index) { // This creates the scrollable list of messages.
                          final message = _messages[index];
                          // Aligns it left or right depending on who sent it
                          return Align(
                            alignment: message.isUser
                                ? Alignment.centerRight
                                : Alignment.centerLeft,
                            // Each message is wrapped in a Container that styles the bubble.
                            child: Container(
                              margin: EdgeInsets.symmetric(
                                vertical: responsive.paddingSmall(4),
                              ),
                              padding: EdgeInsets.symmetric(
                                vertical: responsive.paddingSmall(12),
                                horizontal: responsive.paddingSmall(16),
                              ),
                              constraints: BoxConstraints(
                                maxWidth: responsive.screenWidth * 0.85,
                              ),
                              decoration: BoxDecoration(
                                color: message.isUser
                                    ? Colors.blue.shade700
                                    : Colors.green.shade200,
                                borderRadius: BorderRadius.circular(
                                  responsive.paddingSmall(20),
                                ),
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.black.withOpacity(0.05),
                                    blurRadius: 4,
                                    offset: const Offset(2, 2),
                                  ),
                                ],
                              ),
                              child: SelectableText( // The actual message text — you can even select and copy it.
                                message.text,
                                style: TextStyle(
                                  fontSize: responsive.fontSmall(),
                                  color: message.isUser
                                      ? Colors.white
                                      : Colors.black87,
                                  height: 1.4,
                                ),
                              ),
                            ),
                          );
                        },
                      )

                    // Splash screen when chat is empty
                    : Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Image.asset(
                              'assets/images/ChatBookAILogoSplashState.png',
                              width: responsive.screenWidth * 0.35,
                              height: responsive.screenWidth * 0.35,
                              fit: BoxFit.contain,
                            ),
                            SizedBox(height: responsive.paddingMedium(16)),
                            Text(
                              "ChatBook AI",
                              style: TextStyle(
                                fontSize: responsive.fontTitle(),
                                fontWeight: FontWeight.w600,
                                color: const Color(0xFF1976d2),
                                letterSpacing: 0.5,
                              ),
                            ),
                            SizedBox(height: responsive.paddingSmall(8)),
                            Padding(
                              padding: EdgeInsets.symmetric(
                                horizontal: responsive.paddingMedium(24),
                              ),
                              child: Text(
                                "Ask me about the GSU student handbook",
                                textAlign: TextAlign.center,
                                style: TextStyle(
                                  fontSize: responsive.fontSmall(),
                                  color: Colors.grey,
                                  height: 1.4,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
              ),

              if (_isTyping) // only appears while the AI is responding.
                Padding(
                  padding: EdgeInsets.all(responsive.paddingSmall(8)),
                  child: Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      "Bot is typing...",
                      style: TextStyle(
                        fontStyle: FontStyle.italic,
                        fontSize: responsive.fontSmall(),
                        color: Colors.grey.shade700,
                      ),
                    ),
                  ),
                ),

              // This is where the user types and sends messages.
              Container(
                padding: EdgeInsets.only(
                  left: responsive.paddingSmall(8),
                  right: responsive.paddingSmall(8),
                  top: responsive.paddingSmall(6),
                  bottom: responsive.paddingSmall(6) + MediaQuery.of(context).padding.bottom,
                ),
                decoration: BoxDecoration(
                  color: Colors.white,
                  border: Border(
                    top: BorderSide(color: Colors.grey.shade300),
                  ),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _controller, // stores what you type
                        focusNode: _focusNode, // connects the focus node
                        enabled: !_isTyping, // disables input box when bot is typing to prevent multiple messages
                        onTap: () { // runs when the user taps/clicks on the input box
                          _scrollToBottom(); // automatically scrolls chat to bottom when input is tapped
                        },
                        onSubmitted: (text) => _sendMessage(text), // sends the message when you press Enter
                        decoration: InputDecoration( // adds the gray background and "Message" hint
                          hintText: "Message",
                          hintStyle: TextStyle(
                            color: Colors.grey.shade600,
                            fontWeight: FontWeight.w400,
                            fontSize: responsive.fontSmall(),
                          ),
                          filled: true,
                          fillColor: Colors.grey.shade100,
                          contentPadding: EdgeInsets.symmetric(
                            vertical: responsive.paddingSmall(10),
                            horizontal: responsive.paddingSmall(14),
                          ),
                          border: InputBorder.none,
                          enabledBorder: InputBorder.none,
                          focusedBorder: InputBorder.none,
                        ),
                        style: TextStyle(
                          fontSize: responsive.fontSmall(),
                        ),
                        maxLines: null, // lets the box grow as you type long messages
                        textCapitalization: TextCapitalization.sentences,
                      ),
                    ),
                    SizedBox(width: responsive.paddingSmall(6)),
                    CircleAvatar(
                      radius: responsive.screenWidth * 0.06,
                      backgroundColor: _isTyping 
                          ? Colors.grey.shade400 // gray color when bot is typing (disabled state)
                          : Colors.blue.shade700, // normal blue color when ready to send
                      child: IconButton(
                        icon: Icon(
                          Icons.send,
                          color: Colors.white,
                          size: responsive.fontMedium(),
                        ),
                        onPressed: _isTyping 
                            ? null // disables send button when bot is typing
                            : () => _sendMessage(_controller.text), // sends message when enabled
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


class _Message { // This _Message class describes what one chat message looks like.
  final String text; // holds the actual message content
  final bool isUser;
  _Message({required this.text, required this.isUser});
}