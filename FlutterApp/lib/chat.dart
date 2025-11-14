// chat.dart

import 'package:flutter/material.dart';
import 'main.dart';

// ============ RESPONSIVE HELPER CLASS ============
class ResponsiveSize {
  final BuildContext context;
  ResponsiveSize(this.context);

  double get screenWidth => MediaQuery.of(context).size.width;
  double get screenHeight => MediaQuery.of(context).size.height;

  bool get isSmallPhone => screenWidth < 360;
  bool get isPhone => screenWidth < 600;
  bool get isTablet => screenWidth >= 600;

  double paddingXSmall(double baseValue) => baseValue * (screenWidth / 360);
  double paddingSmall(double baseValue) => baseValue * (screenWidth / 400);
  double paddingMedium(double baseValue) => baseValue * (screenWidth / 500);
  double paddingLarge(double baseValue) => baseValue * (screenWidth / 600);
 
  double fontXSmall() => isSmallPhone ? 10 : 12;
  double fontSmall() => isSmallPhone ? 12 : 14;
  double fontMedium() => isSmallPhone ? 14 : 16;
  double fontLarge() => isSmallPhone ? 16 : 18;
  double fontXLarge() => isSmallPhone ? 18 : 20;
  double fontTitle() => isSmallPhone ? 20 : 24;
}


class ChatPage extends StatefulWidget {
  const ChatPage({Key? key}) : super(key: key);

  @override
  ChatPageState createState() => ChatPageState();
}

class ChatPageState extends State<ChatPage> {
  final List<_Message> _messages = [];
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final FocusNode _focusNode = FocusNode();
  bool _isTyping = false;
  bool _isCancelled = false; // NEW: tracks if user cancelled the response

  
  @override
  void initState() {
    super.initState();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        _focusNode.requestFocus();
      }
    });
  }

  @override
  void dispose() {
    _scrollController.dispose();
    _controller.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }
  
  void autoQuery(String question) {
    _controller.text = question;
    _sendMessage(question);
  }

  // NEW: Function to stop the bot's response
  void _stopResponse() {
    setState(() {
      _isCancelled = true;
      _isTyping = false;
    });
  }

  Future<void> _sendMessage(String text) async {
    final trimmedText = text.trim();
    if (trimmedText.isEmpty) return;

    setState(() {
      _messages.add(_Message(text: trimmedText, isUser: true));
      _isTyping = true;
      _isCancelled = false; // Reset cancellation flag
    });

    _scrollToBottom();
    _controller.clear();

    try {
      final answer = await queryHandbook(trimmedText);

      // Check if response was cancelled before adding the message
      if (!_isCancelled) {
        setState(() {
          _messages.add(_Message(text: answer, isUser: false));
          _isTyping = false;
        });

        _scrollToBottom();
      } else {
        // If cancelled, add a cancellation message
        setState(() {
          _messages.add(_Message(text: "Response stopped by user.", isUser: false));
        });
        _scrollToBottom();
      }
    } 
    catch (e) {
      // Only show error if not cancelled
      if (!_isCancelled) {
        setState(() {
          _messages.add(
            _Message(text: "⚠️ Cannot connect to server.", isUser: false),
          );
          _isTyping = false;
        });

        _scrollToBottom();
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final bool hasMessages = _messages.isNotEmpty;
    final responsive = ResponsiveSize(context);
 
    return Column(
      children: [
        Expanded(
          child: Column(
            children: [
              Expanded(
                child: hasMessages
                    ? ListView.builder(
                        controller: _scrollController,
                        padding: EdgeInsets.all(responsive.paddingSmall(12)),
                        itemCount: _messages.length,
                        itemBuilder: (context, index) {
                          final message = _messages[index];
                          return Align(
                            alignment: message.isUser
                                ? Alignment.centerRight
                                : Alignment.centerLeft,
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
                              child: SelectableText(
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

              if (_isTyping)
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
                        controller: _controller,
                        focusNode: _focusNode,
                        enabled: !_isTyping,
                        onTap: () {
                          _scrollToBottom();
                        },
                        onSubmitted: (text) => _sendMessage(text),
                        decoration: InputDecoration(
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
                        maxLines: null,
                        textCapitalization: TextCapitalization.sentences,
                      ),
                    ),
                    SizedBox(width: responsive.paddingSmall(6)),
                    CircleAvatar(
                      radius: responsive.screenWidth * 0.06,
                      backgroundColor: _isTyping 
                          ? Colors.red.shade600 // Red when typing (stop button)
                          : Colors.blue.shade700, // Blue when ready to send
                      child: IconButton(
                        icon: Icon(
                          _isTyping ? Icons.stop : Icons.send, // Changes icon based on state
                          color: Colors.white,
                          size: responsive.fontMedium(),
                        ),
                        onPressed: _isTyping 
                            ? _stopResponse // Call stop function when typing
                            : () => _sendMessage(_controller.text), // Send message when not typing
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


class _Message {
  final String text;
  final bool isUser;
  _Message({required this.text, required this.isUser});
}