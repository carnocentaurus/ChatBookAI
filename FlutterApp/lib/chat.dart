// chat.dart

import 'package:flutter/material.dart'; // gives access to all Flutter UI tools (widgets, colors, layouts, etc.).
import 'main.dart'; // imports functions or classes (like queryHandbook()) that the chat might use later.
import 'faq.dart'; // imports the FAQ page
import 'feedback.dart'; // imports the Feedback page
import 'about.dart';

// ============ RESPONSIVE HELPER CLASS ============
class ResponsiveSize {
  // This allows it to use MediaQuery to get the screen size of the current device.
  final BuildContext context;
  ResponsiveSize(this.context);

  double get screenWidth => MediaQuery.of(context).size.width;
  double get screenHeight => MediaQuery.of(context).size.height;

  bool get isSmallPhone => screenWidth < 360;
  bool get isPhone => screenWidth < 600;
  bool get isTablet => screenWidth >= 600 && screenWidth < 1024;
  bool get isDesktop => screenWidth >= 1024;

  // Sidebar takes fixed width on desktop/tablet, hidden on mobile
  double get sidebarWidth => isDesktop ? 280.0 : (isTablet ? 240.0 : 0.0);
  bool get showSidebar => isDesktop || isTablet;

  // These functions scale padding proportionally to screen width.
  // .toDouble() added to ensure Windows doesn't crash on type mismatch
  double paddingXSmall(double baseValue) => (baseValue * (screenWidth / 360)).toDouble();
  double paddingSmall(double baseValue) => (baseValue * (screenWidth / 400)).toDouble();
  double paddingMedium(double baseValue) => (baseValue * (screenWidth / 500)).toDouble();
  double paddingLarge(double baseValue) => (baseValue * (screenWidth / 600)).toDouble();
  
  // Each method returns a font size based on the screen type:
  // Using .0 ensures these are treated as doubles, fixing the 'int is not a subtype of double' error
  double fontXSmall() => isSmallPhone ? 10.0 : 12.0;
  double fontSmall() => isSmallPhone ? 12.0 : 14.0;
  double fontMedium() => isSmallPhone ? 14.0 : 16.0;
  double fontLarge() => isSmallPhone ? 16.0 : 18.0;
  double fontXLarge() => isSmallPhone ? 18.0 : 20.0;
  double fontTitle() => isSmallPhone ? 20.0 : 24.0;
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
  bool _isCancelled = false; // NEW: tracks if user cancelled the response
  String _selectedPage = 'chat'; // Tracks current page: 'chat', 'faq', 'feedback', 'about'

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

  // NEW: Function to stop the bot's response
  void _stopResponse() {
    setState(() {
      _isCancelled = true; // marks that the user wants to stop the response
      _isTyping = false; // hides the "Bot is typing..." message
    });
  }

  // heart of your chat system
  Future<void> _sendMessage(String text) async {
    final trimmedText = text.trim(); // removes leading and trailing spaces from the message
    if (trimmedText.isEmpty) return; // checks if the user typed nothing or only spaces.

    setState(() { // updates what's shown on the screen.
      _messages.add(_Message(text: trimmedText, isUser: true)); // adds the trimmed user message to the chat list
      _isTyping = true;
      _isCancelled = false; // Reset cancellation flag for new message
    });

    _scrollToBottom();
    _controller.clear(); // clears the message input box

    try {
      // sends the message (text) to the AI backend and waits for its answer.
      final answer = await queryHandbook(trimmedText); // sends the cleaned message without extra spaces

      // Check if response was cancelled before adding the message
      if (!_isCancelled) {
        setState(() {
          _messages.add(_Message(text: answer, isUser: false));
          _isTyping = false; // hides the "Bot is typing..." message.
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

  // Builds the sidebar navigation menu
  Widget _buildSidebar(ResponsiveSize responsive) {
    return Container(
      width: responsive.isPhone ? null : responsive.sidebarWidth,
      color: Colors.grey.shade50,
      child: SafeArea( // FIXED: Prevents logo from hiding under the status bar (battery/time)
        child: Column(
          children: [
            // Sidebar Header with Logo and Name
            Stack(
              children: [
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(vertical: 30, horizontal: 16),
                  child: Column(
                    children: [
                      Image.asset(
                        'assets/images/ChatBookAILogoBlue.png',
                        height: 60,
                        width: 60,
                      ),
                      const SizedBox(height: 12),
                      Text(
                        "ChatBook AI",
                        style: TextStyle(
                          fontSize: responsive.fontMedium(),
                          fontWeight: FontWeight.bold,
                          color: const Color(0xFF1976d2),
                        ),
                      ),
                    ],
                  ),
                ),
                // Close button for mobile inside the stack
                if (responsive.isPhone)
                  Positioned(
                    top: 8,
                    right: 8,
                    child: IconButton(
                      icon: const Icon(Icons.close, color: Colors.black54),
                      onPressed: () => Navigator.pop(context),
                    ),
                  ),
              ],
            ),
            
            const Divider(height: 1),

            // Sidebar Menu Items
            Expanded(
              child: ListView(
                padding: const EdgeInsets.symmetric(vertical: 8),
                children: [
                  _buildSidebarItem(
                    icon: Icons.chat_bubble_outline,
                    title: 'Chat',
                    isSelected: _selectedPage == 'chat',
                    onTap: () {
                      setState(() {
                        _selectedPage = 'chat';
                      });
                      if (responsive.isPhone) Navigator.pop(context); // Close drawer on mobile
                    },
                    responsive: responsive,
                  ),
                  _buildSidebarItem(
                    icon: Icons.help_outline,
                    title: 'FAQ',
                    isSelected: _selectedPage == 'faq',
                    onTap: () {
                      setState(() {
                        _selectedPage = 'faq';
                      });
                      if (responsive.isPhone) Navigator.pop(context); // Close drawer on mobile
                    },
                    responsive: responsive,
                  ),
                  _buildSidebarItem(
                    icon: Icons.feedback_outlined,
                    title: 'Feedback',
                    isSelected: _selectedPage == 'feedback',
                    onTap: () {
                      setState(() {
                        _selectedPage = 'feedback';
                      });
                      if (responsive.isPhone) Navigator.pop(context); // Close drawer on mobile
                    },
                    responsive: responsive,
                  ),
                  _buildSidebarItem(
                    icon: Icons.info_outline,
                    title: 'About',
                    isSelected: _selectedPage == 'about',
                    onTap: () {
                      setState(() {
                        _selectedPage = 'about';
                      });
                      if (responsive.isPhone) Navigator.pop(context); // Close drawer on mobile
                    },
                    responsive: responsive,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // Builds individual sidebar menu items
  Widget _buildSidebarItem({
    required IconData icon,
    required String title,
    required bool isSelected,
    required VoidCallback onTap,
    required ResponsiveSize responsive,
  }) {
    return InkWell(
      onTap: onTap,
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
        decoration: BoxDecoration(
          color: isSelected ? Colors.blue.shade50 : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(
          children: [
            Icon(
              icon,
              size: responsive.fontLarge(),
              color: isSelected ? Colors.blue.shade700 : Colors.black54,
            ),
            const SizedBox(width: 12),
            Text(
              title,
              style: TextStyle(
                fontSize: responsive.fontSmall(),
                fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
                color: isSelected ? Colors.blue.shade700 : Colors.black87,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // Builds the main chat interface
  Widget _buildChatInterface(ResponsiveSize responsive) {
    final bool hasMessages = _messages.isNotEmpty; // checks if there are any chat messages.

    // starts building the vertical layout of the page — everything stacked top to bottom.
    return Column(
      children: [
        Expanded( // This part should take up all remaining screen space
          child: Column(
            children: [
              Expanded( // the main chat area.
                child: hasMessages
                    ? ListView.builder( // This creates the scrollable list of messages.
                        controller: _scrollController,
                        padding: EdgeInsets.all(responsive.paddingSmall(12.0)),
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
                              margin: const EdgeInsets.symmetric(
                                vertical: 4,
                                horizontal: 8,
                              ),
                              padding: const EdgeInsets.symmetric(
                                vertical: 10,
                                horizontal: 14,
                              ),
                              constraints: BoxConstraints(
                                // FIXED: Bubble logic to prevent them from stretching too far on wide screens
                                maxWidth: responsive.isDesktop 
                                    ? 500.0 // Hard cap for desktop
                                    : (responsive.isTablet ? 400.0 : responsive.screenWidth * 0.75),
                              ),
                              decoration: BoxDecoration(
                                color: message.isUser
                                    ? Colors.green.shade600 // Green for user
                                    : const Color(0xFF1976d2), // Blue for bot
                                borderRadius: BorderRadius.only(
                                  topLeft: const Radius.circular(16),
                                  topRight: const Radius.circular(16),
                                  bottomLeft: Radius.circular(message.isUser ? 16 : 4),
                                  bottomRight: Radius.circular(message.isUser ? 4 : 16),
                                ),
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.black.withOpacity(0.05),
                                    blurRadius: 4,
                                    offset: const Offset(1, 1),
                                  ),
                                ],
                              ),
                              child: SelectableText( // The actual message text — you can even select and copy it.
                                message.text,
                                style: TextStyle(
                                  fontSize: responsive.fontSmall(),
                                  color: Colors.white,
                                  height: 1.4,
                                ),
                              ),
                            ),
                          );
                        },
                      )

                    // Splash screen when chat is empty
                    // FIXED: Reduced sizes for desktop to prevent it looking "too big"
                    : Center(
                        child: SingleChildScrollView(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Image.asset(
                                'assets/images/ChatBookAILogoBlue.png',
                                width: responsive.isDesktop ? 120 : responsive.screenWidth * 0.25,
                                height: responsive.isDesktop ? 120 : responsive.screenWidth * 0.25,
                                fit: BoxFit.contain,
                              ),
                              const SizedBox(height: 20),
                              Text(
                                "ChatBook AI",
                                style: TextStyle(
                                  fontSize: responsive.fontLarge(),
                                  fontWeight: FontWeight.w600,
                                  color: const Color(0xFF1976d2),
                                ),
                              ),
                              const SizedBox(height: 8),
                              Padding(
                                padding: const EdgeInsets.symmetric(horizontal: 40),
                                child: Text(
                                  "Ask me about the GSU student handbook",
                                  textAlign: TextAlign.center,
                                  style: TextStyle(
                                    fontSize: responsive.fontSmall(),
                                    color: Colors.grey.shade500,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
              ),

              if (_isTyping) // only appears while the AI is responding.
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      "Bot is typing...",
                      style: TextStyle(
                        fontStyle: FontStyle.italic,
                        fontSize: responsive.fontSmall(),
                        color: Colors.grey.shade600,
                      ),
                    ),
                  ),
                ),

              // This is where the user types and sends messages.
              // UI FIXED: Reduced height, smaller icon, and arrow_upward icon
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                decoration: BoxDecoration(
                  color: Colors.white,
                  border: Border(top: BorderSide(color: Colors.grey.shade200)),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: Container(
                        decoration: BoxDecoration(
                          color: Colors.grey.shade100,
                          borderRadius: BorderRadius.circular(24),
                        ),
                        child: TextField(
                          controller: _controller, // stores what you type
                          focusNode: _focusNode, // connects the focus node
                          enabled: !_isTyping, // disables input box when bot is typing
                          onTap: () => _scrollToBottom(), 
                          onSubmitted: (text) => _sendMessage(text), // sends message on Enter
                          decoration: InputDecoration(
                            hintText: "Message",
                            hintStyle: TextStyle(
                              color: Colors.grey.shade500,
                              fontSize: responsive.fontSmall(),
                            ),
                            contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                            border: InputBorder.none,
                          ),
                          style: TextStyle(fontSize: responsive.fontSmall()),
                          maxLines: null, 
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    GestureDetector(
                      onTap: _isTyping ? _stopResponse : () => _sendMessage(_controller.text),
                      child: CircleAvatar(
                        radius: 18, // Fixed smaller size for the button
                        backgroundColor: _isTyping ? Colors.red.shade600 : const Color(0xFF1976d2),
                        child: Icon(
                          _isTyping ? Icons.stop : Icons.arrow_upward, // Changed to arrow_upward
                          color: Colors.white,
                          size: 20, // Reduced icon size
                        ),
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

  // Builds FAQ page - FIXED to match new faq.dart UI
Widget _buildFaqPage(ResponsiveSize responsive) {
  return FaqPage(
    responsive: responsive, // Passes the sizing tool to faq.dart
    onQuestionTap: (question) {
      // 1. Switch the view back to the chat screen
      setState(() {
        _selectedPage = 'chat';
      });
      // 2. Automatically type and send the question
      autoQuery(question);
    },
  );
}

  // Builds Feedback page
  Widget _buildFeedbackPage(ResponsiveSize responsive) {
    return FeedbackPage(sessionId: "current_session_id", responsive: responsive);
  }

  @override
  Widget build(BuildContext context) {
    final responsive = ResponsiveSize(context); // creates a helper tool that adjusts the layout
 
    return Scaffold(
      backgroundColor: Colors.white,
      // Mobile drawer for sidebar
      drawer: responsive.showSidebar ? null : Drawer(
        child: _buildSidebar(responsive),
      ),
      // Mobile app bar (only shows on phones)
      appBar: responsive.showSidebar ? null : AppBar(
        backgroundColor: const Color(0xFF1976d2),
        elevation: 0,
        leading: Builder(
          builder: (context) => IconButton(
            icon: const Icon(Icons.menu, color: Colors.white),
            onPressed: () => Scaffold.of(context).openDrawer(),
          ),
        ),
        title: const Text(
          "ChatBook AI",
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: SafeArea( // FIXED: Prevents input box from hiding under system home/back buttons
        top: false, // AppBar already covers the top
        child: Container(
          width: double.infinity,
          height: double.infinity,
          child: Row(
            children: [
              // Persistent sidebar on desktop/tablet
              if (responsive.showSidebar) _buildSidebar(responsive),
              // Main content area
              Expanded(
                child: _selectedPage == 'chat'
                  ? _buildChatInterface(responsive)
                  : _selectedPage == 'faq'
                    ? _buildFaqPage(responsive) // This calls the helper method below
                    : _selectedPage == 'feedback'
                      ? _buildFeedbackPage(responsive)
                        : AboutPage(responsive: responsive),
               ),
            ],
          ),
        ),
      ),
    );
  }
}

class _Message { // This _Message class describes what one chat message looks like.
  final String text; // holds the actual message content
  final bool isUser;
  _Message({required this.text, required this.isUser});
}