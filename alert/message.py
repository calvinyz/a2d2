"""
SMS Message Module
Provides text message sending capabilities with console output for testing.
"""

from typing import Optional

class MessageSender:
    """
    SMS message sending utility.
    
    Simulates sending text messages by printing to console.
    Can be extended to use actual SMS service.
    """
    
    def send_text_message(
        self,
        recipient: str,
        message: str
    ) -> None:
        """
        Send a text message to recipient.
        
        Args:
            recipient: Phone number of recipient
            message: Content of text message
            
        Note:
            Currently prints to console for testing/simulation
        """
        print(f"Sending SMS to {recipient}: {message}")

def main():
    """Test message sending functionality."""
    sms = MessageSender()
    sms.send_text_message('555-555-5555', 'Hello, world!')

if __name__ == '__main__':
    main()