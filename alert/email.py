"""
Email Alert Module
Provides email sending capabilities with SMTP server configuration.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

class EmailSender:
    """
    Email sending utility with SMTP server support.
    
    Handles email composition and delivery through configured SMTP server
    with TLS encryption.
    """
    
    # SMTP Configuration
    SMTP_CONFIG = {
        'server': 'smtp.example.com',
        'port': 587,
        'sender': 'sender@example.com',
        'credentials': {
            'username': 'username',
            'password': 'password'
        }
    }

    def send_email(
        self,
        to_email: str,
        subject: str,
        body: str
    ) -> None:
        """
        Send an email through configured SMTP server.
        
        Args:
            to_email: Recipient email address
            subject: Email subject line
            body: Email body content
            
        Raises:
            smtplib.SMTPException: If email sending fails
        """
        try:
            # Compose email message
            msg = self._create_message(to_email, subject, body)
            
            # Send via SMTP server
            with self._create_smtp_connection() as server:
                server.send_message(msg)
                
        except smtplib.SMTPException as e:
            # Log error and re-raise
            print(f"Failed to send email: {str(e)}")
            raise

    def _create_message(
        self,
        to_email: str,
        subject: str,
        body: str
    ) -> MIMEMultipart:
        """
        Create MIME email message.
        
        Args:
            to_email: Recipient email address
            subject: Email subject line
            body: Email body content
            
        Returns:
            MIMEMultipart: Formatted email message
        """
        msg = MIMEMultipart()
        msg['From'] = self.SMTP_CONFIG['sender']
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Attach plain text body
        msg.attach(MIMEText(body, 'plain'))
        
        return msg

    def _create_smtp_connection(self) -> smtplib.SMTP:
        """
        Create authenticated SMTP server connection.
        
        Returns:
            SMTP: Connected and authenticated SMTP server
            
        Raises:
            smtplib.SMTPException: If connection fails
        """
        # Connect to SMTP server
        server = smtplib.SMTP(
            self.SMTP_CONFIG['server'],
            self.SMTP_CONFIG['port']
        )
        
        # Enable TLS encryption
        server.starttls()
        
        # Authenticate
        server.login(
            self.SMTP_CONFIG['credentials']['username'],
            self.SMTP_CONFIG['credentials']['password']
        )
        
        return server