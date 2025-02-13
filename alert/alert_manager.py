"""
Unified Alert Management System
Handles all types of alerts with priority and fallback support.
"""

from typing import Dict, List, Optional
from enum import Enum
from .email import EmailSender
from .message import MessageSender

class AlertPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class AlertType(Enum):
    EMAIL = "email"
    SMS = "sms"
    ALL = "all"

class AlertManager:
    def __init__(self, config: Dict):
        self.config = config
        self.email_sender = EmailSender()
        self.sms_sender = MessageSender()
        
    def send_alert(
        self,
        message: str,
        priority: AlertPriority,
        alert_type: AlertType = AlertType.ALL,
        recipients: Optional[Dict[str, List[str]]] = None
    ) -> bool:
        """
        Send alert through configured channels based on priority.
        
        Args:
            message: Alert content
            priority: Alert priority level
            alert_type: Type of alert to send
            recipients: Optional override of default recipients
        """
        success = True
        
        if alert_type in [AlertType.EMAIL, AlertType.ALL]:
            try:
                self._send_email_alert(message, priority, recipients)
            except Exception as e:
                success = False
                self._handle_alert_failure("email", e)
                
        if alert_type in [AlertType.SMS, AlertType.ALL]:
            try:
                self._send_sms_alert(message, priority, recipients)
            except Exception as e:
                success = False
                self._handle_alert_failure("sms", e)
                
        return success

    def _send_email_alert(self, message: str, priority: AlertPriority, recipients: Optional[Dict] = None):
        email_recipients = recipients.get('email') if recipients else self.config['email']['recipients']
        subject = f"{priority.name} Priority Alert"
        
        for recipient in email_recipients:
            self.email_sender.send_email(recipient, subject, message)

    def _send_sms_alert(self, message: str, priority: AlertPriority, recipients: Optional[Dict] = None):
        sms_recipients = recipients.get('sms') if recipients else self.config['sms']['recipients']
        
        for recipient in sms_recipients:
            self.sms_sender.send_text_message(recipient, message)

    def _handle_alert_failure(self, alert_type: str, error: Exception):
        # Log failure and potentially trigger fallback
        print(f"Failed to send {alert_type} alert: {str(error)}") 