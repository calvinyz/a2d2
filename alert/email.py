import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class EmailSender:
    def send_email(self, to_email, subject, body):
        # Set up the email message
        msg = MIMEMultipart()
        msg['From'] = 'sender@example.com'
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Connect to the SMTP server
        server = smtplib.SMTP('smtp.example.com', 587)
        server.starttls()
        server.login('username', 'password')

        # Send the email
        server.send_message(msg)

        # Disconnect from the SMTP server
        server.quit()