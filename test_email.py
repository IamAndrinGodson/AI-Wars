import smtplib
import yaml
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def test_email():
    try:
        print("Loading configuration...")
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        # Try to find notifications in root or under response
        notifications = config.get('notifications')
        if not notifications and 'response' in config:
            notifications = config.get('response', {}).get('notifications')
            
        email_config = notifications.get('email', {}) if notifications else {}
        
        server_host = email_config.get('smtp_server')
        server_port = email_config.get('smtp_port')
        sender = email_config.get('sender_email')
        password = email_config.get('sender_password')
        recipients = email_config.get('recipients')
        
        print(f"Server: {server_host}:{server_port}")
        print(f"Sender: {sender}")
        print(f"Recipients: {recipients}")
        print(f"Password provided: {'YES' if password else 'NO'}")
        
        print("\nConnecting to SMTP server...")
        server = smtplib.SMTP(server_host, server_port)
        server.set_debuglevel(1)
        
        print("Starting TLS...")
        server.starttls()
        
        print("Logging in...")
        server.login(sender, password)
        print("Login successful!")
        
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = recipients[0]
        msg['Subject'] = "Test Email from ML Threat Detector"
        msg.attach(MIMEText("This is a test email to verify configuration.", 'plain'))
        
        print("Sending message...")
        server.send_message(msg)
        server.quit()
        
        print("\nSUCCESS! Email sent successfully.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_email()
