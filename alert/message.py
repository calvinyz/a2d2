class MessageSender:
    def sendTxtMsg(self, recipient, message):
        
        print(f"Sending SMS to {recipient}: {message}")

if __name__ == '__main__':
    sms = MessageSender()
    sms.sendTxtMsg('555-555-5555', 'Hello, world!')