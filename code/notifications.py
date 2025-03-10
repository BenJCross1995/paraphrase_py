import os
import json
import requests
from typing import Optional

class PushoverNotifier:
    """
    A notifier class for sending notifications using the Pushover API.

    Attributes:
        user_key (str): The Pushover user key.
        api_token (str): The Pushover API token.
    """

    def __init__(self, credential_path: str):
        """
        Initialize the PushoverNotifier by loading credentials from a JSON file.

        Args:
            credential_path (str): The path to the JSON file containing the credentials.
        """
        self._load_credentials(credential_path)

    def _load_credentials(self, credential_path: str) -> None:
        """
        Load credentials from a JSON file and set them as instance attributes.

        Also sets the credentials as environment variables if needed.

        Args:
            credential_path (str): The path to the JSON file containing the credentials.
        """
        with open(credential_path, 'r') as file:
            data = json.load(file)

        self.user_key = data.get('PUSHOVER_USER_KEY')
        self.api_token = data.get('PUSHOVER_API_TOKEN')

        # Optionally set the credentials in the environment for global access.
        os.environ["PUSHOVER_USER_KEY"] = self.user_key
        os.environ["PUSHOVER_API_TOKEN"] = self.api_token

    def send_notification(self, message: str, retry_count: int = 3, timeout: Optional[float] = None) -> bool:
        """
        Send a notification using the Pushover API.

        Args:
            message (str): The message to send.
            retry_count (int): The number of attempts to send the notification.
            timeout (Optional[float]): The timeout for the HTTP request in seconds.

        Returns:
            bool: True if the notification was sent successfully, False otherwise.
        """
        payload = {
            "token": self.api_token,
            "user": self.user_key,
            "message": message
        }

        for attempt in range(retry_count):
            try:
                response = requests.post(
                    "https://api.pushover.net/1/messages.json",
                    data=payload,
                    timeout=timeout
                )
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                # You can log the error here if needed.
                continue

        return False

# Example usage:
if __name__ == "__main__":
    credential_path = "path/to/credentials.json"
    notifier = PushoverNotifier(credential_path)
    
    if notifier.send_notification("Test notification"):
        print("Notification sent successfully.")
    else:
        print("Failed to send notification.")
