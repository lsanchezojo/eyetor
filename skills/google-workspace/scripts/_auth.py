"""Shared Google OAuth2 authentication helper for google-workspace scripts."""

import json
import os
import sys
from pathlib import Path

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/tasks",
]

CREDS_FILE = Path(
    os.environ.get("GOOGLE_CREDENTIALS_FILE", "~/.eyetor/google_credentials.json")
).expanduser()
TOKEN_FILE = Path("~/.eyetor/google_token.json").expanduser()


def get_credentials():
    """Return valid Google credentials, refreshing or initiating OAuth flow as needed."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(json.dumps({
            "ok": False,
            "error": (
                "Missing dependencies. Run: "
                "pip install google-api-python-client google-auth-oauthlib google-auth-httplib2"
            ),
        }))
        sys.exit(1)

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDS_FILE.exists():
                print(json.dumps({
                    "ok": False,
                    "error": (
                        f"Google credentials file not found at {CREDS_FILE}. "
                        "Download credentials.json from Google Cloud Console "
                        "(OAuth 2.0 > Desktop App) and place it there, or set "
                        "GOOGLE_CREDENTIALS_FILE to its path."
                    ),
                }))
                sys.exit(1)
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(creds.to_json())

    return creds
