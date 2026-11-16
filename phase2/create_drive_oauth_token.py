"""Create a Google Drive refresh token locally for the Kaggle B5 backup.

Run this on your own computer, not in Kaggle. The script prints the refresh
token once so it can be entered directly as a Kaggle Secret.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Authorize the B5 Kaggle notebook to create its own Drive backups."
    )
    parser.add_argument(
        "client_secrets",
        type=Path,
        help="Path to the OAuth Desktop client JSON downloaded from Google Cloud",
    )
    args = parser.parse_args()
    if not args.client_secrets.is_file():
        raise FileNotFoundError(f"OAuth client JSON not found: {args.client_secrets}")

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as error:
        raise RuntimeError(
            "Install the local OAuth helper with: python -m pip install google-auth-oauthlib"
        ) from error

    flow = InstalledAppFlow.from_client_secrets_file(
        str(args.client_secrets),
        scopes=[DRIVE_FILE_SCOPE],
    )
    credentials = flow.run_local_server(
        port=0,
        access_type="offline",
        prompt="consent",
    )
    if not credentials.refresh_token:
        raise RuntimeError(
            "Google did not return a refresh token. Revoke this app's access, then run again and approve consent."
        )

    print("Copy this refresh token directly into the Kaggle Secret named GOOGLE_OAUTH_REFRESH_TOKEN:")
    print(credentials.refresh_token)
    print("Do not paste the token into notebook code, GitHub, or chat.")


if __name__ == "__main__":
    main()
