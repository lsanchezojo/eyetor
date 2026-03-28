"""Gmail operations for the google-workspace skill."""

import argparse
import base64
import email as email_lib
import json
import os
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

sys.path.insert(0, os.path.dirname(__file__))
import _auth


def _build_service():
    try:
        from googleapiclient.discovery import build
    except ImportError:
        print(json.dumps({"ok": False, "error": "Missing dependencies. Run: pip install google-api-python-client google-auth-oauthlib google-auth-httplib2"}))
        sys.exit(1)
    creds = _auth.get_credentials()
    return build("gmail", "v1", credentials=creds)


def _extract_body(payload: dict) -> str:
    """Recursively extract plain text (preferred) or HTML body from message payload."""
    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data", "")

    if mime_type == "text/plain" and body_data:
        return base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="replace")

    parts = payload.get("parts", [])
    # Prefer text/plain
    for part in parts:
        if part.get("mimeType") == "text/plain":
            data = part.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    # Fall back to HTML or recurse
    for part in parts:
        text = _extract_body(part)
        if text:
            return text
    # Last resort: HTML
    if mime_type == "text/html" and body_data:
        return base64.urlsafe_b64decode(body_data + "==").decode("utf-8", errors="replace")

    return ""


def _get_header(headers: list, name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _parse_message_summary(msg: dict) -> dict:
    headers = msg.get("payload", {}).get("headers", [])
    label_ids = msg.get("labelIds", [])
    return {
        "id": msg.get("id", ""),
        "thread_id": msg.get("threadId", ""),
        "from": _get_header(headers, "From"),
        "to": _get_header(headers, "To"),
        "subject": _get_header(headers, "Subject"),
        "date": _get_header(headers, "Date"),
        "snippet": msg.get("snippet", ""),
        "unread": "UNREAD" in label_ids,
    }


def cmd_list(args):
    service = _build_service()
    result = service.users().messages().list(
        userId="me",
        q=args.query,
        maxResults=args.max,
    ).execute()

    messages = []
    for item in result.get("messages", []):
        msg = service.users().messages().get(
            userId="me", id=item["id"], format="metadata",
            metadataHeaders=["From", "To", "Subject", "Date"],
        ).execute()
        messages.append(_parse_message_summary(msg))

    print(json.dumps({"ok": True, "messages": messages, "count": len(messages)}))


def cmd_read(args):
    service = _build_service()
    msg = service.users().messages().get(userId="me", id=args.message_id, format="full").execute()
    headers = msg.get("payload", {}).get("headers", [])
    body = _extract_body(msg.get("payload", {}))

    # Truncate very long bodies
    if len(body) > 10000:
        body = body[:10000] + "\n... [truncated]"

    print(json.dumps({
        "ok": True,
        "id": msg.get("id"),
        "thread_id": msg.get("threadId"),
        "from": _get_header(headers, "From"),
        "to": _get_header(headers, "To"),
        "subject": _get_header(headers, "Subject"),
        "date": _get_header(headers, "Date"),
        "body": body,
    }))


def _encode_message(raw_msg) -> dict:
    return {"raw": base64.urlsafe_b64encode(raw_msg.as_bytes()).decode("utf-8")}


def cmd_send(args):
    service = _build_service()
    msg = MIMEText(args.body, "plain", "utf-8")
    msg["To"] = args.to
    msg["Subject"] = args.subject
    if args.cc:
        msg["Cc"] = args.cc

    sent = service.users().messages().send(userId="me", body=_encode_message(msg)).execute()
    print(json.dumps({"ok": True, "id": sent.get("id"), "thread_id": sent.get("threadId")}))


def cmd_reply(args):
    service = _build_service()
    # Get original message to build reply headers
    original = service.users().messages().get(
        userId="me", id=args.message_id, format="metadata",
        metadataHeaders=["From", "Subject", "Message-ID"],
    ).execute()
    headers = original.get("payload", {}).get("headers", [])

    reply_to = _get_header(headers, "From")
    subject = _get_header(headers, "Subject")
    if not subject.lower().startswith("re:"):
        subject = "Re: " + subject
    message_id = _get_header(headers, "Message-ID")
    thread_id = original.get("threadId")

    msg = MIMEText(args.body, "plain", "utf-8")
    msg["To"] = reply_to
    msg["Subject"] = subject
    if message_id:
        msg["In-Reply-To"] = message_id
        msg["References"] = message_id

    body = _encode_message(msg)
    body["threadId"] = thread_id

    sent = service.users().messages().send(userId="me", body=body).execute()
    print(json.dumps({"ok": True, "id": sent.get("id"), "thread_id": sent.get("threadId")}))


def cmd_trash(args):
    service = _build_service()
    service.users().messages().trash(userId="me", id=args.message_id).execute()
    print(json.dumps({"ok": True, "trashed": args.message_id}))


def cmd_mark_read(args):
    service = _build_service()
    service.users().messages().modify(
        userId="me", id=args.message_id,
        body={"removeLabelIds": ["UNREAD"]},
    ).execute()
    print(json.dumps({"ok": True, "marked_read": args.message_id}))


def main():
    parser = argparse.ArgumentParser(description="Gmail operations")
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="List messages")
    p_list.add_argument("--max", type=int, default=10, help="Max results (default: 10)")
    p_list.add_argument("--query", default="", help="Gmail search query (e.g. 'is:unread')")

    # read
    p_read = sub.add_parser("read", help="Read a message")
    p_read.add_argument("--message-id", required=True, help="Message ID")

    # send
    p_send = sub.add_parser("send", help="Send an email")
    p_send.add_argument("--to", required=True, help="Recipient email")
    p_send.add_argument("--subject", required=True, help="Subject")
    p_send.add_argument("--body", required=True, help="Message body")
    p_send.add_argument("--cc", default="", help="CC recipients")

    # reply
    p_reply = sub.add_parser("reply", help="Reply to a message")
    p_reply.add_argument("--message-id", required=True, help="Message ID to reply to")
    p_reply.add_argument("--body", required=True, help="Reply body")

    # trash
    p_trash = sub.add_parser("trash", help="Move message to trash")
    p_trash.add_argument("--message-id", required=True)

    # mark-read
    p_mark = sub.add_parser("mark-read", help="Mark message as read")
    p_mark.add_argument("--message-id", required=True)

    args = parser.parse_args()

    try:
        dispatch = {
            "list": cmd_list,
            "read": cmd_read,
            "send": cmd_send,
            "reply": cmd_reply,
            "trash": cmd_trash,
            "mark-read": cmd_mark_read,
        }
        dispatch[args.command](args)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
