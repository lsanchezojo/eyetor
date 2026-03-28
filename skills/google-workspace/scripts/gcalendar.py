"""Google Calendar operations for the google-workspace skill."""

import argparse
import json
import sys
import os
from datetime import datetime, timezone, timedelta

# Allow importing _auth from the same directory
sys.path.insert(0, os.path.dirname(__file__))
import _auth


def _build_service():
    try:
        from googleapiclient.discovery import build
    except ImportError:
        print(json.dumps({"ok": False, "error": "Missing dependencies. Run: pip install google-api-python-client google-auth-oauthlib google-auth-httplib2"}))
        sys.exit(1)
    creds = _auth.get_credentials()
    return build("calendar", "v3", credentials=creds)


def _parse_event(event: dict) -> dict:
    start = event.get("start", {})
    end = event.get("end", {})
    return {
        "id": event.get("id", ""),
        "title": event.get("summary", "(no title)"),
        "start": start.get("dateTime") or start.get("date", ""),
        "end": end.get("dateTime") or end.get("date", ""),
        "location": event.get("location", ""),
        "description": event.get("description", ""),
        "html_link": event.get("htmlLink", ""),
    }


def cmd_list(args):
    service = _build_service()
    now = datetime.now(timezone.utc)
    time_min = now.isoformat()
    time_max = (now + timedelta(days=args.days)).isoformat()

    result = service.events().list(
        calendarId=args.calendar,
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy="startTime",
        maxResults=50,
    ).execute()

    events = [_parse_event(e) for e in result.get("items", [])]
    print(json.dumps({"ok": True, "events": events, "count": len(events)}))


def cmd_get(args):
    service = _build_service()
    event = service.events().get(calendarId=args.calendar, eventId=args.event_id).execute()
    print(json.dumps({"ok": True, "event": _parse_event(event)}))


def cmd_create(args):
    service = _build_service()
    body = {
        "summary": args.title,
        "start": {"dateTime": args.start, "timeZone": args.timezone},
        "end": {"dateTime": args.end, "timeZone": args.timezone},
    }
    if args.description:
        body["description"] = args.description
    if args.location:
        body["location"] = args.location

    event = service.events().insert(calendarId=args.calendar, body=body).execute()
    print(json.dumps({"ok": True, "event": _parse_event(event)}))


def cmd_update(args):
    service = _build_service()
    event = service.events().get(calendarId=args.calendar, eventId=args.event_id).execute()

    if args.title:
        event["summary"] = args.title
    if args.description is not None:
        event["description"] = args.description
    if args.location is not None:
        event["location"] = args.location
    if args.start:
        event["start"] = {"dateTime": args.start, "timeZone": args.timezone}
    if args.end:
        event["end"] = {"dateTime": args.end, "timeZone": args.timezone}

    updated = service.events().update(
        calendarId=args.calendar, eventId=args.event_id, body=event
    ).execute()
    print(json.dumps({"ok": True, "event": _parse_event(updated)}))


def cmd_delete(args):
    service = _build_service()
    service.events().delete(calendarId=args.calendar, eventId=args.event_id).execute()
    print(json.dumps({"ok": True, "deleted": args.event_id}))


def cmd_list_calendars(args):
    service = _build_service()
    result = service.calendarList().list().execute()
    calendars = [
        {
            "id": cal.get("id"),
            "name": cal.get("summary"),
            "primary": cal.get("primary", False),
            "access_role": cal.get("accessRole"),
        }
        for cal in result.get("items", [])
    ]
    print(json.dumps({"ok": True, "calendars": calendars}))


def main():
    parser = argparse.ArgumentParser(description="Google Calendar operations")
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="List upcoming events")
    p_list.add_argument("--days", type=int, default=7, help="Number of days ahead (default: 7)")
    p_list.add_argument("--calendar", default="primary", help="Calendar ID (default: primary)")

    # get
    p_get = sub.add_parser("get", help="Get event details")
    p_get.add_argument("--event-id", required=True, help="Event ID")
    p_get.add_argument("--calendar", default="primary")

    # create
    p_create = sub.add_parser("create", help="Create an event")
    p_create.add_argument("--title", required=True, help="Event title")
    p_create.add_argument("--start", required=True, help="Start datetime (ISO 8601)")
    p_create.add_argument("--end", required=True, help="End datetime (ISO 8601)")
    p_create.add_argument("--description", default="", help="Event description")
    p_create.add_argument("--location", default="", help="Event location")
    p_create.add_argument("--calendar", default="primary")
    p_create.add_argument("--timezone", default="UTC", help="Timezone (default: UTC)")

    # update
    p_update = sub.add_parser("update", help="Update an event")
    p_update.add_argument("--event-id", required=True)
    p_update.add_argument("--title")
    p_update.add_argument("--start", help="New start datetime (ISO 8601)")
    p_update.add_argument("--end", help="New end datetime (ISO 8601)")
    p_update.add_argument("--description")
    p_update.add_argument("--location")
    p_update.add_argument("--calendar", default="primary")
    p_update.add_argument("--timezone", default="UTC")

    # delete
    p_delete = sub.add_parser("delete", help="Delete an event")
    p_delete.add_argument("--event-id", required=True)
    p_delete.add_argument("--calendar", default="primary")

    # list-calendars
    sub.add_parser("list-calendars", help="List available calendars")

    args = parser.parse_args()

    try:
        dispatch = {
            "list": cmd_list,
            "get": cmd_get,
            "create": cmd_create,
            "update": cmd_update,
            "delete": cmd_delete,
            "list-calendars": cmd_list_calendars,
        }
        dispatch[args.command](args)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
