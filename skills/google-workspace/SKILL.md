---
name: google-workspace
description: Query and modify Google Calendar events, Gmail messages, and Google Tasks for a connected Google account.
license: MIT
compatibility: Python 3.11+. Requires google-api-python-client, google-auth-oauthlib, google-auth-httplib2.
metadata:
  author: eyetor
  version: "1.0"
---

# Google Workspace

Interact with Google Calendar, Gmail, and Google Tasks for a personal Google account.

## Setup (first time only)

1. Go to [Google Cloud Console](https://console.cloud.google.com/), create a project and enable the Calendar API, Gmail API, and Tasks API.
2. Create OAuth 2.0 credentials (type: **Desktop App**), download `credentials.json`.
3. Place it at `~/.eyetor/google_credentials.json` (or set `GOOGLE_CREDENTIALS_FILE` env var to a custom path).
4. Install dependencies: `pip install google-api-python-client google-auth-oauthlib google-auth-httplib2`
5. On first use, a browser window will open asking to authorize the app. The token is saved automatically.

## Google Calendar

### List upcoming events
```
scripts/gcalendar.py list --days 7
scripts/gcalendar.py list --days 3 --calendar primary
```
Returns: `{"ok": true, "events": [{"id": "...", "title": "...", "start": "...", "end": "...", "location": "...", "description": "..."}]}`

### Get event details
```
scripts/gcalendar.py get --event-id EVENT_ID
scripts/gcalendar.py get --event-id EVENT_ID --calendar primary
```

### Create an event
```
scripts/gcalendar.py create --title "Meeting" --start "2026-03-28T10:00:00+01:00" --end "2026-03-28T11:00:00+01:00"
scripts/gcalendar.py create --title "Lunch" --start "2026-03-28T13:00:00+01:00" --end "2026-03-28T14:00:00+01:00" --location "Cafe Central" --description "With team"
```

### Update an event
```
scripts/gcalendar.py update --event-id EVENT_ID --title "New Title"
scripts/gcalendar.py update --event-id EVENT_ID --start "2026-03-28T11:00:00+01:00" --end "2026-03-28T12:00:00+01:00"
```

### Delete an event
```
scripts/gcalendar.py delete --event-id EVENT_ID
```

### List available calendars
```
scripts/gcalendar.py list-calendars
```
Returns: `{"ok": true, "calendars": [{"id": "primary", "name": "My Calendar", "primary": true}]}`

## Gmail

### List messages
```
scripts/gmail.py list
scripts/gmail.py list --max 20 --query "is:unread"
scripts/gmail.py list --query "from:boss@company.com subject:report"
```
Gmail search syntax is supported in `--query` (is:unread, from:, subject:, after:, before:, has:attachment, etc.)
Returns: `{"ok": true, "messages": [{"id": "...", "from": "...", "subject": "...", "date": "...", "snippet": "...", "unread": true}]}`

### Read a message
```
scripts/gmail.py read --message-id MESSAGE_ID
```
Returns full message body (plain text preferred, HTML fallback).

### Send an email
```
scripts/gmail.py send --to "recipient@example.com" --subject "Hello" --body "Message body here"
scripts/gmail.py send --to "a@b.com" --subject "Hi" --body "..." --cc "c@d.com"
```

### Reply to a message
```
scripts/gmail.py reply --message-id MESSAGE_ID --body "My reply text"
```

### Trash a message
```
scripts/gmail.py trash --message-id MESSAGE_ID
```

### Mark as read
```
scripts/gmail.py mark-read --message-id MESSAGE_ID
```

## Google Tasks

### List task lists
```
scripts/tasks.py list-tasklists
```
Returns: `{"ok": true, "tasklists": [{"id": "...", "title": "My Tasks"}]}`

### List tasks
```
scripts/tasks.py list
scripts/tasks.py list --list-id TASKLIST_ID --show-completed true
```
Returns: `{"ok": true, "tasks": [{"id": "...", "title": "...", "notes": "...", "due": "...", "completed": false}]}`

### Create a task
```
scripts/tasks.py create --title "Buy groceries"
scripts/tasks.py create --title "Submit report" --notes "Q1 financials" --due "2026-03-31T00:00:00Z" --list-id TASKLIST_ID
```

### Complete a task
```
scripts/tasks.py complete --task-id TASK_ID
scripts/tasks.py complete --task-id TASK_ID --list-id TASKLIST_ID
```

### Update a task
```
scripts/tasks.py update --task-id TASK_ID --title "New title" --notes "Updated notes" --due "2026-04-01T00:00:00Z"
```

### Delete a task
```
scripts/tasks.py delete --task-id TASK_ID
scripts/tasks.py delete --task-id TASK_ID --list-id TASKLIST_ID
```

## Notes

- All dates/times use ISO 8601 format (e.g. `2026-03-28T10:00:00+01:00`)
- Default calendar is `primary` when `--calendar` is omitted
- Default task list is the user's primary list when `--list-id` is omitted
- Gmail `--query` uses [Gmail search operators](https://support.google.com/mail/answer/7190)
- `GOOGLE_CREDENTIALS_FILE` env var overrides default credentials path (`~/.eyetor/google_credentials.json`)
