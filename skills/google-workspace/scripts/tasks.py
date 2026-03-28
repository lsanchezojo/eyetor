"""Google Tasks operations for the google-workspace skill."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import _auth


def _build_service():
    try:
        from googleapiclient.discovery import build
    except ImportError:
        print(json.dumps({"ok": False, "error": "Missing dependencies. Run: pip install google-api-python-client google-auth-oauthlib google-auth-httplib2"}))
        sys.exit(1)
    creds = _auth.get_credentials()
    return build("tasks", "v1", credentials=creds)


def _get_default_tasklist(service) -> str:
    """Return the ID of the user's primary task list."""
    result = service.tasklists().list(maxResults=1).execute()
    items = result.get("items", [])
    if not items:
        raise RuntimeError("No task lists found in the account.")
    return items[0]["id"]


def _parse_task(task: dict) -> dict:
    return {
        "id": task.get("id", ""),
        "title": task.get("title", ""),
        "notes": task.get("notes", ""),
        "due": task.get("due", ""),
        "completed": task.get("status") == "completed",
        "updated": task.get("updated", ""),
    }


def cmd_list_tasklists(args):
    service = _build_service()
    result = service.tasklists().list(maxResults=100).execute()
    tasklists = [
        {"id": tl.get("id"), "title": tl.get("title")}
        for tl in result.get("items", [])
    ]
    print(json.dumps({"ok": True, "tasklists": tasklists}))


def _resolve_list(service, list_id: str) -> str:
    if list_id and list_id != "@default":
        return list_id
    return _get_default_tasklist(service)


def cmd_list(args):
    service = _build_service()
    list_id = _resolve_list(service, args.list_id)

    kwargs = {"tasklist": list_id, "maxResults": 100}
    if args.show_completed:
        kwargs["showCompleted"] = True
        kwargs["showHidden"] = True

    result = service.tasks().list(**kwargs).execute()
    tasks = [_parse_task(t) for t in result.get("items", [])]
    if not args.show_completed:
        tasks = [t for t in tasks if not t["completed"]]

    print(json.dumps({"ok": True, "tasks": tasks, "count": len(tasks), "list_id": list_id}))


def cmd_create(args):
    service = _build_service()
    list_id = _resolve_list(service, args.list_id)

    body = {"title": args.title}
    if args.notes:
        body["notes"] = args.notes
    if args.due:
        body["due"] = args.due

    task = service.tasks().insert(tasklist=list_id, body=body).execute()
    print(json.dumps({"ok": True, "task": _parse_task(task)}))


def cmd_complete(args):
    service = _build_service()
    list_id = _resolve_list(service, args.list_id)

    task = service.tasks().get(tasklist=list_id, task=args.task_id).execute()
    task["status"] = "completed"

    updated = service.tasks().update(tasklist=list_id, task=args.task_id, body=task).execute()
    print(json.dumps({"ok": True, "task": _parse_task(updated)}))


def cmd_update(args):
    service = _build_service()
    list_id = _resolve_list(service, args.list_id)

    task = service.tasks().get(tasklist=list_id, task=args.task_id).execute()

    if args.title:
        task["title"] = args.title
    if args.notes is not None:
        task["notes"] = args.notes
    if args.due is not None:
        task["due"] = args.due

    updated = service.tasks().update(tasklist=list_id, task=args.task_id, body=task).execute()
    print(json.dumps({"ok": True, "task": _parse_task(updated)}))


def cmd_delete(args):
    service = _build_service()
    list_id = _resolve_list(service, args.list_id)
    service.tasks().delete(tasklist=list_id, task=args.task_id).execute()
    print(json.dumps({"ok": True, "deleted": args.task_id}))


def main():
    parser = argparse.ArgumentParser(description="Google Tasks operations")
    sub = parser.add_subparsers(dest="command", required=True)

    # list-tasklists
    sub.add_parser("list-tasklists", help="List all task lists")

    # list
    p_list = sub.add_parser("list", help="List tasks")
    p_list.add_argument("--list-id", default="@default", help="Task list ID (default: primary)")
    p_list.add_argument("--show-completed", action="store_true", help="Include completed tasks")

    # create
    p_create = sub.add_parser("create", help="Create a task")
    p_create.add_argument("--title", required=True, help="Task title")
    p_create.add_argument("--notes", default="", help="Task notes")
    p_create.add_argument("--due", default="", help="Due date (ISO 8601, e.g. 2026-03-31T00:00:00Z)")
    p_create.add_argument("--list-id", default="@default")

    # complete
    p_complete = sub.add_parser("complete", help="Mark a task as completed")
    p_complete.add_argument("--task-id", required=True)
    p_complete.add_argument("--list-id", default="@default")

    # update
    p_update = sub.add_parser("update", help="Update a task")
    p_update.add_argument("--task-id", required=True)
    p_update.add_argument("--title")
    p_update.add_argument("--notes")
    p_update.add_argument("--due")
    p_update.add_argument("--list-id", default="@default")

    # delete
    p_delete = sub.add_parser("delete", help="Delete a task")
    p_delete.add_argument("--task-id", required=True)
    p_delete.add_argument("--list-id", default="@default")

    args = parser.parse_args()

    try:
        dispatch = {
            "list-tasklists": cmd_list_tasklists,
            "list": cmd_list,
            "create": cmd_create,
            "complete": cmd_complete,
            "update": cmd_update,
            "delete": cmd_delete,
        }
        dispatch[args.command](args)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
