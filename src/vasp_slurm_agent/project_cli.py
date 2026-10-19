"""Explicit project commands; existing path-based commands stay independent."""

import json
from pathlib import PurePath


def add_parser(commands):
    parser = commands.add_parser("project", help="Manage local projects")
    actions = parser.add_subparsers(dest="project_command", required=True)
    create = actions.add_parser("create", help="Create a project")
    create.add_argument("name")
    listing = actions.add_parser("list", help="List projects")
    listing.add_argument("--include-archived", action="store_true")
    listing.add_argument("--include-deleted", action="store_true")
    for name, help_text in {
        "show": "Show a project and its folders",
        "open": "Open a project",
        "rename": "Rename a project",
        "archive": "Archive or unarchive a project",
        "restore": "Restore a deleted project",
        "delete": "Mark a project deleted; keep its files",
        "sessions": "List project conversations",
        "runs": "List project runs",
    }.items():
        action = actions.add_parser(name, help=help_text)
        action.add_argument("project_id", metavar="PROJECT_ID")
        if name == "rename":
            action.add_argument("name")
        elif name == "archive":
            action.add_argument("--unarchive", action="store_true")
        elif name == "delete":
            action.add_argument("--confirm-name", required=True, help="The project's current name")
    return parser


def _json_path(value):
    if isinstance(value, PurePath):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def dispatch(args):
    from .workspace import Workspace

    workspace = Workspace()
    action = args.project_command
    if action == "create":
        result = workspace.create_project(args.name)
    elif action == "list":
        result = workspace.list_projects(include_archived=args.include_archived,
                                         include_deleted=args.include_deleted)
    elif action in {"show", "open"}:
        project = (workspace.open_project if action == "open" else workspace.get_project)(args.project_id)
        result = {"project": project, "paths": workspace.project_paths(args.project_id)}
    elif action == "rename":
        result = workspace.rename_project(args.project_id, args.name)
    elif action == "archive":
        result = workspace.archive_project(args.project_id, archived=not args.unarchive)
    elif action == "restore":
        result = workspace.restore_project(args.project_id)
    elif action == "delete":
        result = workspace.delete_project(args.project_id, confirmation=args.confirm_name)
    elif action == "sessions":
        result = workspace.list_sessions(args.project_id)
    elif action == "runs":
        result = workspace.list_runs(args.project_id)
    else:
        raise ValueError("Unknown project command.")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_path))
    return 0
