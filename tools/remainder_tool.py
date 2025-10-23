import json
from datetime import datetime

class ReminderTool:
    def __init__(self, storage_file="reminders.json"):
        self.storage_file = storage_file
        try:
            with open(self.storage_file) as f:
                self.reminders = json.load(f)
        except FileNotFoundError:
            self.reminders = []

    def add(self, message, time_str):
        reminder = {"message": message, "time": time_str}
        self.reminders.append(reminder)
        self._save()
        return f"Reminder added: {message} at {time_str}"

    def list(self):
        return self.reminders

    def delete(self, index):
        if 0 <= index < len(self.reminders):
            removed = self.reminders.pop(index)
            self._save()
            return f"Deleted reminder: {removed['message']}"
        return "Invalid index"

    def _save(self):
        with open(self.storage_file, "w") as f:
            json.dump(self.reminders, f, indent=2)
