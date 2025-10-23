class ToDoListTool:
    def __init__(self, storage_file="todo.json"):
        self.storage_file = storage_file
        try:
            with open(self.storage_file) as f:
                self.todos = json.load(f)
        except FileNotFoundError:
            self.todos = []

    def add(self, task):
        self.todos.append(task)
        self._save()
        return f"Task added: {task}"

    def list(self):
        return self.todos

    def remove(self, index):
        if 0 <= index < len(self.todos):
            task = self.todos.pop(index)
            self._save()
            return f"Removed task: {task}"
        return "Invalid index"

    def _save(self):
        with open(self.storage_file, "w") as f:
            json.dump(self.todos, f, indent=2)
