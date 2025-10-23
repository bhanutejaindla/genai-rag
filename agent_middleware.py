class TrimMiddleware:
    def __call__(self, message: str) -> str:
        return message.strip()

class SummarizeMiddleware:
    def __call__(self, message: str) -> str:
        if len(message) > 500:
            return message[:500] + "..."  # simple truncate
        return message
