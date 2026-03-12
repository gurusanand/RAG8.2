"""
Feedback Service — Collects and manages user feedback on RAG responses.
"""
import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict


@dataclass
class FeedbackEntry:
    """A single feedback entry from a user."""
    query: str
    response_preview: str
    rating: int  # 1-5 stars
    comment: str = ""
    timestamp: float = 0.0
    user_id: str = "anonymous"
    session_id: str = ""
    confidence: float = 0.0
    validation_status: str = ""

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class FeedbackService:
    """Manages feedback collection, storage, and retrieval."""

    def __init__(self, settings=None):
        if settings is None:
            from config.settings import get_settings
            settings = get_settings()
        self.settings = settings
        self.feedback_file = os.path.join(
            settings.paths.base_dir,
            settings.feedback.feedback_file
        )
        os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)

    def submit_feedback(self, entry: FeedbackEntry) -> bool:
        """Submit a feedback entry."""
        try:
            existing = self._load_all()
            existing.append(asdict(entry))
            with open(self.feedback_file, "w") as f:
                json.dump(existing, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"[FEEDBACK] Error saving feedback: {e}")
            return False

    def get_all_feedback(self) -> List[Dict]:
        """Get all feedback entries."""
        return self._load_all()

    def get_feedback_stats(self) -> Dict:
        """Get feedback statistics."""
        entries = self._load_all()
        if not entries:
            return {"total": 0, "avg_rating": 0.0, "positive": 0, "negative": 0, "neutral": 0, "satisfaction_rate": "N/A"}

        ratings = [e.get("rating", 3) for e in entries]
        positive = sum(1 for r in ratings if r >= 4)
        return {
            "total": len(entries),
            "avg_rating": sum(ratings) / len(ratings),
            "positive": positive,
            "negative": sum(1 for r in ratings if r <= 2),
            "neutral": sum(1 for r in ratings if r == 3),
            "satisfaction_rate": f"{(positive / len(ratings) * 100):.0f}%",
        }

    def get_stats(self) -> Dict:
        """Alias for get_feedback_stats."""
        return self.get_feedback_stats()

    def _load_all(self) -> List[Dict]:
        """Load all feedback from file."""
        if not os.path.exists(self.feedback_file):
            return []
        try:
            with open(self.feedback_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return []
