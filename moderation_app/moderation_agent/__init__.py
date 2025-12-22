"""Shared agent logic ('brain') for the moderation agent.

Clean Architecture layout:
- domain: entities + invariants + domain rules (no FastAPI)
- application: use-case services
- infrastructure: SQLite + file storage
- ml: classifier interface + implementations
- runners: Sense→Think→Act→Learn ticks
"""
