# halos/storage/offline_queue.py
import aiosqlite
from queue import Queue
import json

class OfflineQueue:
    def __init__(self, db_path='messages.db'):
        self.db_path = db_path
        self.in_memory_queue = Queue()
        
    async def _init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS offline_messages (
                    id INTEGER PRIMARY KEY,
                    room_id TEXT,
                    content BLOB,
                    timestamp FLOAT
                )
            ''')
            await db.commit()

    async def enqueue(self, room_id: str, content: bytes):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                'INSERT INTO offline_messages (room_id, content, timestamp) VALUES (?, ?, ?)',
                (room_id, content, time.time())
            )
            await db.commit()
        self.in_memory_queue.put((room_id, content))

    async def flush(self, messenger):
        """Send all queued messages when back online"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute('SELECT * FROM offline_messages ORDER BY timestamp ASC')
            rows = await cursor.fetchall()
            
            for row in rows:
                room_id, content, _ = row
                try:
                    await messenger.send_message(room_id, content)
                    await db.execute('DELETE FROM offline_messages WHERE id=?', (row[0],))
                except Exception as e:
                    print(f"Failed to send offline message: {e}")
                    break
            
            await db.commit()