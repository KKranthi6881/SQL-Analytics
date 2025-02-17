import sqlite3
from pathlib import Path
from datetime import datetime
import json
from threading import Lock

class ChatDatabase:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent.parent / "chat_history.db")
        
        self.db_path = db_path
        self._lock = Lock()
        self.init_db()

    def _get_connection(self):
        """Get a new database connection"""
        return sqlite3.connect(self.db_path)

    def init_db(self):
        """Initialize the database with conversations table"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        query TEXT,
                        output TEXT,
                        code_context TEXT,
                        technical_details TEXT
                    )
                """)
                conn.commit()

    def save_conversation(self, conversation_id: str, data: dict):
        """Save a conversation to the database"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO conversations 
                    (id, query, output, code_context, technical_details, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        data.get('query', ''),
                        json.dumps(data.get('output', {})),
                        json.dumps(data.get('code_context', {})),
                        json.dumps(data.get('technical_details', '')),
                        datetime.now()
                    )
                )
                conn.commit()

    def get_conversation(self, conversation_id: str):
        """Retrieve a conversation from the database"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM conversations WHERE id = ?",
                    (conversation_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return {
                        'id': row[0],
                        'created_at': row[1],
                        'query': row[2],
                        'output': json.loads(row[3]),
                        'code_context': json.loads(row[4]),
                        'technical_details': json.loads(row[5])
                    }
                return None

    def get_recent_conversations(self, limit: int = 10):
        """Get recent conversations"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM conversations ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                )
                rows = cursor.fetchall()
                
                return [{
                    'id': row[0],
                    'created_at': row[1],
                    'query': row[2],
                    'output': json.loads(row[3]),
                    'code_context': json.loads(row[4]),
                    'technical_details': json.loads(row[5])
                } for row in rows]

    def get_conversation_history(self):
        """Fetch all conversations formatted for history display"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id, created_at, query, output, code_context FROM conversations ORDER BY created_at DESC"
                )
                rows = cursor.fetchall()
                
                return [{
                    'id': row[0],
                    'timestamp': row[1],
                    'query': row[2],
                    'output': json.loads(row[3]),
                    'code_context': json.loads(row[4])
                } for row in rows]

    def get_conversation_history_with_checkpoints(self):
        """Fetch conversations with their checkpoints, organized by topic"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # First, let's get just the conversations with normalized timestamp
                query = """
                    SELECT 
                        id,
                        strftime('%Y-%m-%d %H:%M:%S', created_at) as created_at,  /* Simplified timestamp format */
                        query,
                        output,
                        code_context,
                        technical_details
                    FROM conversations 
                    ORDER BY created_at DESC
                """
                cursor.execute(query)
                rows = cursor.fetchall()
                
                # Group conversations by topic/context
                history = []
                for row in rows:
                    # Get checkpoints for this conversation using thread_id
                    checkpoint_query = """
                        SELECT 
                            checkpoint_id,
                            parent_checkpoint_id,
                            type,
                            checkpoint,
                            metadata
                        FROM checkpoints 
                        WHERE thread_id = ?
                        ORDER BY checkpoint_id
                    """
                    cursor.execute(checkpoint_query, (row[0],))  # using conversation id as thread_id
                    checkpoints = cursor.fetchall()
                    
                    # Process checkpoints if they exist
                    processed_checkpoints = []
                    if checkpoints:
                        for cp in checkpoints:
                            try:
                                # Safely handle binary data
                                checkpoint_data = {
                                    'checkpoint_id': cp[0],
                                    'parent_id': cp[1],
                                    'type': cp[2],
                                    'checkpoint': self._safe_load_binary(cp[3]),
                                    'metadata': self._safe_load_binary(cp[4])
                                }
                                processed_checkpoints.append(checkpoint_data)
                            except Exception as e:
                                print(f"Error processing checkpoint {cp[0]}: {str(e)}")
                                continue

                    history.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'query': row[2],
                        'output': json.loads(row[3]) if row[3] else {},
                        'code_context': json.loads(row[4]) if row[4] else {},
                        'technical_details': json.loads(row[5]) if row[5] else {},
                        'checkpoints': processed_checkpoints
                    })
                
                return history

    def _safe_load_binary(self, binary_data):
        """Safely load binary data that might be JSON"""
        if not binary_data:
            return {}
        try:
            # Try to decode as UTF-8 first
            if isinstance(binary_data, bytes):
                try:
                    return json.loads(binary_data.decode('utf-8'))
                except UnicodeDecodeError:
                    # If UTF-8 fails, try other encodings or return as string
                    return binary_data.decode('latin-1')
            # If it's already a string, try to parse as JSON
            elif isinstance(binary_data, str):
                return json.loads(binary_data)
            else:
                return str(binary_data)
        except Exception as e:
            print(f"Error loading binary data: {str(e)}")
            return {"error": "Could not parse data"} 