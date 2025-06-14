import sqlite3

def init_db():
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            session_id TEXT,
            question TEXT,
            answer TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_chat(session_id, question, answer):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history VALUES (?, ?, ?)", (session_id, question, answer))
    conn.commit()
    conn.close()

def load_chat(session_id):
    conn = sqlite3.connect("chat_history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM chat_history WHERE session_id = ?", (session_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows
