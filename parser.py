import json
import re
from pathlib import Path
import pandas as pd

class Parser:
    def __init__(self, json_path):
        self.json_path = Path(json_path)
        self.df = None
        self.parsed_df = None
        
        self.rating_lbl = {
            "coherence": "پیوستگی",
            "knowledge": "دانش عمومی",
            "teaching": "انتقال",
            "management": "مدیریت",
            "responsiveness": "پاسخگویی",
            "behavior": "آداب و رفتار"
        }
        
        self.columns = [
            "message_id", "date", "term", "professor_id", "professor_name_raw",
            "department", "course_name"
        ] + list(self.rating_lbl.keys()) + [
            "grading_status_raw", "attendance_status_raw", "comment_text"
        ]
    
    def __load_json(self):
        raw_text = self.json_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
        messages = data["messages"] if isinstance(data, dict) else data
        return messages
    
    def __build_base_df(self, messages):
        rows = []

        for msg in messages:
            if isinstance(msg.get("text"), list):
                full_text = "".join(
                    t if isinstance(t, str) else t.get("text", "")
                    for t in msg["text"]
                )
            else:
                full_text = msg.get("text", "")
            rows.append({
                "message_id": msg.get("id"),
                "date": msg.get("date"),
                "full_text": full_text
            })

        self.df = pd.DataFrame(rows)
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
    
    def __parse_messages(self):
        parsed = []
        
        for _, row in self.df.iterrows():
            text = row["full_text"]
            parse_error = []

            # Check date validity
            if pd.isna(row["date"]):
                parse_error.append(self.columns.index("date"))

            # Term
            m = re.search(r"ترم[^\n]*\n?\s*┘\s*([^\n\r]+)", text)
            term = m.group(1).strip() if m else None
            if not term:
                parse_error.append(self.columns.index("term"))

            # Professor name
            m = re.search(r"(?:🧑‍🏫\s*)?([^\n\r]+)\n?\s*🏫", text)
            professor_name_raw = m.group(1).strip() if m else None
            if not professor_name_raw:
                parse_error.append(self.columns.index("professor_name_raw"))

            # Department
            m = re.search(r"#([^\s#]+)", text)
            department = m.group(1).strip() if m else None
            if not department:
                parse_error.append(self.columns.index("department"))

            # Course name (بدون تغییر)
            m = re.search(r"📒\s*(.+)", text)
            course_name = m.group(1).strip() if m else None
            if not course_name:
                parse_error.append(self.columns.index("course_name"))

            # Attendance status
            m = re.search(r"حضور و غیاب\s*┘\s*(.+)", text)
            attendance_status_raw = m.group(1).strip() if m else None
            if not attendance_status_raw:
                parse_error.append(self.columns.index("attendance_status_raw"))

            # Grading status
            m = re.search(r"وضعیت نمره دادن:\s*┘\s*(.+)", text)
            grading_status_raw = m.group(1).strip() if m else None
            if not grading_status_raw:
                parse_error.append(self.columns.index("grading_status_raw"))

            # Comment text
            m = re.search(r"توضیحات:\s*┘\s*(.+?)\n~+", text, re.S)
            comment_text = m.group(1).strip() if m else None
            if not comment_text:
                parse_error.append(self.columns.index("comment_text"))

            # Ratings
            ratings = {}
            for col, label in self.rating_lbl.items():
                m = re.search(fr"{label}[^\d]*(\d{{1,2}})", text)
                if m:
                    ratings[col] = min(max(int(m.group(1)), 0), 10)
                else:
                    ratings[col] = None
                    parse_error.append(self.columns.index(col))

            parsed.append({
                "message_id": row["message_id"],
                "date": row["date"],
                "term": term,
                "professor_id": None,
                "professor_name_raw": professor_name_raw,
                "department": department,
                "course_name": course_name,
                **ratings,
                "grading_status_raw": grading_status_raw,
                "attendance_status_raw": attendance_status_raw,
                "comment_text": comment_text,
                "parse_error": parse_error if parse_error else None
            })

        self.parsed_df = pd.DataFrame(parsed)
    
    # Convenience method
    def run(self):
        messages = self.__load_json()
        self.__build_base_df(messages)
        self.__parse_messages()
        return self.parsed_df
    
    def save_csv(self, path="parsed_professor_reviews.csv"):
        self.parsed_df.to_csv(path, index=False)
        print(f"Saved parsed data to {path}")

if __name__ == "__main__":
    parser = Parser("result.json")
    df = parser.run()
    parser.save_csv("reviews.csv")