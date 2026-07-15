# agent.py
# Rule-based behavioral agent
# Developer: Hamna Munir | Supervisor: Sir Nadeem

import time

class ClassroomAgent:
    def __init__(self):
        self.student_history  = {}   # {sid: [scores]}
        self.flagged_students = {}   # {sid: flag_count}
        self.class_history    = []   # [avg_scores]
        self.alerts           = []   # all alerts
        self.session_start    = time.time()

    def update(self, students):
        """
        Call this after every frame analysis.
        students = list of dicts from face_tracker
        """
        new_alerts = []

        # ── Per student rules
        for s in students:
            sid   = s["id"]
            score = s["engagement"]

            # Update history
            if sid not in self.student_history:
                self.student_history[sid]  = []
                self.flagged_students[sid] = 0
            self.student_history[sid].append(score)

            # Keep last 5 readings only
            if len(self.student_history[sid]) > 5:
                self.student_history[sid].pop(0)

            recent_avg = sum(self.student_history[sid]) / len(self.student_history[sid])

            # Rule 1 — Low engagement for sustained period
            if recent_avg < 35 and len(self.student_history[sid]) >= 3:
                self.flagged_students[sid] += 1
                alert = {
                    "type":     "LOW_ENGAGEMENT",
                    "sid":      sid,
                    "score":    score,
                    "avg":      round(recent_avg),
                    "msg":      f"{sid} has sustained low engagement ({round(recent_avg)}% avg)",
                    "severity": "high",
                    "time":     time.strftime("%H:%M:%S"),
                }
                new_alerts.append(alert)
                self.alerts.append(alert)

            # Rule 2 — Repeated distraction pattern
            if self.flagged_students[sid] >= 3:
                alert = {
                    "type":     "PATTERN_DETECTED",
                    "sid":      sid,
                    "score":    score,
                    "avg":      round(recent_avg),
                    "msg":      f"{sid} flagged {self.flagged_students[sid]}x — consistent distraction pattern",
                    "severity": "critical",
                    "time":     time.strftime("%H:%M:%S"),
                }
                new_alerts.append(alert)
                self.alerts.append(alert)

            # Rule 3 — Not looking forward
            if not s.get("looking_forward", True) and score < 40:
                alert = {
                    "type":     "NOT_ATTENDING",
                    "sid":      sid,
                    "score":    score,
                    "avg":      round(recent_avg),
                    "msg":      f"{sid} not looking at board (score: {score}%)",
                    "severity": "medium",
                    "time":     time.strftime("%H:%M:%S"),
                }
                new_alerts.append(alert)
                self.alerts.append(alert)

        # ── Class-level rules
        if students:
            class_avg = sum(s["engagement"] for s in students) / len(students)
            self.class_history.append(class_avg)
            if len(self.class_history) > 10:
                self.class_history.pop(0)

            # Rule 4 — Class avg dropped
            if len(self.class_history) >= 3:
                recent_class = sum(self.class_history[-3:]) / 3
                if recent_class < 45:
                    alert = {
                        "type":     "CLASS_ATTENTION_DROP",
                        "sid":      "CLASS",
                        "score":    round(recent_class),
                        "avg":      round(recent_class),
                        "msg":      f"Class attention dropped to {round(recent_class)}% — consider activity change",
                        "severity": "high",
                        "time":     time.strftime("%H:%M:%S"),
                    }
                    new_alerts.append(alert)
                    self.alerts.append(alert)

            # Rule 5 — Majority distracted
            distracted = sum(1 for s in students if s["engagement"] < 35)
            if distracted > len(students) * 0.5:
                alert = {
                    "type":     "MAJORITY_DISTRACTED",
                    "sid":      "CLASS",
                    "score":    round(class_avg),
                    "avg":      round(class_avg),
                    "msg":      f"{distracted}/{len(students)} students distracted — immediate attention needed",
                    "severity": "critical",
                    "time":     time.strftime("%H:%M:%S"),
                }
                new_alerts.append(alert)
                self.alerts.append(alert)

        return new_alerts

    def get_student_status(self, sid):
        """Get current status of a student"""
        if sid not in self.student_history:
            return "Unknown"
        avg = sum(self.student_history[sid]) / len(self.student_history[sid])
        flags = self.flagged_students.get(sid, 0)
        if flags >= 3:   return "⚠️ Chronic"
        elif avg < 35:   return "🔴 Distracted"
        elif avg < 60:   return "🟡 Moderate"
        else:            return "🟢 Focused"

    def get_summary(self):
        """Session summary"""
        return {
            "total_alerts":    len(self.alerts),
            "flagged_count":   sum(1 for v in self.flagged_students.values() if v > 0),
            "chronic_count":   sum(1 for v in self.flagged_students.values() if v >= 3),
            "class_avg":       round(sum(self.class_history)/len(self.class_history)) if self.class_history else 0,
            "session_min":     round((time.time()-self.session_start)/60, 1),
            "alerts":          self.alerts[-20:],
        }

    def clear(self):
        """Reset for new session"""
        self.__init__()