#!/usr/bin/env python3
"""Comprehensive performance evaluation script for GenAI Companion with ACE.

This script evaluates:
1. Database health (SQLite conversation history, Chroma vector store)
2. Playbook performance metrics and evolution
3. Agent success metrics from conversation logs
"""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from genai_companion_with_ace.config import CompanionConfig


@dataclass
class DatabaseStats:
    """Statistics about database health."""

    total_sessions: int = 0
    total_messages: int = 0
    avg_messages_per_session: float = 0.0
    sessions_by_mode: dict[str, int] = None
    sessions_by_course: dict[str, int] = None
    oldest_session: str | None = None
    newest_session: str | None = None

    def __post_init__(self) -> None:
        if self.sessions_by_mode is None:
            self.sessions_by_mode = {}
        if self.sessions_by_course is None:
            self.sessions_by_course = {}


@dataclass
class PlaybookMetrics:
    """Performance metrics from playbook."""

    version: str
    iteration: int
    accuracy: float
    bleu_score: float
    rouge_score: float
    exact_match: float
    semantic_similarity: float
    avg_tokens: float
    inference_time_sec: float
    vram_usage_gb: float
    convergence_status: str
    heuristics_count: int
    examples_count: int
    history_entries: int


@dataclass
class ConversationMetrics:
    """Metrics from conversation logs."""

    total_turns: int
    unique_sessions: int
    avg_answer_length: float
    avg_question_length: float
    sessions_with_sources: int
    total_sources_retrieved: int
    avg_sources_per_turn: float
    modes_used: dict[str, int]
    courses_covered: dict[str, int]
    timestamp_range: tuple[str, str] | None


@dataclass
class VectorStoreStats:
    """Statistics about the vector store."""

    collection_exists: bool
    document_count: int | None = None
    collection_name: str | None = None


class PerformanceEvaluator:
    """Comprehensive performance evaluator."""

    def __init__(
        self,
        history_db_path: Path,
        conversation_logs_dir: Path,
        playbook_path: Path,
        chroma_path: Path,
        chroma_collection: str = "ibm_genai_companion",
    ) -> None:
        self.history_db_path = history_db_path
        self.conversation_logs_dir = conversation_logs_dir
        self.playbook_path = playbook_path
        self.chroma_path = chroma_path
        self.chroma_collection = chroma_collection

    def evaluate_all(self) -> dict[str, Any]:
        """Run all evaluations and return comprehensive results."""
        results: dict[str, Any] = {}

        # 1. Database evaluation
        print("[1/4] Evaluating SQLite conversation history database...")
        try:
            results["sqlite_db"] = self.evaluate_sqlite_db()
        except Exception as e:
            results["sqlite_db"] = {"error": str(e)}

        # 2. Vector store evaluation
        print("[2/4] Evaluating Chroma vector store...")
        try:
            results["vector_store"] = self.evaluate_vector_store()
        except Exception as e:
            results["vector_store"] = {"error": str(e)}

        # 3. Playbook evaluation
        print("[3/4] Evaluating ACE playbook performance...")
        try:
            results["playbook"] = self.evaluate_playbook()
        except Exception as e:
            results["playbook"] = {"error": str(e)}

        # 4. Conversation logs evaluation
        print("[4/4] Evaluating conversation logs for agent success...")
        try:
            results["conversations"] = self.evaluate_conversations()
        except Exception as e:
            results["conversations"] = {"error": str(e)}

        # 5. Overall assessment
        results["overall"] = self._generate_overall_assessment(results)

        return results

    def evaluate_sqlite_db(self) -> dict[str, Any]:
        """Evaluate SQLite conversation history database."""
        if not self.history_db_path.exists():
            return {"status": "not_found", "path": str(self.history_db_path)}

        stats = DatabaseStats()

        with sqlite3.connect(self.history_db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get session count
            cursor = conn.execute("SELECT COUNT(*) as count FROM sessions")
            stats.total_sessions = cursor.fetchone()["count"]

            # Get message count
            cursor = conn.execute("SELECT COUNT(*) as count FROM messages")
            stats.total_messages = cursor.fetchone()["count"]

            if stats.total_sessions > 0:
                stats.avg_messages_per_session = stats.total_messages / stats.total_sessions

            # Sessions by mode
            cursor = conn.execute("SELECT mode, COUNT(*) as count FROM sessions WHERE mode IS NOT NULL GROUP BY mode")
            for row in cursor:
                stats.sessions_by_mode[row["mode"]] = row["count"]

            # Sessions by course
            cursor = conn.execute(
                "SELECT course, COUNT(*) as count FROM sessions WHERE course IS NOT NULL GROUP BY course"
            )
            for row in cursor:
                stats.sessions_by_course[row["course"]] = row["count"]

            # Timestamp range
            cursor = conn.execute("SELECT MIN(created_at) as oldest, MAX(created_at) as newest FROM sessions")
            row = cursor.fetchone()
            if row["oldest"]:
                stats.oldest_session = row["oldest"]
                stats.newest_session = row["newest"]

            # Get sample of recent sessions
            cursor = conn.execute(
                "SELECT session_id, created_at, mode, course FROM sessions ORDER BY created_at DESC LIMIT 5"
            )
            recent_sessions = [dict(row) for row in cursor]

        return {
            "status": "healthy",
            "path": str(self.history_db_path),
            "total_sessions": stats.total_sessions,
            "total_messages": stats.total_messages,
            "avg_messages_per_session": round(stats.avg_messages_per_session, 2),
            "sessions_by_mode": stats.sessions_by_mode,
            "sessions_by_course": stats.sessions_by_course,
            "oldest_session": stats.oldest_session,
            "newest_session": stats.newest_session,
            "recent_sessions": recent_sessions,
        }

    def evaluate_vector_store(self) -> dict[str, Any]:
        """Evaluate Chroma vector store."""
        stats = VectorStoreStats(collection_exists=False)

        if not self.chroma_path.exists():
            return {
                "status": "not_found",
                "path": str(self.chroma_path),
                "collection_exists": False,
            }

        try:
            import chromadb
        except ImportError:
            return {
                "status": "chromadb_not_available",
                "path": str(self.chroma_path),
                "error": "chromadb package not installed",
            }

        try:
            # Try to initialize client - this may fail if DB is corrupted
            try:
                client = chromadb.PersistentClient(path=str(self.chroma_path))
                collections = client.list_collections()
            except (Exception, BaseException) as e:  # Catch all exceptions including panics
                error_msg = str(e)
                error_type = type(e).__name__
                # Handle ChromaDB corruption or version issues
                if (
                    "panic" in error_msg.lower()
                    or "out of range" in error_msg.lower()
                    or "PanicException" in error_type
                ):
                    return {
                        "status": "corrupted_or_incompatible",
                        "path": str(self.chroma_path),
                        "error": f"ChromaDB database may be corrupted or incompatible: {error_type}: {error_msg[:200]}",
                        "recommendation": "Consider reinitializing the vector store or checking ChromaDB version compatibility",
                    }
                return {
                    "status": "error",
                    "path": str(self.chroma_path),
                    "error": f"{error_type}: {error_msg}",
                }

            collection_names = [c.name for c in collections]
            stats.collection_exists = self.chroma_collection in collection_names

            if stats.collection_exists:
                collection = client.get_collection(self.chroma_collection)
                stats.collection_name = self.chroma_collection
                stats.document_count = collection.count()

                # Get sample metadata
                sample_results = collection.peek(limit=5)
                sample_metadata = []
                if sample_results.get("metadatas"):
                    for meta in sample_results["metadatas"][:3]:
                        sample_metadata.append(meta)

                return {
                    "status": "healthy",
                    "path": str(self.chroma_path),
                    "collection_exists": True,
                    "collection_name": stats.collection_name,
                    "document_count": stats.document_count,
                    "available_collections": collection_names,
                    "sample_metadata": sample_metadata,
                }
            else:
                return {
                    "status": "collection_not_found",
                    "path": str(self.chroma_path),
                    "collection_exists": False,
                    "collection_name": self.chroma_collection,
                    "available_collections": collection_names,
                }
        except (Exception, BaseException) as e:  # Catch any remaining exceptions
            error_msg = str(e)
            error_type = type(e).__name__
            return {
                "status": "error",
                "path": str(self.chroma_path),
                "error": f"{error_type}: {error_msg[:300]}",
            }

    def evaluate_playbook(self) -> dict[str, Any]:
        """Evaluate ACE playbook performance metrics."""
        if not self.playbook_path.exists():
            return {"status": "not_found", "path": str(self.playbook_path)}

        try:
            with self.playbook_path.open(encoding="utf-8") as f:
                playbook_data = yaml.safe_load(f)

            metadata = playbook_data.get("metadata", {})
            perf_metrics = metadata.get("performance_metrics", {})
            context = playbook_data.get("context", {})
            heuristics = context.get("heuristics", [])
            examples = context.get("few_shot_examples", [])
            history = playbook_data.get("history", [])

            metrics = PlaybookMetrics(
                version=playbook_data.get("version", "unknown"),
                iteration=metadata.get("iteration", 0),
                accuracy=perf_metrics.get("accuracy", 0.0),
                bleu_score=perf_metrics.get("bleu_score", 0.0),
                rouge_score=perf_metrics.get("rouge_score", 0.0),
                exact_match=perf_metrics.get("exact_match", 0.0),
                semantic_similarity=perf_metrics.get("semantic_similarity", 0.0),
                avg_tokens=perf_metrics.get("avg_tokens", 0.0),
                inference_time_sec=perf_metrics.get("inference_time_sec", 0.0),
                vram_usage_gb=perf_metrics.get("vram_usage_gb", 0.0),
                convergence_status=metadata.get("convergence_status", "unknown"),
                heuristics_count=len(heuristics),
                examples_count=len(examples),
                history_entries=len(history),
            )

            # Analyze history for patterns
            history_analysis = self._analyze_playbook_history(history)

            # Analyze heuristic usage from conversation logs
            heuristic_usage = self._analyze_heuristic_usage(heuristics)

            return {
                "status": "loaded",
                "path": str(self.playbook_path),
                "version": metrics.version,
                "iteration": metrics.iteration,
                "performance_metrics": {
                    "accuracy": metrics.accuracy,
                    "bleu_score": round(metrics.bleu_score, 4),
                    "rouge_score": round(metrics.rouge_score, 4),
                    "exact_match": metrics.exact_match,
                    "semantic_similarity": round(metrics.semantic_similarity, 4),
                    "avg_tokens": metrics.avg_tokens,
                    "inference_time_sec": round(metrics.inference_time_sec, 2),
                    "vram_usage_gb": metrics.vram_usage_gb,
                },
                "convergence_status": metrics.convergence_status,
                "content_stats": {
                    "heuristics_count": metrics.heuristics_count,
                    "examples_count": metrics.examples_count,
                    "history_entries": metrics.history_entries,
                },
                "history_analysis": history_analysis,
                "heuristic_usage": heuristic_usage,
                "heuristics": [
                    {
                        "id": h.get("id"),
                        "rule": h.get("rule", "")[:100] + "..." if len(h.get("rule", "")) > 100 else h.get("rule", ""),
                        "confidence": h.get("confidence", 0.0),
                        "usage_count": h.get("usage_count", 0),
                        "success_rate": h.get("success_rate", 0.0),
                    }
                    for h in heuristics[:5]  # Show first 5
                ],
            }
        except Exception as e:
            return {"status": "error", "path": str(self.playbook_path), "error": str(e)}

    def _analyze_playbook_history(self, history: list[dict]) -> dict[str, Any]:
        """Analyze playbook history for patterns."""
        if not history:
            return {"total_changes": 0}

        change_types = Counter(item.get("change_type", "unknown") for item in history)
        iterations = Counter(item.get("iteration", 0) for item in history)

        return {
            "total_changes": len(history),
            "change_types": dict(change_types),
            "changes_by_iteration": dict(iterations),
            "most_common_change": change_types.most_common(1)[0] if change_types else None,
        }

    def _analyze_heuristic_usage(self, heuristics: list[dict]) -> dict[str, Any]:
        """Analyze heuristic usage from conversation logs."""
        if not self.conversation_logs_dir.exists():
            return {"status": "no_logs", "message": "Conversation logs not available"}
        heuristic_map = {h.get("id", ""): h for h in heuristics}
        usage_counts, total_turns_with_heuristics = self._collect_heuristic_usage()
        heuristic_stats = self._build_heuristic_stats(heuristic_map, usage_counts, total_turns_with_heuristics)
        return {
            "total_turns_analyzed": total_turns_with_heuristics,
            "heuristic_stats": heuristic_stats,
            "total_heuristics": len(heuristics),
            "heuristics_with_usage": len([h for h in heuristic_stats if h["usage_count"] > 0]),
        }

    def _collect_heuristic_usage(self) -> tuple[dict[str, int], int]:
        usage_counts: dict[str, int] = {}
        total_turns = 0
        for jsonl_file in self._conversation_log_files():
            total_turns += self._update_usage_from_file(jsonl_file, usage_counts)
        return usage_counts, total_turns

    def _conversation_log_files(self) -> list[Path]:
        return [path for path in self.conversation_logs_dir.glob("*.jsonl") if path.name != "history.db"]

    def _update_usage_from_file(self, jsonl_file: Path, usage_counts: dict[str, int]) -> int:
        turns_with_heuristics = 0
        try:
            with jsonl_file.open(encoding="utf-8") as handle:
                for line in handle:
                    record = self._safe_json_load(line)
                    if not record:
                        continue
                    metadata = record.get("metadata", {})
                    heuristic_ids_str = metadata.get("heuristic_ids", "")
                    if not heuristic_ids_str:
                        continue
                    turns_with_heuristics += 1
                    for h_id in heuristic_ids_str.split(","):
                        identifier = h_id.strip()
                        if identifier:
                            usage_counts[identifier] = usage_counts.get(identifier, 0) + 1
        except Exception:
            return 0
        return turns_with_heuristics

    @staticmethod
    def _safe_json_load(line: str) -> dict[str, Any] | None:
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _build_heuristic_stats(
        heuristic_map: dict[str, dict],
        usage_counts: dict[str, int],
        total_turns_with_heuristics: int,
    ) -> list[dict[str, Any]]:
        stats: list[dict[str, Any]] = []
        for h_id, h_data in heuristic_map.items():
            usage_count = usage_counts.get(h_id, 0)
            stats.append({
                "id": h_id,
                "usage_count": usage_count,
                "usage_percentage": round(
                    (usage_count / total_turns_with_heuristics * 100) if total_turns_with_heuristics > 0 else 0,
                    1,
                ),
                "confidence": h_data.get("confidence", 0.0),
                "success_rate": h_data.get("success_rate", 0.0),
            })
        return stats

    def evaluate_conversations(self) -> dict[str, Any]:
        """Evaluate conversation logs for agent success metrics."""
        if not self.conversation_logs_dir.exists():
            return {"status": "not_found", "path": str(self.conversation_logs_dir)}

        metrics = ConversationMetrics(
            total_turns=0,
            unique_sessions=0,
            avg_answer_length=0.0,
            avg_question_length=0.0,
            sessions_with_sources=0,
            total_sources_retrieved=0,
            avg_sources_per_turn=0.0,
            modes_used={},
            courses_covered={},
            timestamp_range=None,
        )

        session_ids = set()
        answer_lengths = []
        question_lengths = []
        timestamps = []
        sessions_with_sources_set = set()

        jsonl_files = self._conversation_log_files()
        if not jsonl_files:
            return {"status": "no_logs", "path": str(self.conversation_logs_dir)}

        for jsonl_file in jsonl_files:
            self._process_conversation_file(
                jsonl_file,
                metrics,
                session_ids,
                question_lengths,
                answer_lengths,
                sessions_with_sources_set,
                timestamps,
            )

        metrics.unique_sessions = len(session_ids)
        metrics.sessions_with_sources = len(sessions_with_sources_set)

        if answer_lengths:
            metrics.avg_answer_length = statistics.mean(answer_lengths)
        if question_lengths:
            metrics.avg_question_length = statistics.mean(question_lengths)
        if metrics.total_turns > 0:
            metrics.avg_sources_per_turn = metrics.total_sources_retrieved / metrics.total_turns

        if timestamps:
            metrics.timestamp_range = (min(timestamps), max(timestamps))

        # Calculate success indicators
        source_coverage_rate = (
            metrics.sessions_with_sources / metrics.unique_sessions if metrics.unique_sessions > 0 else 0.0
        )

        return self._build_conversation_summary(metrics, source_coverage_rate, jsonl_files)

    def _process_conversation_file(
        self,
        jsonl_file: Path,
        metrics: ConversationMetrics,
        session_ids: set[str],
        question_lengths: list[int],
        answer_lengths: list[int],
        sessions_with_sources_set: set[str],
        timestamps: list[str],
    ) -> None:
        try:
            with jsonl_file.open(encoding="utf-8") as handle:
                for line in handle:
                    record = self._safe_json_load(line)
                    if record is None:
                        continue
                    session_ids.add(record.get("session_id", "unknown"))
                    metrics.total_turns += 1
                    question_lengths.append(len(record.get("question", "")))
                    answer_lengths.append(len(record.get("answer", "")))
                    sources = record.get("retrieved_sources", [])
                    if sources:
                        sessions_with_sources_set.add(record.get("session_id"))
                        metrics.total_sources_retrieved += len(sources)
                    metadata = record.get("metadata", {})
                    mode = metadata.get("mode", "unknown")
                    course = metadata.get("course", "unknown")
                    metrics.modes_used[mode] = metrics.modes_used.get(mode, 0) + 1
                    metrics.courses_covered[course] = metrics.courses_covered.get(course, 0) + 1
                    timestamp = record.get("timestamp")
                    if timestamp:
                        timestamps.append(timestamp)
        except Exception as exc:
            print(f"Warning: Error reading {jsonl_file}: {exc}")

    def _build_conversation_summary(
        self,
        metrics: ConversationMetrics,
        source_coverage_rate: float,
        jsonl_files: list[Path],
    ) -> dict[str, Any]:
        avg_turns = round(metrics.total_turns / metrics.unique_sessions, 2) if metrics.unique_sessions > 0 else 0.0
        answer_to_question_ratio = (
            round(metrics.avg_answer_length / metrics.avg_question_length, 2)
            if metrics.avg_question_length > 0
            else 0.0
        )
        return {
            "status": "analyzed",
            "path": str(self.conversation_logs_dir),
            "total_turns": metrics.total_turns,
            "unique_sessions": metrics.unique_sessions,
            "avg_turns_per_session": avg_turns,
            "answer_quality": {
                "avg_answer_length": round(metrics.avg_answer_length, 0),
                "avg_question_length": round(metrics.avg_question_length, 0),
                "answer_to_question_ratio": answer_to_question_ratio,
            },
            "retrieval_quality": {
                "sessions_with_sources": metrics.sessions_with_sources,
                "source_coverage_rate": round(source_coverage_rate, 3),
                "total_sources_retrieved": metrics.total_sources_retrieved,
                "avg_sources_per_turn": round(metrics.avg_sources_per_turn, 2),
            },
            "usage_patterns": {
                "modes_used": metrics.modes_used,
                "courses_covered": metrics.courses_covered,
            },
            "timestamp_range": metrics.timestamp_range,
            "log_files_analyzed": len(jsonl_files),
        }

    def _generate_overall_assessment(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate overall assessment from all evaluations."""
        assessment = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "health_status": self._overall_health(results),
            "recommendations": self._build_recommendations(results),
            "summary": {},
        }
        assessment["summary"] = {
            "total_sessions": results.get("sqlite_db", {}).get("total_sessions", 0),
            "total_conversation_turns": results.get("conversations", {}).get("total_turns", 0),
            "vector_store_documents": results.get("vector_store", {}).get("document_count", 0),
            "playbook_version": results.get("playbook", {}).get("version", "unknown"),
            "playbook_iteration": results.get("playbook", {}).get("iteration", 0),
        }

        return assessment

    @staticmethod
    def _overall_health(results: dict[str, Any]) -> str:
        statuses = [
            results.get("sqlite_db", {}).get("status", "unknown"),
            results.get("vector_store", {}).get("status", "unknown"),
            results.get("playbook", {}).get("status", "unknown"),
            results.get("conversations", {}).get("status", "unknown"),
        ]
        if all(status in ("healthy", "loaded", "analyzed") for status in statuses):
            return "healthy"
        if any(status == "error" for status in statuses):
            return "degraded"
        return "partial"

    def _build_recommendations(self, results: dict[str, Any]) -> list[str]:
        recommendations: list[str] = []
        sqlite_status = results.get("sqlite_db", {}).get("status", "unknown")
        vector_status = results.get("vector_store", {}).get("status", "unknown")
        playbook_status = results.get("playbook", {}).get("status", "unknown")
        conv_status = results.get("conversations", {}).get("status", "unknown")

        if sqlite_status == "not_found":
            recommendations.append("SQLite database not found - no conversation history stored")
        elif sqlite_status == "healthy" and results.get("sqlite_db", {}).get("total_sessions", 0) == 0:
            recommendations.append("No conversation sessions found in database")

        if vector_status == "not_found":
            recommendations.append("Chroma vector store not found - RAG may not be functional")
        elif vector_status == "collection_not_found":
            recommendations.append(
                f"Chroma collection '{self.chroma_collection}' not found - documents may need to be ingested"
            )

        if playbook_status == "not_found":
            recommendations.append("ACE playbook not found - using default behavior")
        elif playbook_status == "loaded":
            convergence = results.get("playbook", {}).get("convergence_status", "unknown")
            if convergence == "degraded":
                recommendations.append("Playbook convergence status is 'degraded' - consider running ACE cycles")

        if conv_status == "no_logs":
            recommendations.append("No conversation logs found - agent has not been used yet")

        return recommendations

    def generate_report(self, results: dict[str, Any], output_path: Path) -> Path:
        """Generate a comprehensive markdown report."""
        lines = ["# GenAI Companion with ACE - Performance Evaluation Report", ""]
        lines.append(f"**Generated:** {results['overall']['timestamp']}")
        lines.append(f"**Overall Health Status:** {results['overall']['health_status'].upper()}")
        lines.append("")
        self._append_summary(lines, results)
        self._append_sqlite_section(lines, results.get("sqlite_db", {}))
        self._append_vector_section(lines, results.get("vector_store", {}))
        self._append_playbook_section(lines, results.get("playbook", {}))
        self._append_conversation_section(lines, results.get("conversations", {}))
        self._append_recommendations(lines, results["overall"].get("recommendations", []))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def _append_summary(self, lines: list[str], results: dict[str, Any]) -> None:
        lines.append("## Executive Summary")
        lines.append("")
        summary = results["overall"]["summary"]
        lines.append(f"- **Total Sessions:** {summary['total_sessions']}")
        lines.append(f"- **Total Conversation Turns:** {summary['total_conversation_turns']}")
        lines.append(f"- **Vector Store Documents:** {summary['vector_store_documents']}")
        lines.append(f"- **Playbook Version:** {summary['playbook_version']}")
        lines.append(f"- **Playbook Iteration:** {summary['playbook_iteration']}")
        lines.append("")

    def _append_sqlite_section(self, lines: list[str], sqlite: dict[str, Any]) -> None:
        lines.append("## 1. SQLite Conversation History Database")
        lines.append("")
        status = sqlite.get("status")
        if status == "healthy":
            lines.append(f"[OK] **Status:** {status}")
            lines.append(f"- Total Sessions: {sqlite.get('total_sessions', 0)}")
            lines.append(f"- Total Messages: {sqlite.get('total_messages', 0)}")
            lines.append(f"- Avg Messages/Session: {sqlite.get('avg_messages_per_session', 0)}")
            if sqlite.get("sessions_by_mode"):
                lines.append(f"- Sessions by Mode: {sqlite['sessions_by_mode']}")
            if sqlite.get("sessions_by_course"):
                lines.append(f"- Sessions by Course: {sqlite['sessions_by_course']}")
        else:
            lines.append(f"[ERROR] **Status:** {status or 'unknown'}")
            if "error" in sqlite:
                lines.append(f"- Error: {sqlite['error']}")
        lines.append("")

    def _append_vector_section(self, lines: list[str], vector: dict[str, Any]) -> None:
        lines.append("## 2. Chroma Vector Store")
        lines.append("")
        status = vector.get("status")
        if status == "healthy":
            lines.append(f"[OK] **Status:** {status}")
            lines.append(f"- Collection: {vector.get('collection_name', 'unknown')}")
            lines.append(f"- Document Count: {vector.get('document_count', 0)}")
        elif status == "collection_not_found":
            lines.append(f"[WARNING] **Status:** {status}")
            lines.append(f"- Available Collections: {vector.get('available_collections', [])}")
        else:
            lines.append(f"[ERROR] **Status:** {status or 'unknown'}")
            if "error" in vector:
                lines.append(f"- Error: {vector['error']}")
        lines.append("")

    def _append_playbook_section(self, lines: list[str], playbook: dict[str, Any]) -> None:
        lines.append("## 3. ACE Playbook Performance")
        lines.append("")
        if playbook.get("status") != "loaded":
            lines.append(f"[ERROR] **Status:** {playbook.get('status', 'unknown')}")
            if "error" in playbook:
                lines.append(f"- Error: {playbook['error']}")
            lines.append("")
            return
        lines.append(f"[OK] **Status:** {playbook['status']}")
        lines.append(f"- Version: {playbook.get('version', 'unknown')}")
        lines.append(f"- Iteration: {playbook.get('iteration', 0)}")
        lines.append(f"- Convergence Status: **{playbook.get('convergence_status', 'unknown')}**")
        lines.append("")
        self._append_playbook_metrics(lines, playbook)
        self._append_heuristic_usage(lines, playbook.get("heuristic_usage", {}))
        lines.append("")

    @staticmethod
    def _append_playbook_metrics(lines: list[str], playbook: dict[str, Any]) -> None:
        lines.append("### Performance Metrics")
        perf = playbook.get("performance_metrics", {})
        lines.append(f"- Accuracy: {perf.get('accuracy', 0.0):.3f}")
        lines.append(f"- BLEU Score: {perf.get('bleu_score', 0.0):.4f}")
        lines.append(f"- ROUGE Score: {perf.get('rouge_score', 0.0):.4f}")
        lines.append(f"- Semantic Similarity: {perf.get('semantic_similarity', 0.0):.4f}")
        lines.append(f"- Avg Tokens: {perf.get('avg_tokens', 0.0)}")
        lines.append(f"- Inference Time: {perf.get('inference_time_sec', 0.0):.2f}s")
        lines.append("")
        lines.append("### Content Statistics")
        content = playbook.get("content_stats", {})
        lines.append(f"- Heuristics: {content.get('heuristics_count', 0)}")
        lines.append(f"- Examples: {content.get('examples_count', 0)}")
        lines.append(f"- History Entries: {content.get('history_entries', 0)}")

    @staticmethod
    def _append_heuristic_usage(lines: list[str], heuristic_usage: dict[str, Any]) -> None:
        if heuristic_usage.get("status") == "no_logs":
            return
        lines.append("")
        lines.append("### Heuristic Usage Analysis")
        lines.append(f"- Total Turns Analyzed: {heuristic_usage.get('total_turns_analyzed', 0)}")
        lines.append(
            f"- Heuristics with Usage: {heuristic_usage.get('heuristics_with_usage', 0)}/"
            f"{heuristic_usage.get('total_heuristics', 0)}"
        )
        stats = heuristic_usage.get("heuristic_stats", [])
        if stats:
            lines.append("")
            lines.append("Heuristic Usage Details:")
            for stat in stats:
                lines.append(f"- **{stat['id']}**: {stat['usage_count']} uses ({stat['usage_percentage']}%)")

    def _append_conversation_section(self, lines: list[str], conv: dict[str, Any]) -> None:
        lines.append("## 4. Agent Success Metrics (Conversation Logs)")
        lines.append("")
        if conv.get("status") != "analyzed":
            lines.append(f"[ERROR] **Status:** {conv.get('status', 'unknown')}")
            lines.append("")
            return
        lines.append(f"[OK] **Status:** {conv['status']}")
        lines.append(f"- Total Turns: {conv.get('total_turns', 0)}")
        lines.append(f"- Unique Sessions: {conv.get('unique_sessions', 0)}")
        lines.append(f"- Avg Turns/Session: {conv.get('avg_turns_per_session', 0.0)}")
        lines.append("")
        self._append_answer_quality(lines, conv.get("answer_quality", {}))
        self._append_retrieval_quality(lines, conv.get("retrieval_quality", {}))
        self._append_usage_patterns(lines, conv.get("usage_patterns", {}))
        lines.append("")

    @staticmethod
    def _append_answer_quality(lines: list[str], answer_quality: dict[str, Any]) -> None:
        lines.append("### Answer Quality")
        lines.append(f"- Avg Answer Length: {answer_quality.get('avg_answer_length', 0.0):.0f} chars")
        lines.append(f"- Answer/Question Ratio: {answer_quality.get('answer_to_question_ratio', 0.0):.2f}")
        lines.append("")

    @staticmethod
    def _append_retrieval_quality(lines: list[str], retrieval_quality: dict[str, Any]) -> None:
        lines.append("### Retrieval Quality")
        lines.append(f"- Source Coverage Rate: {retrieval_quality.get('source_coverage_rate', 0.0):.1%}")
        lines.append(f"- Avg Sources/Turn: {retrieval_quality.get('avg_sources_per_turn', 0.0):.2f}")
        lines.append("")

    @staticmethod
    def _append_usage_patterns(lines: list[str], usage_patterns: dict[str, Any]) -> None:
        lines.append("### Usage Patterns")
        modes = usage_patterns.get("modes_used")
        courses = usage_patterns.get("courses_covered")
        if modes:
            lines.append(f"- Modes Used: {modes}")
        if courses:
            lines.append(f"- Courses Covered: {courses}")

    @staticmethod
    def _append_recommendations(lines: list[str], recommendations: list[str]) -> None:
        lines.append("## 5. Recommendations")
        lines.append("")
        if recommendations:
            for idx, rec in enumerate(recommendations, 1):
                lines.append(f"{idx}. {rec}")
        else:
            lines.append("[OK] No issues detected. System is operating normally.")
        lines.append("")


def main() -> None:  # noqa: C901
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate GenAI Companion with ACE performance")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/companion_config.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/performance_evaluation.md"),
        help="Output path for report",
    )
    parser.add_argument("--json", action="store_true", help="Also output JSON results")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically detect issues and suggest fixes (interactive mode)",
    )

    args = parser.parse_args()

    # Load config
    config_path = args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return

    companion_config = CompanionConfig.from_file(config_path)

    history_db = companion_config.conversation_history_path()
    conversation_logs = companion_config.outputs.conversations
    playbook_dir = companion_config.outputs.ace_playbooks
    vector_cfg = companion_config.vector_store_config()
    chroma_path = vector_cfg.persist_directory
    chroma_collection = vector_cfg.collection_name

    # Find latest playbook
    playbooks = sorted(playbook_dir.glob("playbook_*.yaml"), key=lambda p: p.stat().st_mtime)
    playbook_path = playbooks[-1] if playbooks else playbook_dir / "playbook_v1.0.13.yaml"

    # Run evaluation
    evaluator = PerformanceEvaluator(
        history_db_path=history_db,
        conversation_logs_dir=conversation_logs,
        playbook_path=playbook_path,
        chroma_path=chroma_path,
        chroma_collection=chroma_collection,
    )

    print("Starting comprehensive performance evaluation...")
    print("")
    results = evaluator.evaluate_all()

    # Generate report
    print("")
    print("Generating report...")
    report_path = evaluator.generate_report(results, args.output)
    print(f"[OK] Report saved to: {report_path}")

    # Save JSON if requested
    if args.json:
        json_path = args.output.with_suffix(".json")
        json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] JSON results saved to: {json_path}")

    print("")
    print("=" * 60)
    print(f"Overall Health Status: {results['overall']['health_status'].upper()}")
    print("=" * 60)

    # Handle --fix flag
    if args.fix:
        print("")
        print("=" * 60)
        print("AUTOMATED FIX SUGGESTIONS")
        print("=" * 60)
        print("")

        fixes_applied = False

        # Check ChromaDB corruption
        vector_status = results.get("vector_store", {}).get("status", "unknown")
        if vector_status in ("corrupted_or_incompatible", "error"):
            print("[ISSUE] ChromaDB vector store is corrupted or has errors")
            print(f"  Error: {results.get('vector_store', {}).get('error', 'Unknown error')}")
            print("")
            print("[FIX] Reset the vector store:")
            print("  genai-companion reset-vector-store")
            print("")
            response = input("Would you like to reset the vector store now? (y/N): ").strip().lower()
            if response == "y":
                try:
                    from genai_companion_with_ace.cli import reset_vector_store

                    reset_vector_store(config_path, force=False)
                    fixes_applied = True
                    print("[OK] Vector store reset initiated")
                except Exception as e:
                    print(f"[ERROR] Failed to reset vector store: {e}")
                    print("  Please run manually: genai-companion reset-vector-store")
            print("")

        # Check playbook convergence
        playbook_status = results.get("playbook", {}).get("status", "unknown")
        convergence = results.get("playbook", {}).get("convergence_status", "unknown")
        if playbook_status == "loaded" and convergence == "degraded":
            print("[ISSUE] Playbook convergence status is 'degraded'")
            print("")
            print("[FIX] Run ACE improvement cycles:")
            print("  genai-companion trigger-ace")
            print("")
            response = input("Would you like to trigger ACE cycles now? (y/N): ").strip().lower()
            if response == "y":
                try:
                    from genai_companion_with_ace.cli import trigger_ace

                    trigger_ace(config_path, iterations=None)
                    fixes_applied = True
                    print("[OK] ACE cycles triggered")
                except Exception as e:
                    print(f"[ERROR] Failed to trigger ACE cycles: {e}")
                    print("  Please run manually: genai-companion trigger-ace")
            print("")

        # Check for no conversation logs
        conv_status = results.get("conversations", {}).get("status", "unknown")
        if conv_status == "no_logs":
            print("[ISSUE] No conversation logs found")
            print("  The agent has not been used yet, so ACE cycles cannot run.")
            print("  Start using the companion to generate conversation logs.")
            print("")

        if not fixes_applied:
            print("[OK] No automated fixes were applied")
            print("  Review the recommendations in the report above for manual fixes.")
        else:
            print("[OK] Some fixes were applied. Re-run evaluation to verify.")
        print("")


if __name__ == "__main__":
    main()
