"""Public ownership/error contracts. CPU doubles only; no model execution."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from aureon.api import GenerationEngine, GenerationClosed
from aureon.api.contracts import INPUT_BYTES, OUTPUT_BYTES, positive, validate_timeout
from aureon.api.lifecycle import close_preserving_error
from aureon.api.parallel_policy import ParallelPolicy, LANE_RESERVATION
from aureon.api.session import EngineSession, SessionFailed


class ValueContracts(unittest.TestCase):
    def test_runtime_schema_distinguishes_required_payload_from_diagnostics(self):
        from aureon.api import GenerationResult, QueueMetadata

        self.assertEqual(GenerationResult.__required_keys__, {"array", "valid_mask", "provenance"})
        self.assertIn("native_model_seconds", GenerationResult.__optional_keys__)
        self.assertIn("collection", QueueMetadata.__optional_keys__)
        self.assertIn("max_pending", QueueMetadata.__required_keys__)

    def test_timeout_validation_does_not_accept_bool_or_nonfinite_values(self):
        for value in (True, False, -1, float("nan"), float("inf"), "1", object()):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_timeout(value)
        self.assertIsNone(validate_timeout(None))
        self.assertEqual(validate_timeout(0), 0)
        self.assertEqual(validate_timeout(1.25), 1.25)

    def test_count_validation_preserves_strict_integer_contract(self):
        for value in (True, False, 0, -1, 1.0, "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                positive(value, "credits")
        self.assertEqual(positive(7, "credits"), 7)

    def test_planning_is_deterministic_detached_and_does_not_read_clock(self):
        limits = dict(
            max_pending=4,
            max_results=3,
            max_input_bytes=4 * INPUT_BYTES,
            max_output_bytes=3 * OUTPUT_BYTES,
        )
        evidence = {"lanes": 5, "identity": {"sha256": "same"}}
        before = deepcopy((limits, evidence))
        policy = ParallelPolicy("auto", workspace_bytes=2 * LANE_RESERVATION)
        with patch("time.monotonic", side_effect=AssertionError("policy read clock")):
            first = policy.plan(limits, evidence)
            second = policy.plan(limits, evidence)
        self.assertEqual(first, second)
        self.assertEqual(first["planned"], 2)
        self.assertEqual(
            first["fallback_reason"],
            "estimated_memory_budget",
        )
        first["planned"] = 999
        self.assertEqual(policy.plan(limits, evidence), second)
        self.assertEqual((limits, evidence), before)


class CleanupContracts(unittest.TestCase):
    def test_cleanup_preserves_primary_and_explicit_close_raises(self):
        primary = ValueError("bad user input")
        cleanup = Mock(side_effect=OSError("close failed"))
        self.assertIsNone(
            close_preserving_error(cleanup, primary, context="test cleanup")
        )
        self.assertIn("test cleanup: close failed", primary.__notes__)
        with self.assertRaisesRegex(OSError, "close failed"):
            close_preserving_error(cleanup, None, context="test cleanup")

    def test_process_interrupt_is_not_suppressed(self):
        for kind in (KeyboardInterrupt, SystemExit):
            with self.subTest(kind=kind), self.assertRaises(kind):
                close_preserving_error(
                    Mock(side_effect=kind()), ValueError("body"), context="test"
                )

    def test_successful_cleanup_returns_receipt_without_note(self):
        primary = ValueError("body")
        receipt = {"passed": True}
        self.assertIs(
            close_preserving_error(lambda: receipt, primary, context="test"), receipt
        )
        self.assertFalse(hasattr(primary, "__notes__"))

    def test_generation_cleanup_failure_is_terminal_retryable_and_preserves_body(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = GenerationEngine(
                model_file="missing",
                tokenizer_file="missing",
                work=Path(directory) / "unused",
                execute_model=True,
            )
            resident = Mock()
            resident.close.side_effect = [
                OSError("receipt unavailable"),
                {"passed": True},
            ]
            engine._session = (
                resident  # Inject cleanup failure without loading or creating assets.
            )
            primary = ValueError("consumer failed")
            with self.assertRaises(ValueError) as raised:
                with engine:
                    raise primary
            self.assertIs(raised.exception, primary)
            self.assertIn(
                "Generation engine cleanup failed: receipt unavailable",
                primary.__notes__,
            )
            self.assertEqual(engine.metadata()["state"], "failed")
            with self.assertRaises(GenerationClosed):
                engine.__enter__()
            self.assertTrue(engine.close()["passed"])
            self.assertTrue(engine.close()["passed"])
            self.assertEqual(resident.close.call_count, 2)
            self.assertFalse(engine.work.exists())

    def test_generation_explicit_cleanup_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = GenerationEngine(
                model_file="missing",
                tokenizer_file="missing",
                work=Path(directory) / "unused",
                execute_model=True,
            )
            engine._session = Mock()
            engine._session.close.side_effect = OSError("cannot close")
            with self.assertRaisesRegex(OSError, "cannot close"):
                with engine:
                    pass
            self.assertEqual(engine.metadata()["state"], "failed")

    def test_generation_context_reentry_during_draining_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = GenerationEngine(
                model_file="missing",
                tokenizer_file="missing",
                work=Path(directory) / "unused",
                execute_model=True,
            )
            dispatcher = Mock()
            dispatcher.shutdown.side_effect = [TimeoutError("still draining"), None]
            dispatcher.metadata.return_value = {"failed": False}
            engine._queue = dispatcher
            with self.assertRaises(TimeoutError):
                engine.close(timeout=0)
            with self.assertRaises(GenerationClosed):
                engine.__enter__()
            self.assertTrue(engine.close()["passed"])

    def test_dense_session_reports_cleanup_failure_without_losing_body(self):
        # Test the public context-manager contract independently of worker startup.
        session = object.__new__(EngineSession)
        for cleanup in (
            OSError("disk full"),
            {"cleanup_passed": False, "errors": ["worker"]},
        ):
            with self.subTest(cleanup=cleanup):
                kwargs = (
                    {"side_effect": cleanup}
                    if isinstance(cleanup, Exception)
                    else {"return_value": cleanup}
                )
                with patch.object(session, "close", **kwargs):
                    primary = ValueError("original request")
                    self.assertFalse(session.__exit__(ValueError, primary, None))
                    self.assertIn(
                        "Engine session cleanup failed:", primary.__notes__[0]
                    )
                    expected = (
                        OSError if isinstance(cleanup, Exception) else SessionFailed
                    )
                    with self.assertRaises(expected):
                        session.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
