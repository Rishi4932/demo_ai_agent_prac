from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from intelligence.vision_reasoner import VisualElement


@dataclass
class VerificationResult:
    success: bool
    confidence: float
    reason: str
    matched_elements: List[VisualElement] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "success": self.success,
            "confidence": self.confidence,
            "reason": self.reason,
            "matched_elements": [
                element.to_dict()
                for element in self.matched_elements
            ],
            "details": self.details,
        }


class ActionVerifier:

    def _normalize_text(self, text: str) -> str:
        """
        Normalize OCR/UI text so small differences in whitespace,
        punctuation and line breaks do not cause false failures.
        """

        if text is None:
            return ""

        text = str(text)

        # Normalize whitespace
        text = " ".join(text.split())

        # Normalize common OCR punctuation artifacts
        text = text.replace("‘", "'")
        text = text.replace("’", "'")
        text = text.replace("“", '"')
        text = text.replace("”", '"')

        return text.strip().lower()

    # =============================================================
    # VISUAL VERIFICATION
    # =============================================================

    def verify_element_present(
        self,
        elements,
        text,
        element_type=None,
    ):
        expected = self._normalize_text(text)

        matches = []

        for element in elements:
            element_text = self._normalize_text(
                getattr(element, "text", "")
            )

            if expected in element_text:

                if (
                    element_type is None
                    or getattr(element, "element_type", None)
                    == element_type
                ):
                    matches.append(element)

        if matches:

            confidence = max(
                getattr(element, "confidence", 0.0)
                for element in matches
            )

            return VerificationResult(
                success=True,
                confidence=confidence,
                reason=f"Expected text '{text}' was detected.",
                matched_elements=matches,
                details={
                    "expected_text": text,
                },
            )

        return VerificationResult(
            success=False,
            confidence=0.0,
            reason=f"Expected text '{text}' was not detected.",
            matched_elements=[],
            details={
                "expected_text": text,
            },
        )

    def verify_element_absent(
        self,
        elements,
        text,
        element_type=None,
    ):
        expected = self._normalize_text(text)

        matches = []

        for element in elements:

            element_text = self._normalize_text(
                getattr(element, "text", "")
            )

            if expected in element_text:

                if (
                    element_type is None
                    or getattr(element, "element_type", None)
                    == element_type
                ):
                    matches.append(element)

        if matches:

            return VerificationResult(
                success=False,
                confidence=0.0,
                reason=f"Unexpected text '{text}' is present.",
                matched_elements=matches,
                details={
                    "expected_text": text,
                },
            )

        return VerificationResult(
            success=True,
            confidence=1.0,
            reason=f"Expected text '{text}' is absent.",
            matched_elements=[],
            details={
                "expected_text": text,
            },
        )

    def verify_text_contains(
        self,
        elements,
        text,
    ):
        """
        Verify that expected text exists somewhere in the observed
        visual elements.

        Handles:
        - OCR line breaks
        - multiple spaces
        - capitalization differences
        - common OCR punctuation artifacts
        """

        expected = self._normalize_text(text)

        if not expected:
            return VerificationResult(
                success=False,
                confidence=0.0,
                reason="Verification failed: expected text is empty.",
                matched_elements=[],
                details={
                    "expected_text": text,
                    "match_mode": "invalid_empty_target",
                },
            )

        # ---------------------------------------------------------
        # 1. Check individual elements
        # ---------------------------------------------------------

        individual_matches = []

        for element in elements:

            element_text = self._normalize_text(
                getattr(element, "text", "")
            )

            if expected in element_text:
                individual_matches.append(element)

        if individual_matches:

            confidence = max(
                getattr(element, "confidence", 0.0)
                for element in individual_matches
            )

            return VerificationResult(
                success=True,
                confidence=confidence,
                reason=f"Expected text '{text}' was detected.",
                matched_elements=individual_matches,
                details={
                    "expected_text": text,
                    "match_mode": "individual_element",
                },
            )

        # ---------------------------------------------------------
        # 2. Check combined OCR text
        # ---------------------------------------------------------

        combined_text = " ".join(
            self._normalize_text(
                getattr(element, "text", "")
            )
            for element in elements
            if getattr(element, "text", "")
        )

        if expected in combined_text:

            return VerificationResult(
                success=True,
                confidence=0.85,
                reason=(
                    f"Expected text '{text}' was detected "
                    "in combined OCR."
                ),
                matched_elements=[],
                details={
                    "expected_text": text,
                    "match_mode": "combined_elements",
                },
            )

        # ---------------------------------------------------------
        # 3. Failure
        # ---------------------------------------------------------

        return VerificationResult(
            success=False,
            confidence=0.0,
            reason=f"Expected text '{text}' was not detected.",
            matched_elements=[],
            details={
                "expected_text": text,
                "match_mode": "none",
            },
        )

    # =============================================================
    # FILE VERIFICATION
    # =============================================================

    def verify_file_exists(
        self,
        path: str,
    ) -> VerificationResult:
        """
        Verify that a file exists at the requested path.
        """

        if not path:
            return VerificationResult(
                success=False,
                confidence=0.0,
                reason="File verification failed: path is empty.",
                matched_elements=[],
                details={
                    "path": path,
                    "match_mode": "invalid_path",
                },
            )

        file_path = Path(path)

        try:
            exists = file_path.exists()
            is_file = file_path.is_file()
        except Exception as error:
            return VerificationResult(
                success=False,
                confidence=0.0,
                reason=(
                    f"File verification failed for '{path}': "
                    f"{error}"
                ),
                matched_elements=[],
                details={
                    "path": path,
                    "match_mode": "filesystem_error",
                },
            )

        if exists and is_file:
            return VerificationResult(
                success=True,
                confidence=1.0,
                reason=f"File '{path}' exists.",
                matched_elements=[],
                details={
                    "path": path,
                    "exists": True,
                    "is_file": True,
                    "match_mode": "file_exists",
                },
            )

        return VerificationResult(
            success=False,
            confidence=0.0,
            reason=f"File '{path}' does not exist.",
            matched_elements=[],
            details={
                "path": path,
                "exists": exists,
                "is_file": is_file,
                "match_mode": "file_not_found",
            },
        )

    def verify_file_content_contains(
        self,
        path: str,
        text: str,
    ) -> VerificationResult:
        """
        Verify that a file exists and contains the expected text.
        """

        if not path:
            return VerificationResult(
                success=False,
                confidence=0.0,
                reason=(
                    "File content verification failed: "
                    "path is empty."
                ),
                matched_elements=[],
                details={
                    "path": path,
                    "expected_text": text,
                    "match_mode": "invalid_path",
                },
            )

        if text is None:
            return VerificationResult(
                success=False,
                confidence=0.0,
                reason=(
                    "File content verification failed: "
                    "expected text is empty."
                ),
                matched_elements=[],
                details={
                    "path": path,
                    "expected_text": text,
                    "match_mode": "invalid_expected_text",
                },
            )

        expected = self._normalize_text(text)

        if not expected:
            return VerificationResult(
                success=False,
                confidence=0.0,
                reason=(
                    "File content verification failed: "
                    "expected text is empty."
                ),
                matched_elements=[],
                details={
                    "path": path,
                    "expected_text": text,
                    "match_mode": "invalid_expected_text",
                },
            )

        file_path = Path(path)

        try:
            if not file_path.exists():
                return VerificationResult(
                    success=False,
                    confidence=0.0,
                    reason=f"File '{path}' does not exist.",
                    matched_elements=[],
                    details={
                        "path": path,
                        "expected_text": text,
                        "match_mode": "file_not_found",
                    },
                )

            if not file_path.is_file():
                return VerificationResult(
                    success=False,
                    confidence=0.0,
                    reason=f"Path '{path}' is not a file.",
                    matched_elements=[],
                    details={
                        "path": path,
                        "expected_text": text,
                        "match_mode": "not_a_file",
                    },
                )

            content = file_path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:
            return VerificationResult(
                success=False,
                confidence=0.0,
                reason=(
                    f"Could not read '{path}' as UTF-8 text."
                ),
                matched_elements=[],
                details={
                    "path": path,
                    "expected_text": text,
                    "match_mode": "decode_error",
                },
            )

        except Exception as error:
            return VerificationResult(
                success=False,
                confidence=0.0,
                reason=(
                    f"Could not read file '{path}': "
                    f"{error}"
                ),
                matched_elements=[],
                details={
                    "path": path,
                    "expected_text": text,
                    "match_mode": "read_error",
                },
            )

        normalized_content = self._normalize_text(
            content
        )

        if expected in normalized_content:
            return VerificationResult(
                success=True,
                confidence=1.0,
                reason=(
                    f"Expected text was found in file '{path}'."
                ),
                matched_elements=[],
                details={
                    "path": path,
                    "expected_text": text,
                    "match_mode": "file_content_contains",
                },
            )

        return VerificationResult(
            success=False,
            confidence=0.0,
            reason=(
                f"Expected text was not found in file '{path}'."
            ),
            matched_elements=[],
            details={
                "path": path,
                "expected_text": text,
                "match_mode": "file_content_not_found",
            },
        )