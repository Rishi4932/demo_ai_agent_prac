import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class PlanResult:
    success: bool
    goal: str
    steps: List[Dict[str, Any]]
    message: str


class Planner:
    """
    Converts an understood task into an ordered execution plan.

    The planner supports:

    1. LLM-based planning through a provider.
    2. Deterministic planning without an LLM.
    3. Action validation.
    4. Basic action dependency handling.
    5. Automatic insertion of required browser startup steps.
    6. Expected-result generation for every step.
    7. Structured verification for save workflows.
    """

    SYSTEM_PROMPT = """
You are the planning component of a Windows computer-use AI agent.

Your job is to convert a structured task into an ordered execution plan.

Return ONLY valid JSON.

Required format:

{
    "goal": "short goal description",
    "steps": [
        {
            "step_number": 1,
            "description": "human-readable description",
            "action": "action_type",
            "parameters": {},
            "expected_result": "what should happen",
            "verification": {}
        }
    ]
}

Rules:

1. Steps must be in execution order.
2. Each step should have one primary action.
3. Do not execute any action.
4. Do not invent unnecessary steps.
5. Include an expected result for every step.
6. Use only supported action types.
7. Browser actions that require an active browser must occur after browser_start.
8. For file-saving workflows, when the saved path and expected file content are known, include a structured verification object such as:
   {"type": "file_content_contains", "path": "C:\\path\\file.txt", "text": "expected content"}
"""

    SUPPORTED_ACTIONS = {
        "browser_start",
        "browser_open_url",
        "browser_click",
        "browser_fill",
        "browser_get_text",
        "browser_screenshot",
        "browser_wait",
        "browser_close",

        "open_application",
        "click_control",
        "type_into_control",
        "get_control_text",

        "uia_click_control",
        "uia_invoke_control",
        "uia_get_control_text",

        "click",
        "double_click",
        "right_click",
        "type",
        "press",
        "hotkey",
        "scroll",
        "wait",
        "screenshot",
    }

    BROWSER_ACTIONS_REQUIRING_START = {
        "browser_open_url",
        "browser_click",
        "browser_fill",
        "browser_get_text",
        "browser_screenshot",
        "browser_wait",
    }

    def __init__(self, provider=None):
        self.provider = provider

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def create_plan(
        self,
        task_description: str,
        understood_steps: List[Dict[str, Any]],
    ) -> PlanResult:

        if not task_description.strip():
            return PlanResult(
                success=False,
                goal="",
                steps=[],
                message="Task description is empty.",
            )

        if not understood_steps:
            return PlanResult(
                success=False,
                goal=task_description,
                steps=[],
                message="No understood steps were provided.",
            )

        if self.provider is None:
            return self._deterministic_plan(
                task_description,
                understood_steps,
            )

        try:
            response = self.provider.generate(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=json.dumps({
                    "task_description": task_description,
                    "steps": understood_steps,
                }),
            )

            return self._parse_response(
                task_description,
                response,
            )

        except Exception as error:
            return PlanResult(
                success=False,
                goal=task_description,
                steps=[],
                message=f"Planning failed: {error}",
            )

    # ==========================================================
    # DETERMINISTIC PLANNER
    # ==========================================================

    def _deterministic_plan(
        self,
        task_description: str,
        understood_steps: List[Dict[str, Any]],
    ) -> PlanResult:

        normalized_steps = []

        for step in understood_steps:

            if not isinstance(step, dict):
                return PlanResult(
                    success=False,
                    goal=task_description,
                    steps=[],
                    message="Every understood step must be a dictionary.",
                )

            action = step.get("action")

            if action not in self.SUPPORTED_ACTIONS:
                return PlanResult(
                    success=False,
                    goal=task_description,
                    steps=[],
                    message=f"Unsupported action: {action}",
                )

            description = step.get(
                "description",
                f"Execute {action}",
            )

            parameters = step.get(
                "parameters",
                {},
            )

            if not isinstance(parameters, dict):
                parameters = {}

            normalized_steps.append({
                "description": description,
                "action": action,
                "parameters": parameters,
            })

        normalized_steps = self._apply_dependencies(
            normalized_steps
        )

        normalized_steps = self._attach_save_verification(
            normalized_steps
        )

        planned_steps = []

        for index, step in enumerate(
            normalized_steps,
            start=1,
        ):

            action = step["action"]

            planned_step = {
                "step_number": index,
                "description": step["description"],
                "action": action,
                "parameters": step["parameters"],
                "expected_result": self._default_expected_result(
                    action
                ),
            }

            if step.get("verification") is not None:
                planned_step["verification"] = dict(
                    step["verification"]
                )

            planned_steps.append(planned_step)

        return PlanResult(
            success=True,
            goal=task_description,
            steps=planned_steps,
            message="Execution plan created successfully.",
        )

    # ==========================================================
    # DEPENDENCY HANDLING
    # ==========================================================

    def _apply_dependencies(
        self,
        steps: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Adds required prerequisite actions when they are missing.

        Current dependency rules:

        browser_open_url
        browser_click
        browser_fill
        browser_get_text
        browser_screenshot
        browser_wait

        require browser_start first.

        browser_close does not require automatic startup.
        """

        result = []
        browser_started = False

        for step in steps:

            action = step["action"]

            if (
                action in self.BROWSER_ACTIONS_REQUIRING_START
                and not browser_started
            ):

                result.append({
                    "description": "Open the browser",
                    "action": "browser_start",
                    "parameters": {},
                })

                browser_started = True

            result.append(step)

            if action == "browser_start":
                browser_started = True

            elif action == "browser_close":
                browser_started = False

        return result

    # ==========================================================
    # SAVE WORKFLOW VERIFICATION
    # ==========================================================

    def _looks_like_file_path(
        self,
        value: Any,
    ) -> bool:
        """
        Return True when a text value looks like a Windows file path.
        """

        if not isinstance(value, str):
            return False

        value = value.strip()

        if not value:
            return False

        if re.match(
            r"^[A-Za-z]:[\\/].+",
            value,
        ):
            return True

        if value.startswith("\\\\"):
            return True

        return False

    def _get_parameter_text(
        self,
        parameters: Dict[str, Any],
    ) -> Optional[str]:
        """
        Extract text-like input from an action's parameters.
        """

        for key in (
            "text",
            "value",
            "content",
        ):

            value = parameters.get(key)

            if isinstance(value, str) and value.strip():
                return value

        return None

    def _is_save_hotkey(
        self,
        parameters: Dict[str, Any],
    ) -> bool:
        """
        Recognize Ctrl+S and Ctrl+Shift+S keyboard shortcuts.
        """

        keys = parameters.get(
            "keys",
            [],
        )

        if not isinstance(
            keys,
            (list, tuple),
        ):
            return False

        normalized = {
            str(key).strip().lower()
            for key in keys
        }

        if normalized == {
            "ctrl",
            "s",
        }:
            return True

        if normalized == {
            "ctrl",
            "shift",
            "s",
        }:
            return True

        return False

    def _is_save_completion_action(
        self,
        step: Dict[str, Any],
    ) -> bool:
        """
        Recognize the action that normally confirms the save dialog.
        """

        action = step.get("action")

        parameters = step.get(
            "parameters",
            {},
        )

        if action == "press":

            key = str(
                parameters.get(
                    "key",
                    "",
                )
            ).strip().lower()

            return key in {
                "enter",
                "return",
            }

        description = str(
            step.get(
                "description",
                "",
            )
        ).lower()

        if action in {
            "click",
            "double_click",
            "click_control",
            "uia_click_control",
            "uia_invoke_control",
        }:
            return "save" in description

        return False

    def _attach_save_verification(
        self,
        steps: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Attach structured file verification to a save workflow.

        The planner looks for this sequence:

            save shortcut
                ↓
            file path entered
                ↓
            save confirmation

        It also remembers the document text entered before the save
        operation.

        When both the path and expected content are known, the final
        save action receives a file_content_contains verification
        requirement.

        file_content_contains also confirms that the file exists,
        because the verifier first checks the file before reading it.
        """

        result = [
            dict(step)
            for step in steps
        ]

        document_text = None
        save_triggered = False
        save_path = None

        for index, step in enumerate(result):

            action = step.get(
                "action"
            )

            parameters = step.get(
                "parameters",
                {},
            )

            if not isinstance(
                parameters,
                dict,
            ):
                parameters = {}
                step["parameters"] = parameters

            text = self._get_parameter_text(
                parameters
            )

            # -------------------------------------------------
            # Remember document text entered before saving.
            # -------------------------------------------------

            if (
                action in {
                    "type",
                    "type_into_control",
                }
                and text
                and not self._looks_like_file_path(
                    text
                )
                and not save_triggered
            ):
                document_text = text

            # -------------------------------------------------
            # Detect Ctrl+S / Ctrl+Shift+S.
            # -------------------------------------------------

            if (
                action == "hotkey"
                and self._is_save_hotkey(
                    parameters
                )
            ):
                save_triggered = True
                continue

            if save_triggered:

                # -------------------------------------------------
                # The first path-like text after the save shortcut
                # is treated as the destination file path.
                # -------------------------------------------------

                if (
                    text
                    and self._looks_like_file_path(
                        text
                    )
                ):
                    save_path = text.strip()
                    continue

                # -------------------------------------------------
                # The action confirming the save receives
                # structured verification.
                # -------------------------------------------------

                if (
                    save_path
                    and self._is_save_completion_action(
                        step
                    )
                ):

                    if document_text:

                        step["verification"] = {
                            "type": "file_content_contains",
                            "path": save_path,
                            "text": document_text,
                        }

                    else:

                        step["verification"] = {
                            "type": "file_exists",
                            "path": save_path,
                        }

                    # Stop after the first confirmed save operation.
                    save_triggered = False
                    save_path = None

        return result

    # ==========================================================
    # EXPECTED RESULTS
    # ==========================================================

    def _default_expected_result(
        self,
        action: str,
    ) -> str:

        expected_results = {

            "browser_start":
                "A browser window is open.",

            "browser_open_url":
                "The requested webpage is loaded.",

            "browser_click":
                "The requested web element is activated.",

            "browser_fill":
                "The requested field contains the provided text.",

            "browser_get_text":
                "The requested webpage text is retrieved.",

            "browser_screenshot":
                "A browser screenshot is saved.",

            "browser_wait":
                "The requested browser wait period has completed.",

            "browser_close":
                "The browser is closed.",

            "open_application":
                "The requested Windows application is open.",

            "click_control":
                "The requested Windows control is activated.",

            "type_into_control":
                "The requested Windows control contains the text.",

            "get_control_text":
                "The requested control text is retrieved.",

            "uia_click_control":
                "The requested native Windows UI control is clicked.",

            "uia_invoke_control":
                "The requested native Windows UI control is invoked.",

            "uia_get_control_text":
                "The requested native Windows UI control text is retrieved.",

            "click":
                "The requested screen location is clicked.",

            "double_click":
                "The requested screen location is double-clicked.",

            "right_click":
                "The requested screen location is right-clicked.",

            "type":
                "The requested text is entered.",

            "press":
                "The requested keyboard key is pressed.",

            "hotkey":
                "The requested keyboard shortcut is executed.",

            "scroll":
                "The screen is scrolled as requested.",

            "wait":
                "The requested wait period has completed.",

            "screenshot":
                "A screenshot is saved.",
        }

        return expected_results.get(
            action,
            f"Action '{action}' completed successfully.",
        )

    # ==========================================================
    # LLM RESPONSE PARSING
    # ==========================================================

    def _parse_response(
        self,
        task_description: str,
        response: str,
    ) -> PlanResult:

        try:
            data = json.loads(response)

        except json.JSONDecodeError:

            cleaned = self._extract_json(
                response
            )

            try:
                data = json.loads(
                    cleaned
                )

            except json.JSONDecodeError as error:

                return PlanResult(
                    success=False,
                    goal=task_description,
                    steps=[],
                    message=(
                        f"Invalid planner JSON: {error}"
                    ),
                )

        if not isinstance(
            data,
            dict,
        ):

            return PlanResult(
                success=False,
                goal=task_description,
                steps=[],
                message=(
                    "Planner response must be "
                    "a JSON object."
                ),
            )

        raw_steps = data.get(
            "steps",
            [],
        )

        if not isinstance(
            raw_steps,
            list,
        ):

            return PlanResult(
                success=False,
                goal=task_description,
                steps=[],
                message="'steps' must be a list.",
            )

        validated_steps = []

        for step in raw_steps:

            if not isinstance(
                step,
                dict,
            ):
                continue

            action = step.get(
                "action"
            )

            if action not in self.SUPPORTED_ACTIONS:

                return PlanResult(
                    success=False,
                    goal=task_description,
                    steps=[],
                    message=(
                        f"Unsupported action in plan: "
                        f"{action}"
                    ),
                )

            description = step.get(
                "description",
                f"Execute {action}",
            )

            parameters = step.get(
                "parameters",
                {},
            )

            if not isinstance(
                parameters,
                dict,
            ):
                parameters = {}

            expected_result = step.get(
                "expected_result",
                self._default_expected_result(
                    action
                ),
            )

            validated_step = {
                "description": description,
                "action": action,
                "parameters": parameters,
                "expected_result": expected_result,
            }

            verification = step.get(
                "verification"
            )

            if isinstance(
                verification,
                dict,
            ):
                validated_step["verification"] = dict(
                    verification
                )

            validated_steps.append(
                validated_step
            )

        if not validated_steps:

            return PlanResult(
                success=False,
                goal=task_description,
                steps=[],
                message="No valid plan steps were produced.",
            )

        # Apply the same automatic save verification used
        # by the deterministic planner when the LLM did
        # not provide it.
        validated_steps = self._attach_save_verification(
            validated_steps
        )

        # Normalize step numbers after validation.

        planned_steps = []

        for index, step in enumerate(
            validated_steps,
            start=1,
        ):

            planned_step = {
                "step_number": index,
                "description": step["description"],
                "action": step["action"],
                "parameters": step["parameters"],
                "expected_result": step["expected_result"],
            }

            if step.get(
                "verification"
            ) is not None:

                planned_step["verification"] = dict(
                    step["verification"]
                )

            planned_steps.append(
                planned_step
            )

        return PlanResult(
            success=True,
            goal=data.get(
                "goal",
                task_description,
            ),
            steps=planned_steps,
            message="Plan created successfully.",
        )

    # ==========================================================
    # JSON EXTRACTION
    # ==========================================================

    def _extract_json(
        self,
        response: str,
    ) -> str:
        """
        Extract the outermost JSON object from a provider response.
        """

        start = response.find(
            "{"
        )

        end = response.rfind(
            "}"
        )

        if (
            start != -1
            and end != -1
            and end > start
        ):
            return response[
                start:end + 1
            ]

        return response


class MockPlannerProvider:
    """
    Development-only planner provider.

    Used to test LLM-style planner responses without
    requiring an external API.
    """

    def __init__(
        self,
        response: str,
    ):
        self.response = response

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return self.response