from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DecisionResult:
    success: bool
    action: Optional[Dict[str, Any]]
    reasoning: str
    message: str
    expected: Optional[Dict[str, Any]] = None


class DecisionEngine:
    """
    Selects the next action based on:

    - Current task step
    - Current computer state
    - Previous execution history
    - Available action information

    The decision engine can operate in two modes:

    1. Deterministic mode
       Uses the current computer state and execution history.

    2. Provider mode
       Delegates the decision to an external AI/LLM provider.
    """

    BROWSER_ACTIONS_REQUIRING_BROWSER = {
        "browser_open_url",
        "browser_click",
        "browser_fill",
        "browser_press",
        "browser_get_text",
        "browser_screenshot",
        "browser_wait",
        "browser_get_title",
        "browser_get_url",
    }

    def __init__(self, provider=None):
        self.provider = provider

    def choose_action(
        self,
        task: str,
        current_step: Dict[str, Any],
        computer_state: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> DecisionResult:

        if not task.strip():
            return DecisionResult(
                success=False,
                action=None,
                reasoning="No task was provided.",
                message="Decision failed: empty task.",
            )

        if not current_step:
            return DecisionResult(
                success=False,
                action=None,
                reasoning="No current step was provided.",
                message="Decision failed: no current step.",
            )

        if self.provider is not None:
            try:
                return self._provider_decision(
                    task=task,
                    current_step=current_step,
                    computer_state=computer_state or {},
                    history=history or [],
                )

            except Exception as error:
                return DecisionResult(
                    success=False,
                    action=None,
                    reasoning="Provider decision failed.",
                    message=f"Decision provider error: {error}",
                )

        return self._deterministic_decision(
            task=task,
            current_step=current_step,
            computer_state=computer_state or {},
            history=history or [],
        )

    def _deterministic_decision(
        self,
        task: str,
        current_step: Dict[str, Any],
        computer_state: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> DecisionResult:

        action_type = current_step.get("action")

        if not action_type:
            return DecisionResult(
                success=False,
                action=None,
                reasoning="Current step has no action.",
                message="Decision failed: action missing.",
            )

        # ---------------------------------------------------------
        # Determine whether browser state is known.
        # ---------------------------------------------------------

        browser_state_known = "browser_open" in computer_state

        browser_open = computer_state.get("browser_open")

        # If the observation does not explicitly tell us whether
        # the browser is open, inspect previous execution history.
        if not browser_state_known:
            browser_open = self._browser_open_from_history(history)

        # ---------------------------------------------------------
        # Browser dependency handling.
        # ---------------------------------------------------------

        if self._browser_action_requires_browser(action_type):

            if browser_open is False:

                return DecisionResult(
                    success=True,
                    action={
                        "action": "browser_start",
                    },
                    reasoning=(
                        f"The current step requires browser action "
                        f"'{action_type}', but the browser is not open. "
                        f"Starting the browser first."
                    ),
                    message=(
                        "Browser is not open; browser_start selected "
                        "as the next action."
                    ),
                    expected=None,
                )

        # ---------------------------------------------------------
        # Build the requested action.
        # ---------------------------------------------------------

        parameters = current_step.get(
            "parameters",
            {},
        )

        action = {
            "action": action_type,
            **parameters,
        }

        # ---------------------------------------------------------
        # Preserve expected verification information.
        # ---------------------------------------------------------

        expected = self._build_expected(
            current_step=current_step,
            action=action,
        )

        reasoning = self._generate_reasoning(
            current_step=current_step,
            computer_state=computer_state,
            history=history,
        )

        return DecisionResult(
            success=True,
            action=action,
            reasoning=reasoning,
            message="Next action selected successfully.",
            expected=expected,
        )

    def _build_expected(
        self,
        current_step: Dict[str, Any],
        action: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Convert the planner's expected_result into the structured
        verification information used by AgentExecutor.

        For now, preserve explicit structured verification data
        when supplied by the planner.

        Plain human-readable expected_result text is retained as
        metadata rather than being blindly sent to OCR verification.
        """

        expected_result = current_step.get(
            "expected_result"
        )

        parameters = current_step.get(
            "parameters",
            {},
        )

        # ---------------------------------------------------------
        # If the planner already supplied structured verification,
        # preserve it exactly.
        # ---------------------------------------------------------

        verification = current_step.get(
            "verification"
        )

        if isinstance(verification, dict):
            return dict(verification)

        # ---------------------------------------------------------
        # Some planners may place verification information inside
        # the action parameters.
        # ---------------------------------------------------------

        parameter_verification = parameters.get(
            "verification"
        )

        if isinstance(parameter_verification, dict):
            return dict(parameter_verification)

        # ---------------------------------------------------------
        # No structured verification information exists yet.
        #
        # Do not convert a generic sentence such as:
        # "File is saved successfully."
        #
        # into OCR verification because that would not verify
        # actual filesystem state.
        # ---------------------------------------------------------

        if expected_result is None:
            return None

        if isinstance(expected_result, dict):
            return dict(expected_result)

        return None

    def _browser_action_requires_browser(
        self,
        action_type: str,
    ) -> bool:
        """
        Return True when an action requires the browser
        to already be running.
        """

        return action_type in self.BROWSER_ACTIONS_REQUIRING_BROWSER

    def _browser_open_from_history(
        self,
        history: List[Dict[str, Any]],
    ) -> Optional[bool]:
        """
        Determine the browser state from successful execution history.

        AgentController normally stores history like this:

        {
            "action": {
                "action": "browser_start"
            },
            "result": {
                "success": True,
                "message": "Browser started."
            }
        }

        Older/simple history may also look like:

        {
            "action": "browser_start",
            "success": True
        }

        Both formats are supported.
        """

        browser_open = None

        for record in history:

            if not isinstance(record, dict):
                continue

            # -----------------------------------------------------
            # Read action information.
            # -----------------------------------------------------

            action_data = record.get("action")

            if isinstance(action_data, dict):
                action = action_data.get("action")
            else:
                action = action_data

            # -----------------------------------------------------
            # Read execution result.
            # -----------------------------------------------------

            result_data = record.get("result")

            if isinstance(result_data, dict):
                success = result_data.get("success")
            else:
                # Backward-compatible format.
                success = record.get("success")

            # Only successful actions should change our
            # understanding of browser state.
            if success is not True:
                continue

            # -----------------------------------------------------
            # Update browser state.
            # -----------------------------------------------------

            if action == "browser_start":
                browser_open = True

            elif action == "browser_close":
                browser_open = False

        return browser_open

    def _generate_reasoning(
        self,
        current_step: Dict[str, Any],
        computer_state: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> str:

        action = current_step.get(
            "action",
            "unknown",
        )

        description = current_step.get(
            "description",
            "",
        )

        expected_result = current_step.get(
            "expected_result",
            "",
        )

        history_count = len(history)

        return (
            f"Current step: {description}. "
            f"Selected action '{action}' because it "
            f"corresponds to the current planned step. "
            f"Expected result: {expected_result}. "
            f"Previous actions recorded: {history_count}."
        )

    def _provider_decision(
        self,
        task: str,
        current_step: Dict[str, Any],
        computer_state: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> DecisionResult:

        response = self.provider.generate(
            task=task,
            current_step=current_step,
            computer_state=computer_state,
            history=history,
        )

        if not isinstance(response, dict):
            raise ValueError(
                "Decision provider must return a dictionary."
            )

        action = response.get("action")

        if not isinstance(action, dict):
            raise ValueError(
                "Decision provider response must contain "
                "an action dictionary."
            )

        expected = response.get(
            "expected"
        )

        if expected is not None and not isinstance(
            expected,
            dict,
        ):
            expected = None

        return DecisionResult(
            success=True,
            action=action,
            reasoning=response.get(
                "reasoning",
                "Provider selected the action.",
            ),
            message="Provider selected next action.",
            expected=expected,
        )


class MockDecisionProvider:
    """
    Development-only decision provider.

    Returns a predefined decision so the decision
    engine can be tested without an external AI API.
    """

    def __init__(
        self,
        action: Dict[str, Any],
        reasoning: str = "Mock decision.",
    ):
        self.action = action
        self.reasoning = reasoning

    def generate(
        self,
        task: str,
        current_step: Dict[str, Any],
        computer_state: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        return {
            "action": self.action,
            "reasoning": self.reasoning,
        }