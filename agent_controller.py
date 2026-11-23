from typing import Any, Dict, List, Optional
from database.connection import DatabaseConnection
from core.task_manager import TaskManager
from core.observation import ObservationManager
from core.state_machine import AgentState, AgentStateMachine

from intelligence.task_understanding import TaskUnderstanding
from intelligence.planner import Planner
from intelligence.decision_engine import DecisionEngine
from intelligence.agent_executor import AgentExecutor


class AgentController:
    """
    Central controller for the computer-use agent.

    Main pipeline:

        User Task
            ↓
        Task Understanding
            ↓
        Planning
            ↓
        Observation
            ↓
        Decision
            ↓
        Execution
            ↓
        Verification
            ↓
        Recovery if required
            ↓
        Next Step
            ↓
        Completion
    """

    @staticmethod
    def _progress(message: str):
        """Print flushed progress messages so long-running stages are visible."""
        print(f"[AGENT] {message}", flush=True)

    def __init__(
        self,
        task_manager: Optional[TaskManager] = None,
        task_understanding: Optional[TaskUnderstanding] = None,
        planner: Optional[Planner] = None,
        decision_engine: Optional[DecisionEngine] = None,
        executor: Optional[AgentExecutor] = None,
        state_machine: Optional[AgentStateMachine] = None,
        observation_manager: Optional[ObservationManager] = None,
    ):
        self.task_manager = (
            task_manager
            if task_manager is not None
            else TaskManager()
        )

        self.task_understanding = (
            task_understanding
            if task_understanding is not None
            else TaskUnderstanding()
        )

        self.planner = (
            planner
            if planner is not None
            else Planner()
        )

        self.decision_engine = (
            decision_engine
            if decision_engine is not None
            else DecisionEngine()
        )

        self.executor = (
            executor
            if executor is not None
            else AgentExecutor()
        )

        self.state_machine = (
            state_machine
            if state_machine is not None
            else AgentStateMachine()
        )

        self.observation_manager = (
            observation_manager
            if observation_manager is not None
            else ObservationManager()
        )

        # ---------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------

        self.current_task = None
        self.current_task_description: Optional[str] = None
        self.current_understanding = None
        self.current_plan = None
        self.last_observation = None
        self.database = DatabaseConnection()
        self.database.initialize()

        self.execution_history: List[
            Dict[str, Any]
        ] = []

        self.run_history: List[
            Dict[str, Any]
        ] = []

    # =============================================================
    # TASK SUBMISSION
    # =============================================================

    def submit_task(self, description: str):
        """
        Create and register a new task.

        A new task always starts from the IDLE state.
        """

        if not isinstance(description, str):
            raise ValueError(
                "Task description must be a string."
            )

        description = description.strip()

        if not description:
            raise ValueError(
                "Task description cannot be empty."
            )

        if self.state_machine.get_state() != AgentState.IDLE:
            self.state_machine.reset()

        self.current_task = (
            self.task_manager.create_task(
                description
            )
        )

        self.current_task_description = description

        self.current_understanding = None
        self.current_plan = None
        self.last_observation = None
        self.execution_history = []
        self.run_history = []

        return self.current_task

    # =============================================================
    # TASK UNDERSTANDING
    # =============================================================

    def understand_task(self) -> Any:
        """
        Understand the current natural-language task.
        """

        if self.current_task is None:
            raise RuntimeError(
                "No active task. Call submit_task() first."
            )

        description = self.current_task_description

        if not description:
            description = getattr(
                self.current_task,
                "description",
                None,
            )

        if not description:
            raise RuntimeError(
                "Active task has no description."
            )

        self._progress("Understanding task...")
        understanding = (
            self.task_understanding.understand(
                description
            )
        )
        self._progress(
            f"Task understanding returned (success={getattr(understanding, 'success', False)})."
        )

        if not getattr(
            understanding,
            "success",
            False,
        ):
            raise RuntimeError(
                "Task understanding failed: "
                + getattr(
                    understanding,
                    "message",
                    "Unknown error.",
                )
            )

        self.current_understanding = understanding

        return understanding

    # =============================================================
    # PLAN CREATION
    # =============================================================

    def create_plan(self) -> Any:
        """
        Create an executable plan for the current task.
        """

        if self.current_task is None:
            raise RuntimeError(
                "No active task. Call submit_task() first."
            )

        if self.current_understanding is None:
            self.understand_task()

        description = self.current_task_description

        if not description:
            description = getattr(
                self.current_task,
                "description",
                None,
            )

        if not description:
            raise RuntimeError(
                "Active task has no description."
            )

        understood_steps = getattr(
            self.current_understanding,
            "steps",
            [],
        )

        self._progress(
            f"Creating plan from {len(understood_steps)} understood step(s)..."
        )
        plan = self.planner.create_plan(
            description,
            understood_steps,
        )
        self._progress(
            f"Planner returned (success={getattr(plan, 'success', True)}, "
            f"steps={len(getattr(plan, 'steps', []) or [])})."
        )

        if not getattr(
            plan,
            "success",
            True,
        ):
            raise RuntimeError(
                "Planning failed: "
                + getattr(
                    plan,
                    "message",
                    "Unknown error.",
                )
            )

        self.current_plan = plan

        plan_steps = getattr(
            plan,
            "steps",
            [],
        )

        self.task_manager.add_plan_steps(
            self.current_task.task_id,
            plan_steps,
        )

        return plan

    # =============================================================
    # STATE MANAGEMENT
    # =============================================================

    def get_state(self) -> AgentState:
        """
        Return the current agent state.
        """

        return self.state_machine.get_state()

    def _transition(self, new_state: AgentState):
        """
        Perform a validated state transition.
        """

        return self.state_machine.transition(
            new_state
        )

    # =============================================================
    # OBSERVATION
    # =============================================================

    def observe(self):
        """
        Capture and analyze the current Windows screen.
        """

        observation = (
            self.observation_manager.observe()
        )

        self.last_observation = observation

        return observation

    # =============================================================
    # LAST OBSERVATION
    # =============================================================

    def get_last_observation(self):
        """
        Return the latest observation.
        """

        return self.last_observation

    # =============================================================
    # CONVERT OBSERVATION FOR DECISION ENGINE
    # =============================================================

    def _observation_to_dict(
        self,
        observation: Any,
    ) -> Dict[str, Any]:
        """
        Convert a VisualState or dictionary observation into
        the dictionary format expected by DecisionEngine.
        """

        if observation is None:
            return {}

        if isinstance(observation, dict):
            return observation

        if hasattr(
            observation,
            "to_dict",
        ):
            try:
                result = observation.to_dict()

                if isinstance(result, dict):
                    return result

            except Exception:
                pass

        result: Dict[str, Any] = {}

        attributes = [
            "screenshot_path",
            "screen_width",
            "screen_height",
            "active_window",
            "window_titles",
            "visible_text",
            "elements",
            "metadata",
        ]

        for attribute in attributes:

            if hasattr(
                observation,
                attribute,
            ):

                value = getattr(
                    observation,
                    attribute,
                )

                if attribute == "screenshot_path":
                    value = str(value)

                result[attribute] = value

        return result

    # =============================================================
    # GET NEXT ACTION
    # =============================================================

    def get_next_action(
        self,
        observation: Optional[Any] = None,
    ):
        """
        Determine the next action using the DecisionEngine.
        """

        if self.current_task is None:
            raise RuntimeError(
                "No active task. Call submit_task() first."
            )

        if self.current_plan is None:
            raise RuntimeError(
                "No active plan. Call create_plan() first."
            )

        if observation is not None:
            self.last_observation = observation

        current_observation = (
            observation
            if observation is not None
            else self.last_observation
        )

        current_step = (
            self._get_current_plan_step()
        )

        if current_step is None:
            return None

        task_description = (
            self.current_task_description
        )

        if not task_description:
            task_description = getattr(
                self.current_task,
                "description",
                "",
            )

        if not isinstance(
            task_description,
            str,
        ):
            task_description = str(
                task_description
            )

        computer_state = (
            self._observation_to_dict(
                current_observation
            )
        )

        history = self.execution_history

        decision = (
            self.decision_engine.choose_action(
                task_description,
                current_step,
                computer_state,
                history,
            )
        )

        return decision

    # =============================================================
    # EXECUTE ACTION
    # =============================================================

    def execute_action(
        self,
        action: Dict[str, Any],
        expected: Optional[Dict[str, Any]] = None,
    ):
        """
        Execute one action through AgentExecutor.

        AgentExecutor performs verification and recovery
        associated with the action.
        """

        if action is None:
            raise ValueError(
                "Action cannot be None."
            )

        if expected is None:
            expected = {}

        result = self.executor.execute(
            action=action,
            expected=expected,
        )

        history_entry = {
            "action": action,
            "result": result,
            "expected": expected,
        }

        self.execution_history.append(
            history_entry
        )

        return result

    # =============================================================
    # EXTRACT RESULT SUCCESS
    # =============================================================

    @staticmethod
    def _result_success(result: Any) -> bool:
        """
        Safely extract success from a result object or dictionary.
        """

        if isinstance(
            result,
            dict,
        ):
            return bool(
                result.get(
                    "success",
                    False,
                )
            )

        return bool(
            getattr(
                result,
                "success",
                False,
            )
        )

    # =============================================================
    # EXTRACT DECISION ACTION
    # =============================================================

    @staticmethod
    def _extract_decision_action(
        decision: Any,
    ):
        """
        Extract action and expected result from a decision.

        Supports both DecisionResult objects and dictionaries.
        """

        action = None
        expected = None

        if isinstance(
            decision,
            dict,
        ):

            action = decision.get(
                "action"
            )

            expected = decision.get(
                "expected"
            )

            if (
                action is not None
                and isinstance(action, str)
            ):
                action = decision

        else:

            action = getattr(
                decision,
                "action",
                None,
            )

            expected = getattr(
                decision,
                "expected",
                None,
            )

        return action, expected

    # =============================================================
    # RUN ONE COMPLETE STEP
    # =============================================================

    def run_step(self):
        """
        Run one complete:

            Observe → Decide → Execute

        cycle.

        AgentExecutor handles action verification
        and action-level recovery.
        """

        if self.current_task is None:
            raise RuntimeError(
                "No active task. Call submit_task() first."
            )

        if self.current_plan is None:
            raise RuntimeError(
                "No active plan. Call create_plan() first."
            )

        # ---------------------------------------------------------
        # 1. Observe
        # ---------------------------------------------------------

        self._progress("Capturing screen observation...")
        observation = self.observe()
        self._progress("Screen observation completed.")

        # ---------------------------------------------------------
        # 2. Decide
        # ---------------------------------------------------------

        self._progress("Choosing the next action...")
        decision = self.get_next_action(
            observation=observation
        )
        self._progress(
            f"Decision returned (has_decision={decision is not None})."
        )

        if decision is None:

            return {
                "success": False,
                "message": "No action available.",
                "observation": observation,
                "decision": None,
                "action": None,
                "expected": None,
                "result": None,
            }

        # ---------------------------------------------------------
        # 3. Extract decision
        # ---------------------------------------------------------

        action, expected = (
            self._extract_decision_action(
                decision
            )
        )

        if action is None:

            return {
                "success": False,
                "message": (
                    "Decision did not contain an action."
                ),
                "observation": observation,
                "decision": decision,
                "action": None,
                "expected": expected,
                "result": None,
            }

        # ---------------------------------------------------------
        # 4. Execute
        # ---------------------------------------------------------

        self._progress(
            f"Executing action: {action.get('action', 'unknown') if isinstance(action, dict) else action!r}..."
        )
        result = self.execute_action(
            action,
            expected,
        )
        self._progress("Action execution returned.")

        success = self._result_success(
            result
        )

        return {
            "success": success,
            "observation": observation,
            "decision": decision,
            "action": action,
            "expected": expected,
            "result": result,
        }

    # =============================================================
    # AUTONOMOUS RUN
    # =============================================================

    def run(
        self,
        description: Optional[str] = None,
        max_steps: int = 50,
    ):
        """
        Run the complete autonomous agent loop.
        """

        self._progress("run() entered.")

        if max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero."
            )

        # ---------------------------------------------------------
        # Task setup
        # ---------------------------------------------------------

        if description is not None:
            self.submit_task(
                description
            )

        if self.current_task is None:
            raise RuntimeError(
                "No active task. Provide a task description "
                "or call submit_task() first."
            )

        # ---------------------------------------------------------
        # Database task record
        # ---------------------------------------------------------

        task_description = self.current_task.description

        if not task_description:
                raise RuntimeError(
                    "Active task has no description."
    )
        self._progress("Saving task record to the database...")
        task_id = self.database.create_task(
            task_description
        )
        self._progress(f"Task record saved (task_id={task_id!r}).")

        # ---------------------------------------------------------
        # Reset state machine if necessary
        # ---------------------------------------------------------

        if self.state_machine.get_state() != AgentState.IDLE:
            self.state_machine.reset()

        self.run_history = []

        # ---------------------------------------------------------
        # Planning phase
        # ---------------------------------------------------------

        try:
            self._progress("Entering planning stage...")
            self._transition(
                AgentState.PLANNING
            )

            self.run_history.append(
                {
                    "state": AgentState.PLANNING.value,
                    "message": "Creating task plan.",
                }
            )

            self.create_plan()
            self._progress("Planning stage completed.")

        except Exception as error:
            self._handle_run_failure(
                error,
                "Planning failed.",
            )

            self.database.update_task(
                task_id,
                "failed",
                False,
                str(error),
            )

            return {
                "success": False,
                "state": self.get_state().value,
                "message": str(error),
                "history": self.run_history,
                "execution_history": (
                    self.execution_history
                ),
            }

        # ---------------------------------------------------------
        # Main execution loop
        # ---------------------------------------------------------

        for step_number in range(
            1,
            max_steps + 1,
        ):
            self._progress(f"Starting execution loop step {step_number}/{max_steps}...")
            current_step = (
                self._get_current_plan_step()
            )

            # -----------------------------------------------------
            # No remaining plan steps
            # -----------------------------------------------------

            if current_step is None:
                if self.state_machine.get_state() == (
                    AgentState.EXECUTING
                ):
                    self._transition(
                        AgentState.VERIFYING
                    )

                if self.state_machine.get_state() == (
                    AgentState.VERIFYING
                ):
                    self._transition(
                        AgentState.COMPLETED
                    )

                self.run_history.append(
                    {
                        "state": AgentState.COMPLETED.value,
                        "message": "All plan steps completed.",
                    }
                )

                self.database.update_task(
                    task_id,
                    "completed",
                    True,
                    "Task completed successfully.",
                )

                return {
                    "success": True,
                    "state": self.get_state().value,
                    "message": (
                        "Task completed successfully."
                    ),
                    "history": self.run_history,
                    "execution_history": (
                        self.execution_history
                    ),
                }

            # -----------------------------------------------------
            # Transition to EXECUTING
            # -----------------------------------------------------

            if self.state_machine.get_state() == (
                AgentState.PLANNING
            ):
                self._transition(
                    AgentState.EXECUTING
                )

            elif self.state_machine.get_state() == (
                AgentState.RECOVERING
            ):
                self._transition(
                    AgentState.EXECUTING
                )

            self.run_history.append(
                {
                    "state": AgentState.EXECUTING.value,
                    "step_number": step_number,
                    "plan_step": current_step,
                }
            )

            # -----------------------------------------------------
            # Execute one step
            # -----------------------------------------------------

            try:
                self._progress(f"Running observe/decide/execute cycle for step {step_number}...")
                cycle = self.run_step()
                self._progress(f"Cycle {step_number} returned (success={cycle.get('success', False)}).")

            except Exception as error:
                self._handle_run_failure(
                    error,
                    "Execution failed.",
                )

                self.database.update_task(
                    task_id,
                    "failed",
                    False,
                    str(error),
                )

                return {
                    "success": False,
                    "state": self.get_state().value,
                    "message": str(error),
                    "history": self.run_history,
                    "execution_history": (
                        self.execution_history
                    ),
                }

            # -----------------------------------------------------
            # Verification state
            # -----------------------------------------------------

            if self.state_machine.get_state() == (
                AgentState.EXECUTING
            ):
                self._transition(
                    AgentState.VERIFYING
                )

            cycle_success = bool(
                cycle.get(
                    "success",
                    False,
                )
            )

            self.run_history.append(
                {
                    "state": AgentState.VERIFYING.value,
                    "step_number": step_number,
                    "success": cycle_success,
                    "cycle": cycle,
                }
            )

            # -----------------------------------------------------
            # Successful step
            # -----------------------------------------------------

            if cycle_success:
                self._advance_task_step()

                next_step = (
                    self._get_current_plan_step()
                )

                if next_step is None:
                    self._transition(
                        AgentState.COMPLETED
                    )

                    self.run_history.append(
                        {
                            "state": (
                                AgentState.COMPLETED.value
                            ),
                            "message": (
                                "Task completed successfully."
                            ),
                        }
                    )

                    self.database.update_task(
                        task_id,
                        "completed",
                        True,
                        "Task completed successfully.",
                    )

                    return {
                        "success": True,
                        "state": (
                            self.get_state().value
                        ),
                        "message": (
                            "Task completed successfully."
                        ),
                        "history": self.run_history,
                        "execution_history": (
                            self.execution_history
                        ),
                    }

                self._transition(
                    AgentState.EXECUTING
                )

                continue

            # -----------------------------------------------------
            # Step failed
            # -----------------------------------------------------

            self._transition(
                AgentState.RECOVERING
            )

            # AgentExecutor already performs action-level retries,
            # verification, and recovery. If it returns failure here,
            # repeating the same controller plan step would execute
            # the same failed action again.
            result = cycle.get("result") or {}
            failure_message = cycle.get(
                "message",
                "The current plan step failed.",
            )

            if isinstance(result, dict):
                final_result = result.get("final_result") or {}
                action_result = (
                    final_result.get("action_result")
                    if isinstance(final_result, dict)
                    else None
                )

                if action_result is not None:
                    failure_message = getattr(
                        action_result,
                        "message",
                        failure_message,
                    )

                recovery = result.get("recovery")
                if isinstance(recovery, dict):
                    recovery_reason = recovery.get("reason")
                    if recovery_reason:
                        failure_message = (
                            f"{failure_message} "
                            f"Recovery: {recovery_reason}"
                        )

            self.run_history.append(
                {
                    "state": AgentState.RECOVERING.value,
                    "step_number": step_number,
                    "message": (
                        "Action-level recovery was exhausted. "
                        "Stopping instead of repeating the same "
                        "plan step."
                    ),
                    "failure": failure_message,
                }
            )

            self._handle_run_failure(
                RuntimeError(failure_message),
                "Step failed after action-level recovery.",
            )

            self.database.update_task(
                task_id,
                "failed",
                False,
                failure_message,
            )

            return {
                "success": False,
                "state": self.get_state().value,
                "message": failure_message,
                "history": self.run_history,
                "execution_history": self.execution_history,
            }

        # ---------------------------------------------------------
        # Maximum step limit reached
        # ---------------------------------------------------------

        error = (
            f"Maximum execution steps ({max_steps}) "
            f"reached before task completion."
        )

        self._handle_run_failure(
            RuntimeError(error),
            error,
        )

        self.database.update_task(
            task_id,
            "failed",
            False,
            error,
        )

        return {
            "success": False,
            "state": self.get_state().value,
            "message": error,
            "history": self.run_history,
            "execution_history": (
                self.execution_history
            ),
        }

    # =============================================================
    # ADVANCE TASK STEP
    # =============================================================

    def _advance_task_step(self):
        """
        Advance the current task to the next plan step.

        The controller maintains current_step_index as the source
        of truth.
        """

        if self.current_task is None:
            return False

        current_index = getattr(
            self.current_task,
            "current_step_index",
            0,
        )

        if not isinstance(
            current_index,
            int,
        ):
            current_index = 0

        next_index = current_index + 1

        # ---------------------------------------------------------
        # Advance controller task state.
        # ---------------------------------------------------------

        self.current_task.current_step_index = next_index

        return True

    # =============================================================
    # FAILURE HANDLING
    # =============================================================

    def _handle_run_failure(
        self,
        error: Exception,
        message: str,
    ):
        """
        Move the state machine into FAILED when possible.
        """

        current_state = (
            self.state_machine.get_state()
        )

        try:

            if current_state == AgentState.PLANNING:

                self._transition(
                    AgentState.FAILED
                )

            elif current_state == AgentState.EXECUTING:

                self._transition(
                    AgentState.FAILED
                )

            elif current_state == AgentState.VERIFYING:

                self._transition(
                    AgentState.FAILED
                )

            elif current_state == AgentState.RECOVERING:

                self._transition(
                    AgentState.FAILED
                )

        except Exception:
            pass

        self.run_history.append(
            {
                "state": AgentState.FAILED.value,
                "message": message,
                "error": str(error),
            }
        )

    # =============================================================
    # EXECUTION HISTORY
    # =============================================================

    def get_execution_history(self):
        """
        Return execution history for the current task.
        """

        return self.execution_history

    # =============================================================
    # RUN HISTORY
    # =============================================================

    def get_run_history(self):
        """
        Return state-machine/runtime history for the current run.
        """

        return self.run_history

    # =============================================================
    # RESET
    # =============================================================

    def reset(self):
        """
        Reset the controller to an empty state.
        """

        self.current_task = None
        self.current_task_description = None
        self.current_understanding = None
        self.current_plan = None
        self.last_observation = None
        self.execution_history = []
        self.run_history = []

        self.state_machine.reset()

    # =============================================================
    # CURRENT PLAN STEP
    # =============================================================

    def _get_current_plan_step(self):
        """
        Return the currently active plan step.

        Supports both:

        1. PlanResult objects:
               plan.steps

        2. Dictionary plans:
               plan["steps"]
        """

        if self.current_plan is None:
            return None

        # ---------------------------------------------------------
        # Extract steps
        # ---------------------------------------------------------

        if isinstance(
            self.current_plan,
            dict,
        ):

            steps = self.current_plan.get(
                "steps"
            )

        else:

            steps = getattr(
                self.current_plan,
                "steps",
                None,
            )

        if not steps:
            return None

        # ---------------------------------------------------------
        # Current step index
        # ---------------------------------------------------------

        index = getattr(
            self.current_task,
            "current_step_index",
            0,
        )

        if not isinstance(
            index,
            int,
        ):
            index = 0

        if index < 0:
            index = 0

        if index >= len(steps):
            return None

        return steps[index]