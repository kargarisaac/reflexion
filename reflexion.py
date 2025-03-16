import time
from typing import Callable, Dict, List, Optional

from logging import getLogger

from smolagents.models import (
    ChatMessage,
    MessageRole,
)
from smolagents.monitoring import (
    LogLevel,
)
from smolagents.tools import Tool
from smolagents.utils import (
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
)

from smolagents.agents import CodeAgent

logger = getLogger(__name__)

class ReflexionAgent(CodeAgent):
    """
    An agent that implements the Reflexion framework for language agents with verbal reinforcement learning.

    Reflexion converts binary or scalar feedback into verbal feedback in the form of textual
    summaries, which are added as additional context for future trials. This self-reflective
    feedback acts as a 'semantic gradient signal', helping the agent learn from prior mistakes.

    Args:
        tools (List[Tool]): List of tools that the agent can use.
        model (Callable): Model that will generate the agent's actions.
        max_trials (int): Maximum number of trials to attempt for a task.
        max_reflections (int): Maximum number of reflections to store in memory.
        reflection_prompt_template (str, optional): Template for generating error reflections.
        success_criteria (str, optional): User-defined success criteria for the output. If provided,
            the agent will check if the output meets these criteria and retry if not.
        track_metrics (bool): Whether to track performance metrics across trials.
        **kwargs: Additional arguments to pass to the parent CodeAgent.
    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        max_trials: int = 5,
        max_reflections: int = 3,
        reflection_prompt_template: Optional[str] = None,
        success_criteria: Optional[str] = None,
        track_metrics: bool = True,
        **kwargs,
    ):
        # Initialize these attributes first before the parent class initialization
        self.max_trials = max_trials
        self.max_reflections = max_reflections
        self.reflections = []
        self.trial_count = 0
        self.track_metrics = track_metrics
        self.metrics = {"trials": [], "errors": {}, "success_rate": 0.0} if track_metrics else None
        self.success_criteria = success_criteria

        # Default reflection prompt for error handling
        self.reflection_prompt_template = (
            reflection_prompt_template
            or """
        You are helping an agent reflect on its failed attempt to solve a task.
        
        Task: {task}
        
        Here is what the agent did:
        {memory_text}
        
        Error encountered: {error}
        Error type: {error_type}
        
        Please provide a thoughtful reflection on what went wrong and how the agent could improve on its next attempt.
        Focus on specific mistakes and actionable improvements. Be concise but thorough.
        Identify concrete lessons learned and clear steps to take on the next attempt.
        
        The reflection should follow this format:
        1. What went wrong: [specific issue identified]
        2. Why it happened: [analysis of the cause]
        3. How to fix it: [concrete actionable steps]
        4. Key lesson: [primary takeaway for future attempts]
        """
        )

        # Then call the parent's __init__
        super().__init__(tools=tools, model=model, **kwargs)

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: Optional[List[str]] = None,
        additional_args: Optional[Dict] = None,
    ):
        """
        Run the agent with the Reflexion framework, allowing multiple trials with verbal reflection.

        Args:
            task (str): The task to perform.
            stream (bool): Whether to run in streaming mode.
            reset (bool): Whether to reset the conversation history.
            images (List[str], optional): List of image paths.
            additional_args (Dict, optional): Additional arguments for the task.

        Returns:
            Dict containing the output, success status, trials, reflections, and metrics.
        """
        # Reset reflections and trial count only if we're starting a new task
        if reset:
            self.reflections = []
            self.trial_count = 0
            if self.track_metrics:
                self.metrics = {"trials": [], "errors": {}, "success_rate": 0.0}

        successful = False
        result = None

        # Add success criteria to task if provided
        full_task = task
        if self.success_criteria:
            full_task = f"{task}\n\nYour response must meet these criteria:\n{self.success_criteria}"

        while self.trial_count < self.max_trials:
            self.trial_count += 1
            self.logger.log_rule(f"Trial {self.trial_count}/{self.max_trials}", level=LogLevel.INFO)

            try:
                # First trial uses reset=True, subsequent trials use reset=False
                trial_reset = reset if self.trial_count == 1 else False

                # Run a trial
                result = super().run(
                    task=full_task, stream=stream, reset=trial_reset, images=images, additional_args=additional_args
                )

                # Check if result meets success criteria (if any)
                if self.success_criteria and not self._check_success_criteria(result):
                    # Criteria not met, generate reflection
                    self.logger.log(
                        f"Trial {self.trial_count} response doesn't meet success criteria", level=LogLevel.INFO
                    )

                    # Generate a reflection on the criteria failure
                    reflection = self._generate_criteria_reflection(task, result)
                    self.reflections.append(reflection)

                    # Keep only the most recent max_reflections
                    if len(self.reflections) > self.max_reflections:
                        self.reflections = self.reflections[-self.max_reflections :]

                    # Update the system prompt for the next trial
                    self.system_prompt = self.initialize_system_prompt()
                    self.logger.log(f"Success Criteria Reflection: {reflection}", level=LogLevel.INFO)

                    # Track in metrics
                    if self.track_metrics:
                        self.metrics["trials"].append(
                            {"trial": self.trial_count, "status": "failure", "error_type": "criteria_not_met"}
                        )
                        self.metrics["errors"]["criteria_not_met"] = (
                            self.metrics["errors"].get("criteria_not_met", 0) + 1
                        )

                    # Small delay between trials
                    time.sleep(1)
                    continue

                # If we got here without error or criteria failure, we succeeded
                self.logger.log(f"Trial {self.trial_count} succeeded.", level=LogLevel.INFO)

                # Track successful case
                if self.track_metrics:
                    self.metrics["trials"].append({"trial": self.trial_count, "status": "success"})
                    total_trials = len(self.metrics["trials"])
                    successful_trials = sum(1 for t in self.metrics["trials"] if t["status"] == "success")
                    self.metrics["success_rate"] = successful_trials / total_trials

                # Generate success reflection if this wasn't the first trial
                if self.trial_count > 1:
                    success_reflection = self._generate_success_reflection(task)
                    self.logger.log(f"Success Reflection: {success_reflection}", level=LogLevel.INFO)

                successful = True
                break

            except Exception as e:
                # Identify error type
                error_type = self._categorize_error(e)
                error_msg = str(e)

                # Track error
                if self.track_metrics:
                    self.metrics["trials"].append(
                        {"trial": self.trial_count, "status": "failure", "error_type": error_type}
                    )
                    self.metrics["errors"][error_type] = self.metrics["errors"].get(error_type, 0) + 1

                # Generate a reflection on the failure
                self.logger.log(
                    f"Trial {self.trial_count} failed with error type {error_type}: {error_msg}", level=LogLevel.INFO
                )

                reflection = self._generate_reflection(task, e, error_type)
                self.reflections.append(reflection)

                # Keep only the most recent max_reflections
                if len(self.reflections) > self.max_reflections:
                    self.reflections = self.reflections[-self.max_reflections :]

                # If this was the last trial, re-raise the error
                if self.trial_count >= self.max_trials:
                    if self.track_metrics:
                        self.logger.log(f"Metrics after all trials: {self.metrics}", level=LogLevel.INFO)
                    raise e

                # Update the system prompt for the next trial
                self.system_prompt = self.initialize_system_prompt()

                self.logger.log(f"Reflection: {reflection}", level=LogLevel.INFO)

                # Small delay between trials to ensure clean separation
                time.sleep(1)

        if successful:
            if self.track_metrics:
                self.logger.log(f"Metrics after successful completion: {self.metrics}", level=LogLevel.INFO)

            # Prepare a comprehensive return value
            return {
                "output": result,
                "success": True,
                "trials": self.trial_count,
                "reflections": self.reflections,
                "metrics": self.metrics if self.track_metrics else None,
            }

        # We shouldn't reach this point due to the error re-raising above,
        # but just in case, return the final result
        return {
            "output": None,
            "success": False,
            "trials": self.trial_count,
            "reflections": self.reflections,
            "error": "Max trials exceeded without success",
            "metrics": self.metrics if self.track_metrics else None,
        }

    def _check_success_criteria(self, response: str) -> bool:
        """
        Ask the model to evaluate if the response meets the success criteria.

        Args:
            response: The response text to check

        Returns:
            Boolean indicating if criteria are met
        """
        if not self.success_criteria:
            return True

        prompt = f"""
        You are an objective evaluator. Your task is to determine if the following response meets all the specified criteria.
        
        Success criteria:
        {self.success_criteria}
        
        Response to evaluate:
        {response}
        
        Does this response fully satisfy ALL the criteria listed above? Answer with ONLY 'Yes' or 'No'.
        """

        try:
            evaluation = self.model([{"role": MessageRole.USER, "content": [{"type": "text", "text": prompt}]}])
            result = evaluation.content.strip().lower()
            return "yes" in result and "no" not in result
        except:
            # If evaluation fails, assume criteria are not met
            return False

    def _generate_criteria_reflection(self, task: str, response: str) -> str:
        """
        Generate a reflection based on not meeting success criteria.

        Args:
            task: The original task
            response: The agent's response

        Returns:
            A reflection on how to improve the response
        """
        prompt = f"""
        You are helping an agent improve its response to better meet specific criteria.
        
        Task: {task}
        
        Success criteria that must be met:
        {self.success_criteria}
        
        Here is the agent's current response:
        {response}
        
        Please provide a thoughtful reflection on how the agent could improve its response to meet all the criteria.
        Focus on specific suggestions and actionable improvements. Be concise but thorough.
        
        The reflection should follow this format:
        1. What's missing: [analysis of what criteria aren't being met]
        2. Why it matters: [explanation of why meeting these criteria is important]
        3. How to improve: [concrete suggestions for improving the response]
        4. Key lesson: [primary takeaway for future attempts]
        """

        # Generate reflection using the model
        try:
            reflection_message = self.model(
                [{"role": MessageRole.USER, "content": [{"type": "text", "text": prompt}]}]
            )
            return reflection_message.content
        except:
            return "Unable to generate reflection on meeting success criteria."

    def initialize_system_prompt(self):
        """
        Initialize the system prompt, including any reflections from previous trials.
        """
        base_prompt = super().initialize_system_prompt()

        # Add reflections if we have any
        if self.reflections:
            reflection_text = "\n\n".join(
                [f"Reflection from previous trial {i + 1}: {r}" for i, r in enumerate(self.reflections)]
            )

            base_prompt += f"\n\n### Reflexion Instructions\nYou have attempted this task before and failed. Here are reflections on previous failures to guide you:\n{reflection_text}\n\nPlease use these reflections to improve your approach and avoid repeating the same mistakes. Before writing any code, think about how these reflections apply to the current task."

        return base_prompt

    def _categorize_error(self, error: Exception) -> str:
        """
        Categorize the error type to provide more specific feedback.

        Args:
            error (Exception): The exception that occurred during execution.

        Returns:
            str: A categorized error type
        """
        error_str = str(error)

        if isinstance(error, AgentParsingError):
            return "code_parsing_error"
        elif isinstance(error, AgentGenerationError):
            return "generation_error"
        elif isinstance(error, AgentExecutionError):
            if "Import of " in error_str and " is not allowed" in error_str:
                return "unauthorized_import_error"
            elif "IndentationError" in error_str or "SyntaxError" in error_str:
                return "syntax_error"
            elif "NameError" in error_str:
                return "name_error"
            elif "IndexError" in error_str or "KeyError" in error_str:
                return "index_key_error"
            elif "TypeError" in error_str or "ValueError" in error_str:
                return "type_value_error"
            else:
                return "runtime_error"
        elif isinstance(error, AgentMaxStepsError):
            return "max_steps_error"
        else:
            return "unknown_error"

    def _generate_reflection(self, task: str, error: Exception, error_type: str) -> str:
        """
        Generate a verbal reflection on why the agent failed at a task.

        Args:
            task: The task the agent failed at
            error: The error that occurred
            error_type: The categorized error type

        Returns:
            A verbal reflection on what went wrong and how to fix it
        """
        PROMPT = f"""You are an expert software developer debugging a failed execution.
Task: {task}

Error: {str(error)}
Error type: {error_type}

Please perform a detailed analysis of what went wrong. Focus on:
1. Root cause of the failure
2. What assumptions or approach led to this error
3. How to fix the issue
4. A better approach to solve the original task

Your reflection should be thorough yet concise (aim for 3-5 sentences). Be specific with suggestions for improvement."""

        # Generate reflection using the model
        try:
            reflection_message = self.model(
                [{"role": MessageRole.USER, "content": [{"type": "text", "text": PROMPT}]}]
            )
            return reflection_message.content
        except:
            return "Unable to generate reflection on the error."

    def _generate_success_reflection(self, task: str) -> str:
        """
        Generate a reflection on what led to success after previous failures.

        Args:
            task: The task that was successfully completed

        Returns:
            A reflection on what led to success
        """
        previous_reflections = "\n".join([f"- {r}" for r in self.reflections])

        PROMPT = f"""You are an expert software developer analyzing a successful task completion after previous failures.
Task: {task}

Previous reflections on failures:
{previous_reflections}

Given these previous reflections, please analyze what likely led to the eventual success.
Focus on:
1. Which key insights from previous reflections were most helpful
2. What changes in approach likely led to success
3. What lessons can be applied to future similar tasks

Your analysis should be thorough yet concise (aim for 3-5 sentences). Be specific with insights gained."""

        # Generate success reflection using the model
        try:
            success_reflection_message = self.model(
                [{"role": MessageRole.USER, "content": [{"type": "text", "text": PROMPT}]}]
            )
            return success_reflection_message.content
        except:
            return "Unable to generate success reflection."