#!/usr/bin/env python
# coding=utf-8

import os
import sys
import time
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv
import argparse

load_dotenv()
# Add the repository root to the path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smolagents import LiteLLMModel, DuckDuckGoSearchTool
from reflexion import ReflexionAgent

console = Console()


def run_agent_with_criteria(question, success_criteria, force_reflection=False):
    """
    Run the ReflexionAgent with specific success criteria.

    Args:
        question: The question to ask
        success_criteria: The success criteria to apply
        force_reflection: Whether to force a reflection by failing the first attempt
    """
    # Initialize the model using Gemini
    console.print("Initializing Gemini model...", style="yellow")
    model = LiteLLMModel(
        model_id="gemini/gemini-2.0-flash",  # Can also use "gemini/gemini-pro"
        api_key=os.environ.get("GEMINI_API_KEY"),
        temperature=0.7,
        max_tokens=1000,
    )

    # Create the tools
    search_tool = DuckDuckGoSearchTool()

    console.print("Success Criteria:", style="bold cyan")
    console.print(Panel(success_criteria, expand=False, style="cyan"))

    # Create a custom success criteria checker if we want to force reflection
    if force_reflection:
        # Create a wrapper class that will fail the first check
        class ForceReflectionAgent(ReflexionAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.first_check = True

            def _check_success_criteria(self, response):
                if self.first_check:
                    console.print("Forcing first attempt to fail for demonstration purposes...", style="bold red")
                    self.first_check = False
                    return False
                return super()._check_success_criteria(response)

        agent_class = ForceReflectionAgent
        console.print("Creating ReflexionAgent with FORCED first failure...", style="bold red")
    else:
        agent_class = ReflexionAgent
        console.print("Creating ReflexionAgent with success criteria...", style="yellow")

    # Create the agent
    agent = agent_class(
        tools=[search_tool],
        model=model,
        max_trials=3,  # Allow up to 3 attempts to solve the task
        max_reflections=2,  # Keep the 2 most recent reflections
        max_steps=4,  # Maximum steps per trial
        verbosity_level=4,  # Detailed logging
        track_metrics=True,  # Track performance metrics
        success_criteria=success_criteria,  # Add the success criteria
    )

    console.print(f"\nQuestion: [bold cyan]{question}[/bold cyan]\n", style="bold")
    console.print("Running ReflexionAgent with multiple trials if needed...", style="yellow")

    # Run the agent
    start_time = time.time()
    result = agent.run(question)
    end_time = time.time()

    # Print the final result
    console.print("\n[bold green]Final Answer:[/bold green]", style="bold")
    console.print(Panel(result["output"] if isinstance(result, dict) else result, expand=False))

    # Print metrics if available
    if agent.track_metrics:
        console.print("\n[bold green]Agent Metrics:[/bold green]", style="bold")
        metrics = agent.metrics
        console.print(f"Total trials: {len(metrics['trials'])}")
        console.print(f"Success rate: {metrics['success_rate']:.2%}")

        if metrics["errors"]:
            console.print("Error distribution:")
            for error_type, count in metrics["errors"].items():
                console.print(f"  - {error_type}: {count}")

    console.print(f"\nTotal time taken: {end_time - start_time:.2f} seconds", style="dim")

    # Print reflections if any
    if agent.reflections:
        console.print("\n[bold green]Agent Reflections:[/bold green]", style="bold")
        for i, reflection in enumerate(agent.reflections):
            console.print(Panel(reflection, title=f"Reflection {i + 1}", style="cyan"))

    # Print trial information if result is a dictionary
    if isinstance(result, dict):
        console.print(f"\nCompleted in {result['trials']} trial(s)", style="bold green")
        if result["success"]:
            console.print("Task completed successfully!", style="bold green")
        else:
            console.print(f"Task failed: {result.get('error', 'Unknown error')}", style="bold red")

    return agent, result


def main():
    """
    Run an example of the ReflexionAgent using the Gemini model to answer a complex question about climate change.
    This demonstrates how the agent can learn from failures and improve its answer through multiple trials
    based on success criteria.
    """
    parser = argparse.ArgumentParser(description="Run ReflexionAgent demo with different options")
    parser.add_argument(
        "--force-reflection", action="store_true", help="Force a reflection by failing the first attempt"
    )
    parser.add_argument("--very-strict", action="store_true", help="Use very strict success criteria")
    args = parser.parse_args()

    console.print(Panel.fit("Initializing ReflexionAgent Demo with Success Criteria", style="bold green"))

    # The question from the reflexion_langgraph example
    question = "How should we handle the climate crisis?"

    # Define success criteria for the agent's response
    if args.very_strict:
        # Very strict criteria that will likely cause natural failures
        success_criteria = """
        1. The answer MUST begin with a section explicitly titled "Introduction" that explains the climate crisis.
        2. The answer MUST include EXACTLY 5 specific strategies or approaches to address the climate crisis.
        3. Each strategy MUST be numbered and have its own heading.
        4. Each strategy explanation MUST be exactly 3 sentences long - no more, no less.
        5. The answer MUST cite at least 4 specific sources with complete URLs.
        6. The answer MUST end with a section titled "Conclusion" that summarizes the key points.
        7. The answer MUST use at least 3 statistical figures or percentages.
        8. The answer MUST mention at least 2 international climate agreements by name.
        """
    else:
        # Standard criteria
        success_criteria = """
        1. The answer must begin with a brief introduction to the climate crisis. It should have a section called "Introduction".
        2. The answer must include at least 3 specific strategies or approaches to address the climate crisis.
        3. Each strategy should be explained in 2-3 sentences.
        4. The answer must cite at least 2 specific sources with links or references.
        5. The answer should be well-structured with clear sections or bullet points.
        """

    run_agent_with_criteria(question, success_criteria, force_reflection=args.force_reflection)


if __name__ == "__main__":
    main()
