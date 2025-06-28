"""Base prompt templates and utilities."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class PromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    @abstractmethod
    def format(self, **kwargs) -> str:
        """Format the prompt with provided variables."""
        pass

    @property
    @abstractmethod
    def required_vars(self) -> list[str]:
        """Return list of required variables for this prompt."""
        pass


class StringPromptTemplate(PromptTemplate):
    """Simple string-based prompt template."""

    def __init__(self, template: str, required_vars: list[str]):
        self.template = template
        self._required_vars = required_vars

    def format(self, **kwargs) -> str:
        """Format the prompt template with provided variables."""
        # Validate required variables
        missing_vars = [var for var in self._required_vars if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        return self.template.format(**kwargs)

    @property
    def required_vars(self) -> list[str]:
        """Return list of required variables for this prompt."""
        return self._required_vars.copy()


def validate_prompt_variables(template: str, variables: Dict[str, Any]) -> None:
    """
    Validate that all required variables are provided for a prompt template.

    Args:
        template: The prompt template string
        variables: Dictionary of variables to validate

    Raises:
        ValueError: If required variables are missing
    """
    import re

    # Find all format placeholders in the template
    placeholders = re.findall(r"\{(\w+)\}", template)
    required_vars = list(set(placeholders))

    missing_vars = [var for var in required_vars if var not in variables]
    if missing_vars:
        raise ValueError(f"Missing required variables for prompt: {missing_vars}")
