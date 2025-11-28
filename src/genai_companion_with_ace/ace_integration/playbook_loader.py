"""Utilities for loading ACE playbooks and exposing prompt context."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from genai_companion_with_ace.config import CompanionConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class PlaybookContext:
    """Structured playbook contents used to prime the LLM."""

    version: str
    system_instructions: str
    heuristics: list[str] = field(default_factory=list)
    heuristic_ids: list[str] = field(default_factory=list)  # Track heuristic IDs for usage tracking
    examples: list[dict[str, str]] = field(default_factory=list)

    def to_prompt_block(self) -> str:
        """Render the playbook context into a textual block."""
        segments: list[str] = [self.system_instructions.strip()]
        if self.heuristics:
            segments.append("Heuristics:")
            segments.extend(f"- {heuristic}" for heuristic in self.heuristics)
        if self.examples:
            segments.append("Examples:")
            for example in self.examples:
                user = example.get("user")
                assistant = example.get("assistant")
                if user:
                    segments.append(f"User: {user}")
                if assistant:
                    segments.append(f"Assistant: {assistant}")
        return "\n".join(segments)
    
    def get_heuristic_ids(self) -> list[str]:
        """Get list of heuristic IDs that are currently active."""
        return self.heuristic_ids if self.heuristic_ids else [f"h{i+1:03d}" for i in range(len(self.heuristics))]


class PlaybookLoader:
    """Loads the most recent ACE-generated playbook or falls back to a default."""

    def __init__(
        self,
        playbook_dir: Path,
        *,
        default_path: Path | None = None,
    ) -> None:
        self._playbook_dir = playbook_dir
        self._default_path = default_path
        self._playbook_dir.mkdir(parents=True, exist_ok=True)

    def load_latest(self) -> PlaybookContext:
        """Load the most recent playbook from the project's playbook directory.
        
        Priority order:
        1. ACE-generated playbooks (playbook_*.yaml) - most recent first
        2. default_playbook.yaml if it exists
        3. Hardcoded default playbook
        """
        candidates = self.list_available()
        if candidates:
            latest = candidates[-1]
            LOGGER.info("Loading playbook from %s", latest)
            return self._parse_playbook(latest)

        # If no playbooks found, use hardcoded default
        LOGGER.warning("No playbook files found. Using fallback instructions tailored for GenAI Companion.")
        return self._get_default_playbook()

    def _parse_playbook(self, path: Path) -> PlaybookContext:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        context = payload.get("context", {})
        
        # Extract heuristics - they can be either strings or dicts with id/rule
        raw_heuristics = context.get("heuristics", [])
        heuristic_rules: list[str] = []
        heuristic_ids: list[str] = []
        
        for heuristic in raw_heuristics:
            if isinstance(heuristic, dict):
                # Extract rule and ID from dict format
                rule = heuristic.get("rule", "")
                heuristic_id = heuristic.get("id", "")
                if rule:
                    heuristic_rules.append(rule)
                if heuristic_id:
                    heuristic_ids.append(heuristic_id)
            elif isinstance(heuristic, str):
                # Simple string format
                heuristic_rules.append(heuristic)
                # Generate ID based on index
                heuristic_ids.append(f"h{len(heuristic_rules):03d}")
        
        return PlaybookContext(
            version=str(payload.get("version", path.stem)),
            system_instructions=context.get(
                "system_instructions",
                "You are the IBM Gen AI Study Companion. Provide grounded, citation-rich answers.",
            ),
            heuristics=heuristic_rules,
            heuristic_ids=heuristic_ids,
            examples=context.get("few_shot_examples", context.get("examples", [])),
        )

    def list_available(self) -> list[Path]:
        """List all available playbooks in the project directory, sorted by modification time.
        
        Includes both ACE-generated playbooks (playbook_*.yaml) and the default playbook.
        """
        playbooks = list(self._playbook_dir.glob("playbook_*.yaml"))
        # Also include default_playbook.yaml if it exists
        default_playbook = self._playbook_dir / "default_playbook.yaml"
        if default_playbook.exists() and default_playbook not in playbooks:
            playbooks.append(default_playbook)
        return sorted(playbooks, key=lambda path: path.stat().st_mtime)

    def _get_default_playbook(self) -> PlaybookContext:
        """Return a default playbook tailored for the GenAI Companion project."""
        return PlaybookContext(
            version="default",
            system_instructions=(
                "You are the IBM Gen AI Study Companion, an intelligent assistant designed to help students "
                "learn about Generative AI and related technologies. Your role is to provide clear, accurate, "
                "and pedagogically sound explanations based on official course materials.\n\n"
                "Key principles:\n"
                "- Always ground your answers in the retrieved course materials and cite sources explicitly\n"
                "- Use clear, step-by-step explanations for complex concepts\n"
                "- Provide practical examples and code snippets when relevant\n"
                "- Admit when information is not available in the course materials\n"
                "- Encourage learning by asking follow-up questions when appropriate"
            ),
            heuristics=[
                "When explaining technical concepts, break them down into digestible steps with clear explanations for each part.",
                "For coding questions, provide complete, runnable code examples with comments explaining key parts.",
                "Always cite sources using the format '[Source: Course X - Module Y]' when referencing course materials.",
                "If a question requires information not in the retrieved context, clearly state this and suggest what the student might need to look up.",
                "For conceptual questions, use analogies and real-world examples to aid understanding.",
                "When discussing best practices, explain both the 'what' and the 'why' to deepen understanding.",
            ],
            heuristic_ids=["h001", "h002", "h003", "h004", "h005", "h006"],
            examples=[
                {
                    "user": "What is the difference between supervised and unsupervised learning?",
                    "assistant": (
                        "Supervised and unsupervised learning are two fundamental approaches in machine learning:\n\n"
                        "**Supervised Learning:**\n"
                        "- Uses labeled training data (input-output pairs)\n"
                        "- The model learns to map inputs to known outputs\n"
                        "- Examples: classification (spam detection), regression (price prediction)\n"
                        "- Goal: Learn a function that can predict outputs for new inputs\n\n"
                        "**Unsupervised Learning:**\n"
                        "- Uses unlabeled data (only inputs, no outputs)\n"
                        "- The model finds patterns or structures in the data\n"
                        "- Examples: clustering (customer segmentation), dimensionality reduction\n"
                        "- Goal: Discover hidden patterns or groupings in the data\n\n"
                        "[Source: Course 1 - Introduction to Machine Learning]"
                    ),
                },
                {
                    "user": "How do I implement a simple neural network in Python?",
                    "assistant": (
                        "Here's a simple neural network implementation using PyTorch:\n\n"
                        "```python\n"
                        "import torch\n"
                        "import torch.nn as nn\n"
                        "import torch.optim as optim\n\n"
                        "# Define the neural network architecture\n"
                        "class SimpleNN(nn.Module):\n"
                        "    def __init__(self, input_size, hidden_size, output_size):\n"
                        "        super(SimpleNN, self).__init__()\n"
                        "        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer\n"
                        "        self.relu = nn.ReLU()  # Activation function\n"
                        "        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer\n\n"
                        "    def forward(self, x):\n"
                        "        x = self.fc1(x)\n"
                        "        x = self.relu(x)\n"
                        "        x = self.fc2(x)\n"
                        "        return x\n\n"
                        "# Example usage\n"
                        "model = SimpleNN(input_size=784, hidden_size=128, output_size=10)\n"
                        "criterion = nn.CrossEntropyLoss()\n"
                        "optimizer = optim.Adam(model.parameters(), lr=0.001)\n"
                        "```\n\n"
                        "This creates a two-layer neural network with ReLU activation. "
                        "The forward method defines how data flows through the network.\n\n"
                        "[Source: Course 3 - Neural Networks and Deep Learning]"
                    ),
                },
                {
                    "user": "What is attention mechanism in transformers?",
                    "assistant": (
                        "The attention mechanism is a key innovation in transformer architectures that allows "
                        "the model to focus on different parts of the input when processing each element.\n\n"
                        "**Key Concepts:**\n"
                        "1. **Self-Attention**: Each token in a sequence can attend to all other tokens, "
                        "including itself, to build contextualized representations\n"
                        "2. **Query, Key, Value (QKV)**: The mechanism uses three learned projections:\n"
                        "   - Query: What am I looking for?\n"
                        "   - Key: What do I represent?\n"
                        "   - Value: What information do I contain?\n"
                        "3. **Scaled Dot-Product Attention**: Computes attention scores and applies them to values\n\n"
                        "**Why it matters:** Unlike RNNs that process sequentially, attention allows parallel "
                        "processing and direct connections between distant tokens, making transformers highly "
                        "efficient and effective for long sequences.\n\n"
                        "[Source: Course 12 - Transformer Architecture]"
                    ),
                },
            ],
        )

    @classmethod
    def from_config(cls, config_path: Path) -> PlaybookLoader:
        """Create a PlaybookLoader from configuration file.
        
        Playbooks are now self-contained within this project in the outputs/ace_playbooks directory.
        """
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        ace_config = config.get("ace", {}) if isinstance(config, dict) else {}
        playbook_dir = Path(ace_config.get("playbook_output_dir", "outputs/ace_playbooks"))
        
        # Optional: Check for a default playbook in the project (not external repo)
        default_path = None
        default_playbook = playbook_dir / "default_playbook.yaml"
        if default_playbook.exists():
            default_path = default_playbook

        return cls(playbook_dir=playbook_dir, default_path=default_path)

    @classmethod
    def from_companion_config(cls, config: "CompanionConfig") -> PlaybookLoader:
        """Instantiate loader using a validated CompanionConfig."""
        playbook_dir = config.outputs.ace_playbooks
        default_playbook = playbook_dir / "default_playbook.yaml"
        default_path = default_playbook if default_playbook.exists() else None
        return cls(playbook_dir=playbook_dir, default_path=default_path)

