"""Starter harness — scaffolding hooks for McpMHAgent.

The MetaHarness evolver can modify this file to change how the agent
assembles prompts, selects tools, and constructs user messages.

Available hooks (all optional — delete or leave unimplemented to use defaults):

  build_system_prompt(base_prompt: str, skills: list[SkillMeta], task_prompt: str | None) -> str
  build_user_prompt(task_id: str, task_input: str) -> str | None
  pre_solve(task_metadata: dict) -> dict
"""


def build_system_prompt(base_prompt: str, skills: list, task_prompt: str | None = None) -> str:
    """Assemble the system prompt.

    The default implementation returns the base prompt as-is.
    Skills are available in the skills/ directory but are NOT injected
    into the prompt by default — the parent class's read_skill tool
    does not exist in the solver's tool set.

    The MetaHarness proposer can evolve this hook to incorporate
    skill content, add strategies, or restructure the prompt.
    """
    return base_prompt
