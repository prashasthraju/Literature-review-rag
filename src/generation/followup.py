"""Turn a follow-up message into a standalone question for retrieval and answering.

"What are its limitations?" retrieves nothing useful on its own. The
conversation is used only to resolve such references; the answer step still
sees just the standalone question and the retrieved sources, so the measured
single-turn behaviour (docs/generation.md) is what a chat turn gets.
"""

import re

from src.generation.answer import MARKER_RE, load_prompt, named_documents

CONDENSE_PROMPT = "condense_v1"
HISTORY_TURNS = 3
MAX_ANSWER_CHARS = 600
MAX_QUESTION_WORDS = 80
LABEL_RE = re.compile(r"^(?:standalone\s+)?(?:question|rewritten(?:\s+question)?)\s*:\s*", re.IGNORECASE)


def build_condense_messages(history, message, prompt_version=CONDENSE_PROMPT):
    """``history``: [(user_message, assistant_answer)] oldest first."""
    lines = []
    for user, assistant in history[-HISTORY_TURNS:]:
        lines.append(f"User: {user.strip()}")
        if assistant:
            answer = MARKER_RE.sub("", assistant).strip()
            if len(answer) > MAX_ANSWER_CHARS:
                answer = answer[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0] + " …"
            lines.append(f"Assistant: {answer}")
    conversation = "\n".join(lines)
    return [
        {"role": "system", "content": load_prompt(prompt_version)},
        {
            "role": "user",
            "content": f"<conversation>\n{conversation}\n</conversation>\n\nLast message: {message.strip()}\n\nStandalone question:",
        },
    ]


def parse_condensed(raw, message):
    """The model's first line, cleaned; the original message if the output is unusable."""
    line = next((line.strip() for line in (raw or "").splitlines() if line.strip()), "")
    line = LABEL_RE.sub("", line).strip().strip('"“”\'`').strip()
    if not line or len(line.split()) > MAX_QUESTION_WORDS:
        return message.strip()
    return line


def rewrite(generator, history, message, max_new_tokens, aliases=None):
    """The standalone question.

    A message that names a paper is about that paper and already stands alone, so it is used
    as is: rewriting it only added risks (swapping DF40 for MotionDiffuse, appending the
    previous turn's topic). Otherwise the model resolves the reference from the conversation.
    """
    if named_documents(message, aliases or {}):
        return message.strip()
    raw = generator.generate(build_condense_messages(history, message), max_new_tokens=max_new_tokens)
    return parse_condensed(raw, message)
