from __future__ import annotations

from genai_companion_with_ace.chat.modes import ConversationMode
from genai_companion_with_ace.chat.response_formatter import Citation, ResponseFormatter


def test_response_formatter_renders_citations_and_followups() -> None:
    formatter = ResponseFormatter()
    formatted = formatter.format_answer(
        "Backpropagation computes gradients for each weight.",
        citations=[Citation(source="Course 9 - Module 2", snippet="Explains gradient descent.")],
        mode=ConversationMode.STUDY,
        follow_ups=["Would you like a code example?"],
    )
    # By default, citations are not included in render (inline citations are in answer text)
    rendered = formatted.render(include_citations=False)
    assert "Sources" not in rendered
    assert "Course 9 - Module 2" not in rendered
    assert "Suggested follow-ups" in rendered
    
    # But can be explicitly included
    rendered_with_citations = formatted.render(include_citations=True)
    assert "Sources" in rendered_with_citations
    assert "Course 9 - Module 2" in rendered_with_citations


def test_response_formatter_fallback_contains_reason() -> None:
    formatter = ResponseFormatter()
    fallback = formatter.format_fallback(
        mode=ConversationMode.QUICK,
        reason="The topic is outside of the IBM curriculum",
        follow_up_suggestion="Try rephrasing with a course-specific concept.",
    )
    assert "outside of the IBM curriculum" in fallback.answer
    assert fallback.disclaimer is not None

