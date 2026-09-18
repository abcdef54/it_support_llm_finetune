"""Budget the complete Qwen chat, preserving the entire user question."""
from benchmark.baseline.config import ENABLE_THINKING, MAX_INPUT_TOKENS
from rag import config as C


def messages_for(question, context):
    return [{"role": "system", "content": C.RAG_SYSTEM_PROMPT},
            {"role": "user", "content": "Retrieved technical context:\n" + (context or "(none fits)")
             + "\n\nUser question:\n" + question}]


def prompt_tokens(tokenizer, messages):
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              enable_thinking=ENABLE_THINKING, return_dict=False))


def build_context(question, records, tokenizer, *, max_context_tokens=C.RAG_MAX_CONTEXT_TOKENS,
                  max_input_tokens=MAX_INPUT_TOKENS):
    if not question.strip() or max_context_tokens < 1 or max_input_tokens < 1:
        raise ValueError("Question and positive context/input budgets are required")
    if prompt_tokens(tokenizer, messages_for(question, "")) > max_input_tokens:
        raise ValueError("User question exceeds input budget; it will not be silently truncated")
    context, used, skipped = "", [], []
    for record in records:
        label = "Stack Overflow" if record["source"] == "stackoverflow" else "IBM TechQA"
        passage = f"[{len(used) + 1}] {label} ({record['id']})\n{record['context_text']}"
        proposed = context + ("\n\n" if context else "") + passage
        if (len(tokenizer.encode(proposed, add_special_tokens=False)) > max_context_tokens
                or prompt_tokens(tokenizer, messages_for(question, proposed)) > max_input_tokens):
            skipped.append(record["id"])
            continue
        context = proposed
        used.append({"id": record["id"], "source": record["source"], "score": record["score"],
                     "citation": len(used) + 1})
    messages = messages_for(question, context)
    return {"messages": messages, "context": context, "used_documents": used, "skipped_document_ids": skipped,
            "context_tokens": len(tokenizer.encode(context, add_special_tokens=False)),
            "input_tokens": prompt_tokens(tokenizer, messages), "policy": C.CONTEXT_POLICY}
