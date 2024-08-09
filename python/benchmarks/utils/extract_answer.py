def extract_boxed_answer(response: str) -> str:
    """
    Naive parser to extract the answer from \\boxed{...}.

    See https://github.com/microsoft/promptbase/blob/main/src/promptbase/math/math.py
    for something that cleans a bit more thoroughly.
    """
    BOXED_COMMAND = "\\boxed{"
    COMMAND_LENGTH = len(BOXED_COMMAND)

    start = response.find(BOXED_COMMAND)
    if start == -1:
        return ""

    brace_count = 0
    for i in range(start + COMMAND_LENGTH, len(response)):
        if response[i] == "{":
            brace_count += 1
        elif response[i] == "}":
            if brace_count == 0:
                return response[start + COMMAND_LENGTH : i]
            brace_count -= 1

    return ""


def clean_latex_answer(answer: str) -> str:
    return NotImplemented


def latex_to_float(answer: str) -> float | None:
    return NotImplemented
