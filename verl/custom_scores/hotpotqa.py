def compute_score(solution_str, ground_truth: list) -> float: 
    def compute_score_single(solution_str, ground_truth) -> float:
        ground_truth = ground_truth.lower()

        retval = 0.
        try:
            string_in_last_boxed = last_boxed_only_string(solution_str)
            if string_in_last_boxed is not None:
                answer = remove_boxed(string_in_last_boxed)
                retval = custom_compute(answer, ground_truth)
        except Exception as e:
            print(e)
        return retval
    solution_str = solution_str[-300:].lower()
    return max(compute_score_single(solution_str, gt) for gt in ground_truth)


def custom_compute(str1, str2):
    if not str1:
        return 0.0
    
    s1 = str1.lower()
    s2 = str2.lower()
    
    
    s1 = s1.split()
    s2 = s2.split()
    
    n, m = len(s1), len(s2)
    dp = [0] * (m + 1)
    
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
            
    lcs_len = dp[m]
    ratio = lcs_len / max(len(s1), len(s2))
    
    return ratio
    
    return 0.0

# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"
    


    assert s[:len(left)] == left
    assert s[-1] == "}"

    s = s[len(left):-1]


    if ("\\text{" in s) and ("}" in s):
        s = s.split("\\text")[-1].strip(' {}')

    return s

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    # remove spaces
    string = string.replace(" ", "")

    return string
