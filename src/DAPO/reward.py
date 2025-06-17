import re
import ast


def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        try:
            completion = "<think>" + completion
            regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\s*<answer>([\s\S]*?)<\/answer>\s*$"
            match = re.search(regex, completion, re.DOTALL)
            
            if match is None or len(match.groups()) != 2:
                rewards.append(-1.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(-1.0)
    return rewards


def remove_comments_and_blank_lines(code):
    string_pattern = r'(\'\'\'(.*?)\'\'\'|"""(.*?)"""|\'(.*?)\'|"(.*?)")'
    comment_pattern = r'#.*?$|"""(.*?)"""|\'\'\'(.*?)\'\'\''

    placeholder_map = {}
    def replace_strings(match):
        placeholder = f"STRING_PLACEHOLDER_{len(placeholder_map)}"
        placeholder_map[placeholder] = match.group(0)
        return placeholder

    def restore_strings(code):
        for placeholder, original in placeholder_map.items():
            code = code.replace(placeholder, original)
        return code

    code_without_strings = re.sub(string_pattern, replace_strings, code, flags=re.DOTALL)

    code_without_comments = re.sub(comment_pattern, '', code_without_strings, flags=re.MULTILINE | re.DOTALL)

    final_code = restore_strings(code_without_comments)

    final_code = re.sub(r'\n\s*\n', '\n', final_code)

    return re.sub(r"^\s*$\n", "", final_code.strip(), flags=re.MULTILINE)


def syntax_check(completion):
    try:
        ast.parse(completion)
        return True
    except SyntaxError:
        return False


def em_star(completions, target, **kwargs):
    rewards = []
    for completion, tar in zip(completions, target):
        matches = re.findall(r"```python\n(.*?)\n```", completion, re.DOTALL)
        if not matches:
            rewards.append(-2.0)
            continue
        completion = remove_comments_and_blank_lines(matches[0].strip())
        if not syntax_check(completion):
            rewards.append(-2.0)
        elif completion == remove_comments_and_blank_lines(tar):
            rewards.append(2.0)
        else:
            rewards.append(-1.5)
    return rewards


def es_star(completions, target, **kwargs):
    def levenshtein_distance(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + 1)
        return dp[m][n]
    
    def levenshtein_similarity(s1, s2):
        distance = levenshtein_distance(s1, s2)
        max_length = max(len(s1), len(s2))
        similarity = 1 - (distance / max_length)
        return similarity

    rewards = []
    for completion, tar in zip(completions, target):
        matches = re.findall(r"```python\n(.*?)\n```", completion, re.DOTALL)
        if not matches:
            rewards.append(-2.0)
            continue
        completion = remove_comments_and_blank_lines(matches[0].strip())
        if not syntax_check(completion):
            rewards.append(-2.0)
        else:
            tar = remove_comments_and_blank_lines(tar)
            reward = levenshtein_similarity(completion, tar) * 3.5 - 1.5
            rewards.append(reward)
    return rewards