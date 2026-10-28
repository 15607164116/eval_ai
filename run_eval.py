import json
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量中的 API Key
load_dotenv()

# 初始化大模型裁判客户端（用于 llm-rubric 评分）
# 假设你用 DeepSeek 或 阿里云，替换 base_url 和 key 即可
judge_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

AGENT_API_URL = "http://127.0.0.1:8000/chat"


def call_agent(query: str) -> dict:
    """1. 发送请求给你的 Agent 接口"""
    try:
        response = requests.post(AGENT_API_URL, json={"query": query}, timeout=60)
        response.raise_for_status()
        return response.json()  # 期望返回 {"answer": "...", "trace": [...]}
    except Exception as e:
        print(f"请求 Agent 失败: {e}")
        return {"answer": "", "trace": []}


def llm_as_judge(query: str, answer: str, rubric_prompt: str) -> tuple[bool, str]:
    """2. 调用大模型作为裁判进行评分 (对应 llm-rubric)"""
    full_prompt = f"""
    {rubric_prompt}

    【用户问题】: {query}
    【助手回答】: {answer}

    请严格按要求输出。如果回答通过评估，请在第一行输出 'RESULT: PASS'，否则输出 'RESULT: FAIL'。并在第二行给出具体理由。
    """

    try:
        response = judge_client.chat.completions.create(
            model="gpt-4o-mini",  # 裁判模型
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0
        )
        result_text = response.choices[0].message.content

        # 解析大模型的结论
        is_pass = "RESULT: PASS" in result_text.upper()
        return is_pass, result_text
    except Exception as e:
        return False, f"裁判模型调用失败: {e}"


def run_evaluation(dataset_file: str):
    """3. 核心调度逻辑：读取测试集 -> 请求 -> 断言 -> 评分"""
    with open(dataset_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    total_tests = len(test_cases)
    passed_tests = 0
    results_log = []

    print(f"🚀 开始执行评测: {dataset_file} (共 {total_tests} 条用例)\n" + "=" * 50)

    for case in test_cases:
        desc = case.get("description", "未命名用例")
        query = case.get("vars", {}).get("query", "")
        asserts = case.get("assert", [])

        print(f"▶ 正在测试: {desc}")
        print(f"  Query: {query}")

        # 请求你的 Agent
        agent_response = call_agent(query)
        answer = agent_response.get("answer", "")
        trace = agent_response.get("trace", [])  # 比如 ["get_weather"]

        case_passed = True

        # 遍历该用例的所有断言
        for assertion in asserts:
            assert_type = assertion.get("type")
            metric = assertion.get("metric", "未命名指标")

            # A. 代码断言 (javascript 对应的 Python 实现)
            if assert_type == "javascript":
                # 注意：这里直接把 trace 列表传入，模拟 JS 里的 JSON.parse(output)
                # 实际工程中，建议将 JS 代码转换为对应的 Python lambda 或 eval 表达式
                js_logic = assertion.get("value", "")

                # 简单模拟：如果 JS 代码里有 includes('xxx')，就检查 Python 列表里有没有
                if f"includes('" in js_logic:
                    import re
                    tools_needed = re.findall(r"includes\('([^']+)'\)", js_logic)
                    for tool in tools_needed:
                        if tool not in trace:
                            case_passed = False
                            print(f"  ❌ [代码断言-{metric}] 失败: 未调用期望的工具 [{tool}]")
                            break
                    if case_passed:
                        print(f"  ✅ [代码断言-{metric}] 通过")

            # B. 大模型裁判 (llm-rubric)
            elif assert_type == "llm-rubric":
                rubric = assertion.get("value", "")
                is_pass, reason = llm_as_judge(query, answer, rubric)

                if is_pass:
                    print(f"  ✅ [LLM裁判-{metric}] 通过")
                else:
                    case_passed = False
                    print(f"  ❌ [LLM裁判-{metric}] 失败")
                    print(f"     理由: {reason.splitlines()[-1] if reason else '无'}")  # 打印最后一行理由

        if case_passed:
            passed_tests += 1

        results_log.append({
            "description": desc,
            "query": query,
            "answer": answer,
            "trace": trace,
            "passed": case_passed
        })
        print("-" * 50)

    # 4. 输出最终报告
    pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print("\n" + "=" * 50)
    print(f"🎯 评测完成！整体通过率: {pass_rate:.1f}% ({passed_tests}/{total_tests})")

    # 保存报告
    report_path = dataset_file.replace(".json", "_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"pass_rate": pass_rate, "details": results_log}, f, ensure_ascii=False, indent=2)
    print(f"📄 详细报告已保存至: {report_path}")


if __name__ == "__main__":
    # 运行指定的测试集文件
    run_evaluation("datasets/routing.json")
    # run_evaluation("datasets/rag.json")