import json
import os
import requests
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

# 加载 API Key
load_dotenv()
judge_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
)

AGENT_API_URL = "http://127.0.0.1:8000/chat"


def call_agent(query: str) -> dict:
    """请求你的 FastAPI 接口"""
    try:
        res = requests.post(AGENT_API_URL, json={"query": query}, timeout=30)
        return res.json()  # 期望返回 {"answer": "...", "trace": [...]}
    except Exception as e:
        print(f"❌ 请求 Agent 失败: {e}")
        return {"answer": "", "trace": []}


def run_js_assertion(js_code: str, output_data: list) -> bool:
    """用 Node.js 执行你的 JavaScript 断言"""
    # 把你的 trace 列表转成 JSON 字符串传给 JS
    output_str = json.dumps(output_data)

    # 包装一下你的 JS 代码，让它返回 true 或 false
    script = f"""
    try {{
        const output = JSON.stringify({output_str});
        const result = (function() {{
            {js_code}
        }})();
        console.log(result);
    }} catch (e) {{
        console.log("false");
    }}
    """

    try:
        # 在终端执行 Node.js
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
        return "true" in result.stdout.lower()
    except Exception:
        return False


def run_llm_judge(query: str, answer: str, rubric: str) -> tuple[bool, str]:
    """调用大模型裁判"""
    prompt = f"""
    {rubric}

    【用户问题】: {query}
    【助手回答】: {answer}

    请严格按照要求评估。如果通过，第一行输出 'RESULT: PASS'；否则输出 'RESULT: FAIL'，第二行给出理由。
    """
    try:
        res = judge_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        text = res.choices[0].message.content
        is_pass = "RESULT: PASS" in text.upper()
        return is_pass, text
    except Exception as e:
        return False, str(e)


def evaluate_file(dataset_file: str):
    """主流程：读取 JSON -> 请求 Agent -> 运行断言 -> 报告"""
    with open(dataset_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    pass_count = 0

    print(f"🚀 开始执行: {dataset_file} | 共 {len(test_cases)} 条用例\n" + "=" * 50)

    for case in test_cases:
        desc = case.get("description")
        query = case["vars"]["query"]
        asserts = case.get("assert", [])

        print(f"▶ 测试: {desc}")
        print(f"  Query: {query[:20]}...")

        # 1. 请求你的 Agent 拿到结果
        agent_res = call_agent(query)
        answer = agent_res.get("answer", "")
        trace = agent_res.get("trace", [])
        print(f"  Agent 调用工具: {trace}")

        case_pass = True

        # 2. 依次执行断言
        for a in asserts:
            a_type = a.get("type")

            # (A) 代码断言
            if a_type == "javascript":
                # 执行你 JSON 里写的 JS 代码
                js_ok = run_js_assertion(a["value"], trace)
                if js_ok:
                    print(f"  ✅ [JS断言] 通过")
                else:
                    print(f"  ❌ [JS断言] 失败: {a['value']}")
                    case_pass = False
                    break  # 代码断言挂了，直接中断，不跑大模型裁判了（省钱省时间！）

            # (B) LLM 裁判断言
            elif a_type == "llm-rubric":
                # 如果前面的 JS 断言没过，这里根本不会执行
                is_pass, reason = run_llm_judge(query, answer, a["value"])
                if is_pass:
                    print(f"  ✅ [LLM裁判] 通过")
                else:
                    print(f"  ❌ [LLM裁判] 失败")
                    # 提取一行原因
                    clean_reason = [line for line in reason.split('\n') if "FAIL" not in line and line.strip()]
                    if clean_reason: print(f"     > 理由: {clean_reason[0][:50]}...")
                    case_pass = False
                    break  # 裁判挂了，也提前中断当前用例

        if case_pass:
            pass_count += 1
        print("-" * 50)

    print(f"📊 整体通过率: {pass_count}/{len(test_cases)}")


if __name__ == "__main__":
    # 运行你的异常测试集
    evaluate_file("datasets/abnormal_input.json")