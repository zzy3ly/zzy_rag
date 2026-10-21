import time
from LLM import LLMInteraction

# 初始化 LLMInteraction
llm_handler = LLMInteraction()

# 检查是否使用本地或付费 API
print(f"Paid Status: {llm_handler.get_paid_status()}")

# 测试查询
query = "What is NBA?"
print(query)

# 选项：如果你想测试不同的模型，可以取消注释
# llm_handler.switch_model("openai/gpt-4-turbo-preview")

# 生成 LLM 响应
print("\nGenerating response...\n")
response_generator = llm_handler.generate_response(query)

# 逐步打印 LLM 响应（适用于流式输出）
for chunk in response_generator:
     print(chunk, end="", flush=True)
     time.sleep(0.05)  # 模拟流式输出

print("\n\n=== Response End ===\n")

# 模拟文档检索测试（重要）
similar_documents = """
<$$ THIS SIGNIFIES THE BEGINNING OF RELEVANT CONTEXTUAL DOCUMENTS $$>

<$$ Document Index: 1 $$>
<$$ Filename: AI_Intro $$>
<$$ Content: Artificial intelligence is a field of computer science that aims to create intelligent machines. $$>

<$$ Document Index: 2 $$>
<$$ Filename: ML_Concepts $$>
<$$ Content: Machine learning is a subset of AI focused on algorithms that improve automatically through experience. $$>

<$$ THIS SIGNIFIES THE END OF ATTACHED DATA AND MARKS THE BEGINNING OF USER QUERY $$>
"""

print("\nGenerating response with structured similar documents...\n")
query2= "What is Artificial intelligence?"
response_generator_with_docs = llm_handler.generate_response(query2, similar_documents=similar_documents)

for chunk in response_generator_with_docs:
    print(chunk, end="", flush=True)