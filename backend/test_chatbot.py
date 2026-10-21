from ChatBot import ModularChatbot
from Database import Database  # 这里假设你有一个模拟数据库的类

# 创建一个模拟数据库实例
mock_db = Database()

# 初始化 chatbot（不提供任何文档）
chatbot = ModularChatbot(database=mock_db)

# 设置测试查询
query = "What is USA?"
userid = "test_user"
conversation_id = "test_convo"

# 获取 chatbot 响应
print("=== Test 1: Asking a general knowledge question ===")
response_generator = chatbot.handle_query(query, userid, conversation_id)

for response, _ in response_generator:
    print(response, end='', flush=True)  # 让字符不换行输出

# from ChatBot import ModularChatbot
# from Database import Database  # 这里假设你有一个模拟数据库的类

# # 创建一个模拟数据库实例
# mock_db = Database()

# # 初始化 chatbot（不提供任何文档）
# chatbot = ModularChatbot(database=mock_db)

# # 切换到本地模型
# chatbot.llm_interaction.switch_model("llama3:instruct")  # 确保切换到本地模型

# # 设置测试查询
# query = "What is artificial intelligence?"
# userid = "test_user"
# conversation_id = "test_convo"

# # 获取 chatbot 响应
# print("=== Test 1: Asking a general knowledge question ===")
# response_generator = chatbot.handle_query(query, userid, conversation_id)

# for response, _ in response_generator:
#     print(response, end='', flush=True)  # 让字符不换行输出
