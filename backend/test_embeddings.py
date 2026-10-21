from Embeddings import EmbeddingHandler

# ✅ 1. 初始化 EmbeddingHandler
handler = EmbeddingHandler()

# ✅ 2. 测试文件处理功能
with open("D:\\UserData\\Downloads\\Intelligent-QA-Chatbot\\Intelligent-QA-Chatbot\\backend\\test.txt", "rb") as file:
    file_content = file.read()
result = handler.fully_process_file(file_content, "test.txt", "doc_123")
print("File Processing Result:", result)

# ✅ 3. 测试 URL 处理功能
url_result = handler.fully_process_url("https://example.com", "doc_456")
print("URL Processing Result:", url_result)

# ✅ 4. 测试 ZIP 文件读取
with open("D:\\UserData\\Downloads\\Intelligent-QA-Chatbot\\Intelligent-QA-Chatbot\\backend\\test.zip", "rb") as zip_file:
    zip_content = zip_file.read()
zip_result = handler.read_zip_contents(zip_content, "test.zip")
print("ZIP Read Result:", zip_result)

# ✅ 5. 测试文档查询
query_result = handler.retrieve_documents("地震应急措施", num_docs=3)
print("Query Result:", query_result)

# ✅ 6. 测试删除文档
remove_result = handler.remove_documents(["doc_123"])
print("Remove Document Result:", remove_result)
