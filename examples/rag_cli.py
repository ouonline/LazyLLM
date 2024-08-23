# -*- coding: utf-8 -*-

import lazyllm
from lazyllm import (
    pipeline,
    parallel,
    bind,
    SentenceSplitter,
    Document,
    Retriever,
    Reranker,
    TimeRecorder,
)

# ----- Part 1 ----- #

prompt = (
    "You will play the role of an AI Q&A assistant and complete a dialogue task. In this task, "
    "you need to provide your answer based on the given context and question."
)

documents = Document(
    dataset_path="rag_master", embed=lazyllm.OnlineEmbeddingModule(), create_ui=False
)

documents.create_node_group(
    name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100
)

# ----- Part 2 ----- #

with pipeline() as ppl:

    # ----- 2.1 ----- #

    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(
            documents,
            group_name="CoarseChunk",
            similarity="bm25_chinese",
            similarity_cut_off=0.003,
            topk=3,
        )
        prl.retriever2 = Retriever(
            documents, group_name="sentences", similarity="cosine", topk=3
        )

    # ----- 2.2 ----- #

    ppl.reranker = Reranker(
        "ModuleReranker", model="bge-reranker-large", topk=1
    ) | bind(query=ppl.input)

    # ----- 2.3 ----- #

    ppl.formatter = (
        lambda nodes, query: dict(
            context_str="".join([node.get_content() for node in nodes]), query=query
        )
    ) | bind(query=ppl.input)

    # ----- 2.4 ----- #

    ppl.llm = lazyllm.OnlineChatModule(stream=False).prompt(
        lazyllm.ChatPrompter(prompt, extro_keys=["context_str"])
    )

# ----- Part 3 ----- #

rag = lazyllm.ActionModule(ppl)
rag.start()

query = '道德经的作者是？'
res = rag(query)
print(f"answer: {str(res)}\n")

ddd = TimeRecorder.data.sort(reverse=True, key=lambda l: l[3])
for d in TimeRecorder.data:
    print(d)
