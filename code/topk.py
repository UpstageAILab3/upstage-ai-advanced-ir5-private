def answer_question(messages, initial_topk=3, max_topk=10, relevance_threshold=0.5):
    response = {"standalone_query": "", "topk": [], "references": [], "answer": ""}

    msg = [{"role": "system", "content": persona_function_calling}] + messages
    try:
        result = client.chat.completions.create(
            model=llm_model,
            messages=msg,
            tools=tools,
            temperature=0,
            seed=1,
            timeout=10
        )
    except Exception as e:
        traceback.print_exc()
        return response

    if result.choices[0].message.tool_calls:
        tool_call = result.choices[0].message.tool_calls[0]
        function_args = json.loads(tool_call.function.arguments)
        standalone_query = function_args.get("standalone_query")

        # 동적 topk 설정
        query_complexity = len(standalone_query.split())
        topk = min(max(initial_topk, query_complexity // 5), max_topk)

        search_result = sparse_retrieve(standalone_query, topk)

        response["standalone_query"] = standalone_query
        retrieved_context = []
        relevant_results = 0

        for i, rst in enumerate(search_result['hits']['hits']):
            if rst["_score"] >= relevance_threshold:
                retrieved_context.append(rst["_source"]["content"])
                response["topk"].append(rst["_source"]["docid"])
                response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})
                relevant_results += 1

        # 적응형 topk: 관련성 높은 결과가 충분하지 않으면 topk 증가
        if relevant_results < initial_topk and topk < max_topk:
            additional_results = sparse_retrieve(standalone_query, max_topk - topk)
            for rst in additional_results['hits']['hits']:
                if rst["_score"] >= relevance_threshold:
                    retrieved_context.append(rst["_source"]["content"])
                    response["topk"].append(rst["_source"]["docid"])
                    response["references"].append({"score": rst["_score"], "content": rst["_source"]["content"]})
                    relevant_results += 1
                if relevant_results >= initial_topk:
                    break

        content = json.dumps(retrieved_context)
        messages.append({"role": "assistant", "content": content})
        msg = [{"role": "system", "content": persona_qa}] + messages
        try:
            qaresult = client.chat.completions.create(
                    model=llm_model,
                    messages=msg,
                    temperature=0,
                    seed=1,
                    timeout=30
                )
        except Exception as e:
            traceback.print_exc()
            return response
        response["answer"] = qaresult.choices[0].message.content

    else:
        response["answer"] = result.choices[0].message.content

    return response