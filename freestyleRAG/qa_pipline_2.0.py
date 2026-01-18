# import os
# import json
# import asyncio
# import time
# import re
# import ast
# from .baseClasses import (TextChunkSchema, BaseGraphStorage, BaseKVStorage, 
#                           BaseVectorStorage, QueryParameters)
# from .prompt_template import PROMPTS
# from .utils import (encode_string_by_tiktoken, decode_tokens_by_tiktoken,
#                     pack_user_ass_to_openai_messages, split_string_by_delimiters,
#                     clean_str, logger, compute_mdhash_id, truncate_list_by_token_size,
#                     is_float_regex, list_of_list_to_csv, split_string_when_redescription,
#                     PlanningModelConfig)
# from .storageClasses import KVCacheProcess
# from collections import Counter, defaultdict
# from .temporary_storage import TemporarySave


# async def get_query_context_with_keywords(
#     keywords,
#     knowledge_graph_inst: BaseGraphStorage,
#     relationships_vdb: BaseVectorStorage,
#     text_chunks_db: BaseKVStorage[TextChunkSchema],
#     query_param: QueryParameters,
#     use_model_func,
#     need_info_list:bool=False
# ):
#     results = await relationships_vdb.query(keywords, top_k=query_param.top_k)
#     if not len(results):
#         return None
    
#     edge_datas = await asyncio.gather(
#         *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
#     )
#     if not all([n is not None for n in edge_datas]):
#         logger.warning("Some edges are missing, maybe the storage is damaged")
#     edge_degree = await asyncio.gather(
#         *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
#     )
#     edge_datas = [
#         {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
#         for k, v, d in zip(results, edge_datas, edge_degree)
#         if v is not None
#     ]
#     edge_datas = sorted(  # sort from largest to smallest
#         edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
#     )
    
#     edge_datas = truncate_list_by_token_size(  # For descriptions, truncation is required beyond the maximum length
#         edge_datas,
#         key=lambda x: x['description'],
#         max_token_size=query_param.max_token_for_global_context
#     )

#     use_text_units, use_text_units_with_id  = await find_related_text_unit_from_relationships(
#         edge_datas, query_param, text_chunks_db
#     )

#     logger.info(
#         f"Query uses {len(edge_datas)} relations, {len(use_text_units)} text units"
#     )
#     # new_edge_datas = []
#     # new_edge_datas = await regenerate_description(edge_datas, use_text_units_with_id, use_model_func)
#     # if new_edge_datas:
#     #     edge_datas = new_edge_datas

#     relations_section_list = [
#         ["id", "source entity", "relationship between the source entity and the target entity", "target entity", "weight", "rank"]
#     ]
#     for i, e in enumerate(edge_datas):
#         relations_section_list.append(
#             [
#                 i,
#                 e["src_id"],
#                 e["description"],
#                 e["tgt_id"],
#                 e["weight"],
#                 e["rank"],
#             ]
#         )

#     relations_context = list_of_list_to_csv(relations_section_list)

#     text_units_section_list = [["id", "content"]]
#     for i, t in enumerate(use_text_units):
#         text_units_section_list.append([i, t["content"]])
#     text_units_context = list_of_list_to_csv(text_units_section_list)

#     info_str = f"""
# -----Relationships-----
# ```csv
# {relations_context}
# ```
# -----Sources-----
# ```csv
# {text_units_context}
# ```
# """
#     if need_info_list:
#         return info_str, relations_section_list, text_units_section_list
#     else:
#         return info_str


# class QA_Pipeline_v2:
#     def __init__(self, original_q, knowledge_graph_inst, 
#                  relationships_vdb, text_chunks_db, q_param, global_config):
        
#         self.original_q = original_q
#         self.q1 = None
#         self.ans1 = None
#         self.q2 = None
#         self.ans2 = None
#         self.q3 = None
#         self.ans3 = None
#         self.final_ans = None
#         self.strategy = False
#         self.knowledge_graph_inst = knowledge_graph_inst
#         self.relationships_vdb = relationships_vdb
#         self.text_chunks_db = text_chunks_db
#         self.q_param = q_param
#         self.q_param.top_k = 40
#         self.final_summary_q_param = q_param
#         self.global_config = global_config
#         self.info_relation_list = []
#         self.info_text_list = []
#         self.q1_summary: str = ''
#         self.q2_summary: str = ''
#         self.q3_summary: str = ''

#     def str2set(self,s:str):
#         s = s.strip()
#         ans_list = s.lstrip('[').rstrip(']').split('#')
#         if(isinstance(ans_list, str)):
#             ans_list = [ans_list]
#         for i in range(len(ans_list)):
#             a = ans_list[i].strip()
#             a = a.lstrip('[').rstrip(']')
#             ans_list[i] = a
#         # if(len(ans_list)>5):
#         #     ans_list = ans_list[0:5]
#         ans_set = set(ans_list)
#         ans_set.discard('None')
#         return ans_set

#     async def merge_info(self):  # 这是否需要再调整，看看三个问题时会不会太长
#         order_relation_list = []
#         order_text_list = []
#         seen_elements = set()
#         for sublist in self.info_relation_list:
#             judge_str = sublist[1] + sublist[2] + sublist[3]
#             if judge_str not in seen_elements:
#                 order_relation_list.append(sublist)
#                 seen_elements.add(judge_str)
#         seen_elements = set()
#         for sublist in self.info_text_list:
#             judge_str = sublist[1]
#             if judge_str not in seen_elements:
#                 order_text_list.append(sublist)
#                 seen_elements.add(judge_str)
#         str_relation = list_of_list_to_csv(order_relation_list)
#         str_text = list_of_list_to_csv(order_text_list)
#         return f"""
# -----Relationships-----
# ```csv
# {str_relation}
# ```
# -----Sources-----
# ```csv
# {str_text}
# ```
# """

#     async def getting_response_from_info(self, info, query, sub_query_info=None, who_call = None):

#         use_model_func = self.global_config["llm_model_func"]
#         if info is None:
#             return PROMPTS["fail_response"]

#         # 这里加上总结，用于提炼关系和对文本进行总结
#         if self.q_param.need_summary_return:
#             summary_prompt = PROMPTS["summary_response"]
#             summary_prompt = summary_prompt.format(input_question=query, input_data_table=info)
#             temp_info = await use_model_func(prompt = summary_prompt)
#             if "<|end|>" in temp_info:
#                 temp_info = re.sub(r'<\|end\|>', '', temp_info)
#             if "Relationships" in temp_info and "Sources" in temp_info:
#                 info = temp_info
#                 if who_call == "q1":
#                     self.q1_summary = info
#                 elif who_call == "q2":
#                     self.q2_summary = info
#                 elif who_call == "q3":
#                     self.q3_summary = info
#         elif self.q_param.final_answer_summary and who_call == "final_q":
#             summary_prompt = PROMPTS["summary_response"]
#             summary_prompt = summary_prompt.format(input_question=query, input_data_table=info)
#             temp_info = await use_model_func(prompt = summary_prompt)
#             if "<|end|>" in temp_info:
#                 temp_info = re.sub(r'<\|end\|>', '', temp_info)
#             info = temp_info

#         if self.q_param.task_name == "qa":
#             prompt_temp = PROMPTS['qa_response']
#             if sub_query_info is not None:
#                 sys_prompt = prompt_temp.format(
#                     context=info+sub_query_info, input=query)
#             else:
#                 sys_prompt = prompt_temp.format(
#                     context=info, input=query)
#             response = await use_model_func(
#                 prompt=sys_prompt)
#         else:
#             sys_prompt_temp = PROMPTS["rag_response"]
#             sys_prompt = sys_prompt_temp.format(
#                 context_data=info, response_type=self.q_param.response_type
#             )
#             response = await use_model_func(
#             query,
#             system_prompt=sys_prompt,
#             )

#         if len(response) > len(sys_prompt):
#             response = (
#                 response.replace(sys_prompt, "")
#                 .replace("user", "")
#                 .replace("model", "")
#                 .replace(query, "")
#                 .replace("<system>", "")
#                 .replace("</system>", "")
#                 .strip()
#             )
#         return response

#     async def Final_QA(self,final_call:str):
#         answer = None
#         if('Ans_1' in final_call):
#             answer = self.ans1
#             self.final_ans = answer[0]
#         elif('Ans_2' in final_call):
#             if(self.ans2):
#                 if len(self.ans2) == 1:
#                     answer = self.ans2[0]
#                 else:
#                     answer = set()
#                     for s in self.ans2:
#                         if(s):
#                             answer = answer.union(s)
#             self.final_ans = answer
#         elif('Ans_3' in final_call):
#             if(self.ans3):
#                 if len(self.ans3) == 1:
#                     answer = self.ans3[0]
#                 else:
#                     answer = set()
#                     for s in self.ans3:
#                         if(s):
#                             answer = answer.union(s)  # 去除重复项
#             self.final_ans = answer

#         # 最后对原始答案进行总结，并与self.final_ans进行比较，如果答案不一样，则相信总结的答案。
#         if self.q_param.final_answer_summary:
#             use_model_func = self.global_config["llm_model_func"]
#             info = await get_query_context_with_keywords(
#                 keywords=self.original_q,
#                 knowledge_graph_inst=self.knowledge_graph_inst,
#                 relationships_vdb=self.relationships_vdb,
#                 text_chunks_db=self.text_chunks_db,
#                 query_param=self.final_summary_q_param,
#                 use_model_func=use_model_func,
#                 )
#             sub_ques_info = await self.get_sub_query_info()
#             summary_ans = await self.getting_response_from_info(info, self.original_q, sub_ques_info, who_call="final_q")
#             match_num = 0
#             if isinstance(self.final_ans, set):
#                 final_ans_str = ""
#                 for ansline in self.final_ans:
#                     if isinstance(ansline, str):
#                         final_ans_str += ansline
#                     else:
#                         self.final_ans = None
#                 anslist = self.split_str(final_ans_str)
#             else:
#                 final_ans = self.final_ans
#                 anslist = self.split_str(final_ans)
#             if anslist:
#                 for one_ans in anslist:
#                     if one_ans in summary_ans:
#                         match_num += 1
#                 if match_num == 0 or len(anslist) > 3:
#                     self.final_ans = summary_ans
#             else:
#                 self.final_ans = None

#     def split_str(self, text):
#         if text:
#             # 按逗号分割字符串
#             sub_strings = text.split(',')
#             # 去除每个子字符串的首尾空格（如果有）
#             sub_strings = [s.strip() for s in sub_strings]
#             return sub_strings
#         else:
#             return ""

#     async def question_answer_link(self, q, ans, only_question=False):
#         sub_answer = ""

#         if only_question:
#             if isinstance(q, list):
#                 for i in range(len(q)):
#                    if q[i]:
#                     sub_answer += f"sub-question: {q[i]}\n"
#             else:
#                 if q:
#                     sub_answer = f"sub-question: {q}\n"
#             return sub_answer

#         if q and ans:
#             if isinstance(q, list) and isinstance(ans, list):
#                 min_length = min(len(q), len(ans))
#                 for i in range(min_length):
#                     if q[i] and ans[i]:
#                         sub_answer += f"Question: {q[i]} Answer: {ans[i]}\n"
#             elif isinstance(q, list):
#                 for q_single in q:
#                     if q_single:
#                         sub_answer += f"Question: {q_single} Answer: {ans}\n"
#             elif isinstance(ans, list):
#                 for ans_single in ans:
#                     if ans_single:
#                         sub_answer += f"Question: {q} Answer: {ans_single}\n"
#             else:
#                 sub_answer = f"Question: {q} Answer: {ans}\n"

#         return sub_answer       

#     async def get_sub_query_info(self, only_question=False, only_original_question=False):

#         if only_question:
#             sub_q1 = await self.question_answer_link(q=self.q1, ans=self.ans1, only_question=only_question)
#             sub_q2 = await self.question_answer_link(q=self.q2, ans=self.ans2, only_question=only_question)
#             sub_q3 = await self.question_answer_link(q=self.q3, ans=self.ans3, only_question=only_question)
#             sub_q = sub_q1 + sub_q2 + sub_q3
#             if len(sub_q):
#                 sub_prefix = "\n(Here are some sub-questions related to the final question and they might help you to answer. Please note that you should not answer these sub-questions: \n"
#                 sub_query_information = sub_prefix + sub_q + ")\n"
#             else:
#                 sub_query_information = None
#             return sub_query_information
        
#         if only_original_question:
#             sub_query_information = f'\n### The prerequisite conditions for the current question:###{self.original_q}\n'
#             return sub_query_information

#         sub_answer1 = await self.question_answer_link(q=self.q1, ans=self.ans1)
#         sub_answer2 = await self.question_answer_link(q=self.q2, ans=self.ans2)
#         sub_answer3 = await self.question_answer_link(q=self.q3, ans=self.ans3)
#         sub_answer = sub_answer1 + sub_answer2 + sub_answer3
#         if len(sub_answer):
#             sub_prefix = "\n("
#             sub_query_information = sub_prefix + sub_answer + ")\n"
#         else:
#             sub_query_information = None
#         return sub_query_information

#     async def output_parse(self, llm_predict:str):
#         # 当中间某个子问题，LLM无法回答时，记为True，调用保底
#         error_flag = False
#         predict_lines = llm_predict.split('\n')
#         for line in predict_lines:  # 以换行分割出来的每一行，thought的部分丢弃不要
#             if('Sub_Question_1: str = ' in line):
#                 q1 = line.split('Sub_Question_1: str = ')[1].lstrip('f').strip("\"")  # 获取到子问题
#                 if ("{Ans_1}" in q1):
#                     break
#                 print('Sub_Question_1:'+q1)
#                 self.q1 = [q1]
#                 # 下一步就是进行检索然后进行回答，得到答案和检索到的信息
#                 information, info_relation_list, info_text_list = \
#                 await get_query_context_with_keywords(
#                     keywords=q1,
#                     knowledge_graph_inst=self.knowledge_graph_inst,
#                     relationships_vdb=self.relationships_vdb,
#                     text_chunks_db=self.text_chunks_db,
#                     query_param=self.q_param,
#                     use_model_func=self.global_config["llm_model_func"],
#                     need_info_list=True
#                     )
#                 self.info_relation_list.extend(info_relation_list)
#                 self.info_text_list.extend(info_text_list)
#                 ans1 = await self.getting_response_from_info(information, q1, who_call="q1")
#                 self.ans1 = [ans1]
#                 # self.info1 = [information]
#                 print('Ans_1:'+str(self.ans1))
#                 if ans1:
#                     error_flag = True
#                 else:
#                     break
#             if('Sub_Question_2: str = ' in line):
#                 q2 = line.split('Sub_Question_2: str = ')[1].lstrip('f').strip("\"")
#                 print('Sub_Question_2:'+q2)
#                 question2 = []
#                 # info2 = []
#                 answer2 = []
#                 if("{Ans_1}" in q2):
#                     if len(self.ans1) == 1:
#                         ans_1 = self.ans1
#                     else:
#                         ans_1 = set()
#                         for s in self.ans1:
#                             if(s):
#                                 ans_1 = ans_1.union(s)
#                     for idx, a in enumerate(ans_1):
#                         q2_1 = q2.replace("{Ans_1}",a)
#                         print('Sub_Question_2_' + str(idx) + ':' +q2_1)
#                         information, info_relation_list, info_text_list = \
#                         await get_query_context_with_keywords(
#                             keywords=q2_1,
#                             knowledge_graph_inst=self.knowledge_graph_inst,
#                             relationships_vdb=self.relationships_vdb,
#                             text_chunks_db=self.text_chunks_db,
#                             query_param=self.q_param,
#                             use_model_func=self.global_config["llm_model_func"],
#                             need_info_list=True
#                             )
#                         self.info_relation_list.extend(info_relation_list)
#                         self.info_text_list.extend(info_text_list)
#                         sub_query_info = await self.get_sub_query_info(only_original_question=True)
#                         ans2 = await self.getting_response_from_info(information, q2_1, sub_query_info, who_call="q2")
#                         answer2.append(ans2)
#                         print('Ans_2_'+ str(idx) + ':' + str(ans2))
#                         question2.append(q2_1)
#                         # info2.append(information)
#                 elif ("{Ans_2}" in q2):
#                     break
#                 else:
#                     information, info_relation_list, info_text_list = \
#                         await get_query_context_with_keywords(
#                         keywords=q2,
#                         knowledge_graph_inst=self.knowledge_graph_inst,
#                         relationships_vdb=self.relationships_vdb,
#                         text_chunks_db=self.text_chunks_db,
#                         query_param=self.q_param,
#                         use_model_func=self.global_config["llm_model_func"],
#                         need_info_list=True
#                         )
#                     self.info_relation_list.extend(info_relation_list)
#                     self.info_text_list.extend(info_text_list)
#                     sub_query_info = await self.get_sub_query_info(only_original_question=True)
#                     ans2 = await self.getting_response_from_info(information, q2, sub_query_info, who_call="q2")
#                     answer2.append(ans2)
#                     print('Ans_2:' + str(ans2))
#                     question2.append(q2)
#                     # info2.append(information)
#                 self.q2 = question2
#                 self.ans2 = answer2
#                 # self.info2 = info2
#                 print('Sub_Question2:' + q2)
#                 print('All Ans_2:' + str(self.ans2))
#                 if self.ans2:
#                     error_flag = True
#                 else:
#                     break
#             if('Sub_Question_3: str = ' in line):
#                 q3 = line.split('Sub_Question_3: str = ')[1].lstrip('f').strip("\"")
#                 print('Sub_Question_3:'+q3)
#                 question3 = []
#                 # info3 = []
#                 answer3 = []
#                 if len(self.ans1) == 1:
#                     ans_1 = self.ans1
#                 else:
#                     ans_1 = set()
#                     for s in self.ans1:
#                         if(s):
#                             ans_1 = ans_1.union(s)
#                 if len(self.ans2) == 1:
#                     ans_2 = self.ans2
#                 else:
#                     ans_2 = set()
#                     for s in self.ans2:
#                         if(s):
#                             ans_2 = ans_2.union(s)
#                 if ("{Ans_1}" in q3):
#                     for idx1, a in enumerate(ans_1):
#                         q3_1 = q3.replace("{Ans_1}",a)
#                         if("{Ans_2}" in q3_1):
#                             for idx2, b in enumerate(ans_2):
#                                 q3_2 = q3_1.replace("{Ans_2}",b)
#                                 print('Sub_Question_3_' + str(idx1) + '_' + str(idx2) + ':' +q3_2)
#                                 information, info_relation_list, info_text_list = \
#                                     await get_query_context_with_keywords(
#                                     keywords=q3_2,
#                                     knowledge_graph_inst=self.knowledge_graph_inst,
#                                     relationships_vdb=self.relationships_vdb,
#                                     text_chunks_db=self.text_chunks_db,
#                                     query_param=self.q_param,
#                                     use_model_func=self.global_config["llm_model_func"],
#                                     need_info_list=True
#                                     )
#                                 self.info_relation_list.extend(info_relation_list)
#                                 self.info_text_list.extend(info_text_list)
#                                 sub_query_info = await self.get_sub_query_info(only_original_question=True)
#                                 ans3 = await self.getting_response_from_info(information, q3_2, sub_query_info, who_call="q3")
#                                 print('Ans_3_'+ str(idx1) + '_' + str(idx2) + ':' + str(ans3))
#                                 question3.append(q3_2)
#                                 answer3.append(ans3)
#                                 # info3.append(information)
#                         else:
#                             print('Sub_Question_3_' + str(idx1) + ':' +q3_1)
#                             information, info_relation_list, info_text_list = \
#                                 await get_query_context_with_keywords(
#                                 keywords=q3_1,
#                                 knowledge_graph_inst=self.knowledge_graph_inst,
#                                 relationships_vdb=self.relationships_vdb,
#                                 text_chunks_db=self.text_chunks_db,
#                                 query_param=self.q_param,
#                                 use_model_func=self.global_config["llm_model_func"],
#                                 need_info_list=True
#                                 )
#                             self.info_relation_list.extend(info_relation_list)
#                             self.info_text_list.extend(info_text_list)
#                             sub_query_info = await self.get_sub_query_info(only_original_question=True)
#                             ans3 = await self.getting_response_from_info(information, q3_1, sub_query_info, who_call="q3")
#                             answer3.append(ans3)
#                             print('Ans_3_'+ str(idx1) + ':' + str(ans3))
#                             question3.append(q3)
#                             # info3.append(information)
#                 elif ("{Ans_2}" in q3):
#                     for idx,b in enumerate(ans_2):
#                         q3_1 = q3.replace("{Ans_2}",b)
#                         print('Sub_Question_3_' + str(idx) + ':' +q3_1)
#                         information, info_relation_list, info_text_list = \
#                             await get_query_context_with_keywords(
#                             keywords=q3_1,
#                             knowledge_graph_inst=self.knowledge_graph_inst,
#                             relationships_vdb=self.relationships_vdb,
#                             text_chunks_db=self.text_chunks_db,
#                             query_param=self.q_param,
#                             use_model_func=self.global_config["llm_model_func"],
#                             need_info_list=True
#                             )
#                         self.info_relation_list.extend(info_relation_list)
#                         self.info_text_list.extend(info_text_list)
#                         sub_query_info = await self.get_sub_query_info(only_original_question=True)
#                         ans3 = await self.getting_response_from_info(information, q3_1, sub_query_info, who_call="q3")
#                         # info3.append(information)
#                         question3.append(q3_1)
#                         answer3.append(ans3)
#                         print('Ans_3_'+ str(idx) + ':' + str(ans3))
#                 elif ("{Ans_3}" in q3):
#                     break
#                 else:
#                     information, info_relation_list, info_text_list = \
#                         await get_query_context_with_keywords(
#                         keywords=q3,
#                         knowledge_graph_inst=self.knowledge_graph_inst,
#                         relationships_vdb=self.relationships_vdb,
#                         text_chunks_db=self.text_chunks_db,
#                         query_param=self.q_param,
#                         use_model_func=self.global_config["llm_model_func"],
#                         need_info_list=True
#                         )
#                     self.info_relation_list.extend(info_relation_list)
#                     self.info_text_list.extend(info_text_list)
#                     sub_query_info = await self.get_sub_query_info(only_original_question=True)
#                     ans3 = await self.getting_response_from_info(information, q3, sub_query_info, who_call="q3")
#                     answer3.append(ans3)
#                     print('Ans_3:' + str(ans3))
#                     question3.append(q3)
#                     # info3.append(information)
#                 self.q3 = question3
#                 # self.info3 = info3
#                 self.ans3 = answer3
#                 print('Sub_Question3:' + q3)
#                 print('All Ans_3:' + str(self.ans3))
#                 if self.ans3:
#                     error_flag = True
#                 else:
#                     break
#             if('Final_Answer: str = ' in line):
#                 await self.Final_QA(line)

#         if (self.final_ans is None or len(self.final_ans) == 0 or error_flag == False):

#             sub_query_information = await self.get_sub_query_info()
   
#             final_response = await query_with_origional_keywords_(
#                 query=self.original_q,
#                 knowledge_graph_inst=self.knowledge_graph_inst,
#                 relationships_vdb=self.relationships_vdb,
#                 text_chunks_db=self.text_chunks_db,
#                 q_param=self.q_param,
#                 global_config=self.global_config,
#                 sub_query_info=sub_query_information
#             )
#             return final_response
#         else:
#             if isinstance(self.final_ans, set):
#                 final_ans = ", ".join(self.final_ans)
#             else:
#                 final_ans = self.final_ans
#             print('Final Answer:'+str(final_ans))
#             return str(final_ans)