from tqdm import tqdm
import re
import json
import time
from ner import NER_Model
from retriever import Retriever
from prompts import extract_relation_prompt
from prompts import prune_entity_prompt
from sentence_transformers import SentenceTransformer, util



# entity linking
# kg_entities: kg中的所有实体，是一个list
# retriever: 检索器
# 返回""或者是链接的实体
def check_match(kg_entities, retriever, entity):
    assert entity is not None
    if entity in kg_entities:
        return entity
    else:
        retrieve_res = retriever.retrieve(query=entity, top_k=5)
        if len(retrieve_res) != 0:
            return retrieve_res[0]['text']
        else:
            return ""


# 从问题中进行实体识别和实体链接，返回连接后的结果，注意链接后的结果可能为空字符串
# 返回的是可能有元素为空字符串的list
def get_topic_entities(query, kg_entities, retriever, ner_model):
    result = []
    extract_result = ner_model.ner(text=query)


    for ent in extract_result:
        for out in ent["output"]:
            mapped_ent = check_match(kg_entities, retriever, out['span'])
            result.append(mapped_ent)

    return list(set([x for x in result if x]))



# pre-defined cyphers
# 查询（出边）的所有关系名称
cypher_head_relations = """MATCH (n:Entity {id: '%s'})-[r]->()
WHERE r.name IS NOT NULL  
RETURN DISTINCT r.name as relation"""
# 查询“指向”当前实体（入边）的所有关系名称
cypher_tail_relations = """
MATCH (n:Entity {id: '%s'})<-[r]-()
WHERE r.name IS NOT NULL
RETURN DISTINCT r.name as relation
"""

def abandon_rels(relation):
    # TODO 完善这个关系过滤函数
    return False
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True

def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    return extract_relation_prompt + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations)

def select_relations(string, entity_id, head_relations, tail_relations):
    last_brace_l = string.rfind('[')
    last_brace_r = string.rfind(']')

    if last_brace_l < last_brace_r:
        string = string[last_brace_l:last_brace_r+1]

    relations=[]
    rel_list = eval(string.strip())
    for relation in rel_list:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "head": True})
        elif relation in tail_relations:
            relations.append({"entity": entity_id, "relation": relation, "head": False})

    if not relations:
        return False, []
    return True, relations

# dao: kgdao
# entity_id：要进行双向关系探索的实体的id
# entity_name: 要进行双向关系探索的实体的name 
# return: 返回的是探索出来的关系的name，是一个list,如： {'entity': 'aaa', 'head': True, 'relation': 'r'}, entity是实体的id，head代表方向，relation是关系的name
def relation_search_prune(dao, entity_id, entity_name, pre_relations, pre_head, question, chat_model, args):
    """
    核心功能：关系检索与剪枝
    输入：当前实体ID、子问题、实体名、上一跳关系历史、上一跳方向、原始问题、全局参数
    输出：筛选后的关系列表（包含方向信息）、消耗的Token数量
    """

    # --- 步骤 1: 物理检索 (KG Retrieval) ---
    # 构建 cypher 查询，查找以当前实体为“头实体”(Subject) 的所有出边关系
    # 对应三元组模式: (ns:entity_id ?relation ?x)
    cypher_relations_extract_head = cypher_head_relations % (entity_id)
    # 调用 execurte_cypher 执行查询，返回原始 JSON 结果
    head_relations = dao.execute(cypher_relations_extract_head)
    # [修改点 1]：将字典列表转换为纯关系名列表
    # 假设 dao.execute 返回格式为 [{'relation': 'rel_name1'}, {'relation': 'rel_name2'}]
    head_relations = [item['relation'] for item in head_relations]

    # 构建 cypher 查询，查找以当前实体为“尾实体”(Object) 的所有入边关系
    # 对应三元组模式: (?x ?relation ns:entity_id)
    cypher_relations_extract_tail= cypher_tail_relations % (entity_id)
    tail_relations = dao.execute(cypher_relations_extract_tail)
    # [修改点 2]：将字典列表转换为纯关系名列表
    tail_relations = [item['relation'] for item in tail_relations]

    # --- 步骤 2: 规则过滤 (Rule-based Filtering) ---
    # 如果开启了 remove_unnecessary_rel 开关，过滤掉无用的系统关系
    if args.remove_unnecessary_rel:
        # abandon_rels 函数用于判断关系是否为噪声（如 type.object.type, common.topic 等）
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]

    # --- 步骤 3: 历史路径剪枝 (Backtracking Prevention) ---
    # 防止搜索掉头走回头路。
    # pre_head=True 表示上一跳是通过“出边”来到当前实体的（Previous -> relation -> Current）。
    # 那么对于 Current 来说，这个 relation 是“入边”。为了不退回去，需要在 tail_relations 中移除它。
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        # 反之，如果上一跳是通过“入边”来的（Current -> relation -> Previous），
        # 那么对于 Current，这个 relation 是“出边”，需要在 head_relations 中移除。
        head_relations = list(set(head_relations) - set(pre_relations))

    # 去重处理
    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))

    # 合并出边和入边，形成提供给 LLM 的完整候选列表
    total_relations = head_relations + tail_relations
    # 关键步骤：排序。确保每次运行 LLM 时，候选关系的顺序一致，保证结果的可复现性，也能提高 LLM 缓存命中率
    total_relations.sort()


    # --- 步骤 4: LLM 语义剪枝 (LLM Reasoning) ---
    # 调用 construct_relation_prune_prompt 构建提示词
    # 提示词会包含：原始问题、分解后的子问题、当前实体名、以及上面准备好的 total_relations
    prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)

    # 调用llm
    # args.temperature_exploration 控制探索阶段的发散程度
    result = chat_model.chat_(prompt)

    # --- 步骤 5: 结果解析与方向恢复 (Parsing) ---
    # LLM 返回的是纯文本列表（如 "['relation_a', 'relation_b']"）。
    # select_relations 函数负责将这些字符串映射回具体的方向（是出边还是入边）。
    # 这一点非常重要，因为后续查询实体时 cypher 语句的方向不同。
    flag, retrieve_relations = select_relations(result, entity_id, head_relations, tail_relations)

    if flag:
        return retrieve_relations
    else:
        # 如果解析失败（格式错误）或没有选出任何关系，返回空列表
        return []


# 对应查询模式: (Subject/n) -[Predicate/r]-> (Object/m)
# 第一个 %s 填入 n.id (Subject)
# 第二个 %s 填入 r.name (Predicate)
cypher_tail_entities_extract = """
MATCH (n {id: '%s'})-[r]->(m)
WHERE r.name = '%s'
RETURN m.id as id, m.name as name
"""
# 对应查询模式: (Subject/m) -[Predicate/r]-> (Object/n)
# 我们要找的是 m (Subject/tailEntity)
# 第一个 %s 填入 r.name (Relation)
# 第二个 %s 填入 n.id (Entity/Object)
cypher_head_entities_extract = """
MATCH (m)-[r]->(n)
WHERE r.name = '%s' AND n.id = '%s'
RETURN m.id as id, m.name as name
"""
# entity_id: 实体的id
# relation: 关系的name
# return： 返回查询出的实体的id的list, 每一个元素是一个dict，如 {'id': 'id1', 'name': 'name1'}
def entity_search(dao, entity_id, relation, head=True):
    # 1. 根据关系的方向构建 cypher
    if head:
        # 如果当前实体是头实体(Subject)，我们要找尾实体(Object)
        # 对应查询模式: (current_entity, relation, ?tailEntity)
        tail_entities_extract = cypher_tail_entities_extract % (entity_id, relation)
        entities = dao.execute(tail_entities_extract)
    else:
        # 如果当前实体是尾实体(Object)，我们要找头实体(Subject)
        # 对应查询模式: (?tailEntity, relation, current_entity)
        head_entities_extract = cypher_head_entities_extract % (relation, entity_id)
        entities = dao.execute(head_entities_extract)

    return entities


# 使用distilbert生成相似度最高的几个候选
# query： 问题
# docs： list形式的实体
# model： distilbert, 本身是sentence_transformers的SentenceTransformer
def retrieve_top_docs(query, docs, model, width=3):
    query_emb = model.encode(query)
    doc_emb = model.encode(docs)
    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)
    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]
    return top_docs, top_scores

# entid_name: map, 实体的从id到name的映射，是一个dict
# name_entid: map, 实体的从name到id的映射，是一个dict
# ent_rel_ent_dict: 格式像这样的字典：
    # ere = {}
    # ere['e1'] = {}
    # ere['e1']['head'] = {}
    # ere['e1']['head']['r1'] = []
    # ere['e1']['head']['r1'].append(['e3', 'e4'])


    # ere['e1']['tail'] = {}
    # ere['e1']['tail']['r2'] = []
    # ere['e1']['tail']['r2'].append(['e5', 'e6'])
# model: 预训练的SentenceTransformer模型
# chat_model: llm调用
# TODO 返回的值是什么？
def entity_condition_prune(question, 
                           ent_rel_ent_dict, 
                           entid_name, 
                           name_entid,
                           chat_model,
                           model):
    """
    实体剪枝函数 (精简版)
    作用：结合 DistilBERT (针对大量实体) 和 LLM (针对语义匹配) 从候选实体中筛选出最相关的实体。

    参数:
    - question: 原始自然语言问题 (String)
    - ent_rel_ent_dict: 包含当前深度检索到的所有原始路径的字典 (Dict)
      结构: { 源实体ID: { 'head'/'tail': { 关系名: [候选实体ID列表] } } }
    - entid_name: ID 到 名字 的映射字典
    - name_entid: 名字 到 ID 的映射字典
    - args: 全局参数配置 (包含 max_length, temperature 等)
    - model: SentenceTransformer 模型对象 (用于 DistilBERT 相似度计算)

    返回:
    - flag: 是否成功找到有效实体 (Bool)
    - cluster_chain_of_entities: 筛选后的三元组路径链 (List)，格式是[['h1', 'r1', 't1'], ['h2', 'r2', 't2']]
    - filter_entities_id: 下一跳的实体 ID 列表 (List)
    - filter_relations: 下一跳的实体对应的上个关系列表 (List),比如 当前三元组从e1出发探索探索到e2,关系可能是e1->r1->e2也可能是e2->r1->e1，那么filter_entities_id[ii]=id(e2),filter_relations[ii]=r1
    - filter_head: 对应的方向列表 (List)
    - new_ent_rel_ent_dict: 剪枝后的结构化字典 (Dict),格式像这样的字典：
            # ere = {}
            # ere['e1'] = {}
            # ere['e1']['head'] = {}
            # ere['e1']['head']['r1'] = []
            # ere['e1']['head']['r1'].append(['e3', 'e4'])


            # ere['e1']['tail'] = {}
            # ere['e1']['tail']['r2'] = []
            # ere['e1']['tail']['r2'].append(['e5', 'e6'])
    """

    # 1. 初始化输出容器
    # new_ent_rel_ent_dict 将只存储筛选后留下的路径
    new_ent_rel_ent_dict = {}

    # 下面这些列表用于扁平化存储筛选结果，方便主循环使用
    filter_entities_id, filter_tops, filter_relations, filter_candidates, filter_head = [], [], [], [], []

    # 2. 遍历输入的原始字典 (三层嵌套循环)
    # 层级：当前实体 (topic_e_id) -> 方向 (h_t) -> 关系 (rela_name) -> 候选实体ID列表 (eid_list)
    for topic_e_id, h_t_dict in sorted(ent_rel_ent_dict.items()):
        for h_t, r_e_dict in sorted(h_t_dict.items()):
            for rela_name, eid_list in sorted(r_e_dict.items()):

                # 3. 策略一：基于相似度的大规模召回 (DistilBERT)
                # 如果候选实体数量太多 (>70)，直接扔给 LLM 会超出上下文窗口
                # TODO 修改阈值
                if len(eid_list) >= 1:
                    # 将 ID 转换为自然语言名称列表
                    sorted_ename_list = [entid_name[e_id] for e_id in eid_list]

                    # 使用 SentenceTransformer 计算 问题 与 实体名 的相似度
                    # retrieve_top_docs 假设是外部定义的函数，返回最相似的 top-n 实体名
                    topn_entities, _ = retrieve_top_docs(question, sorted_ename_list, model, 70)

                    # 将筛选后的实体名转回 ID，更新 eid_list
                    eid_list = [name_entid[e_n] for e_n in topn_entities]
                    # print('DistilBERT recalled entities:', topn_entities)

                # 4. 策略二：基于 LLM 的语义精确筛选
                # 准备 LLM 的上下文数据：将 ID 转为名字
                sorted_ename_list = [entid_name[e_id] for e_id in sorted(eid_list)]

                # 构建 Prompt
                # prune_entity_prompt 需要在外部定义 (如 prompt_list.py)
                # 格式示例: "问题: Q. 三元组: Taylor Swift Born_in ['USA', 'China']"
                prompt = prune_entity_prompt + question + '\nTriples: '
                # prompt += entid_name[topic_e_id] + ' ' + rela_name + ' ' + str(sorted_ename_list)
                sub = entid_name[topic_e_id]
                rel = rela_name
                prompt += " ".join([f"({sub}, {rel}, {obj})" for obj in sorted_ename_list])


                # 调用 LLM (移除 token 统计接收)
                # run_llm 假设返回 (生成文本, token字典)
                result = chat_model.chat_(prompt)

                # 5. 解析 LLM 的输出
                # LLM 通常返回如 "['USA', 'China']" 的字符串
                last_brace_l = result.rfind('[')
                last_brace_r = result.rfind(']')

                if last_brace_l < last_brace_r:
                    result = result[last_brace_l:last_brace_r+1]

                try:
                    # 尝试将字符串转为 Python List
                    result = eval(result.strip())
                except:
                    # 容错处理：手动分割字符串
                    result = result.strip().strip("[").strip("]").split(', ')
                    result = [x.strip("'") for x in result]

                # 过滤：确保 LLM 返回的实体确实在我们的候选列表中 (防止 LLM 幻觉生成不存在的词)
                select_ename = sorted(result)
                select_ename = [x for x in select_ename if x in sorted_ename_list]

                # 如果没有选中任何实体，跳过当前关系的构建
                if len(select_ename) == 0 or all(x == '' for x in select_ename):
                    continue

                # 6. 重建数据结构 (仅包含选中的实体)
                # 只有当路径有效时，才在字典中逐层创建 Key
                if topic_e_id not in new_ent_rel_ent_dict.keys():
                    new_ent_rel_ent_dict[topic_e_id] = {}
                if h_t not in new_ent_rel_ent_dict[topic_e_id].keys():
                    new_ent_rel_ent_dict[topic_e_id][h_t] = {}
                if rela_name not in new_ent_rel_ent_dict[topic_e_id][h_t].keys():
                    new_ent_rel_ent_dict[topic_e_id][h_t][rela_name] = []

                # 填充数据到字典和扁平列表
                for ent in select_ename:
                    # 再次确认 (虽然上面过滤过，双重保险)
                    if ent in sorted_ename_list:
                        # 字典存 ID (用于后续搜索)
                        new_ent_rel_ent_dict[topic_e_id][h_t][rela_name].append(name_entid[ent])

                        # 列表存用于记录的元数据
                        filter_tops.append(entid_name[topic_e_id]) # 源实体名
                        filter_relations.append(rela_name)           # 关系名
                        filter_candidates.append(ent)           # 选中实体名
                        filter_entities_id.append(name_entid[ent]) # 选中实体 ID

                        # 记录方向 (Bool)
                        if h_t == 'head':
                            filter_head.append(True)
                        else:
                            filter_head.append(False)

    # 7. 最终检查与返回
    # 如果所有分支都被剪枝完了，返回 False
    if len(filter_entities_id) == 0:
        return False, [], [], [], [], new_ent_rel_ent_dict

    # 构建路径链格式，用于 Memory 记录
    # 格式: [(Source, Relation, Target), ...]
    cluster_chain_of_entities = [[(filter_tops[i], filter_relations[i], filter_candidates[i]) for i in range(len(filter_candidates))]]

    # 返回精简后的结果 (移除了 cur_call_time, cur_token)
    return True, cluster_chain_of_entities, filter_entities_id, filter_relations, filter_head, new_ent_rel_ent_dict


import ast
import re
from prompts import answer_prompt


def answer(question, enames, cluster_chain_of_entities, chat_model):
    """
    根据问题、初始实体和推理链生成最终答案。
    """

    # --- 1. 定义递归展平函数 ---
    def flatten_triplets(data):
        """递归处理最高 N 层的嵌套列表，提取 tuple"""
        triplets = []
        if isinstance(data, list):
            for item in data:
                triplets.extend(flatten_triplets(item))
        elif isinstance(data, tuple):
            # 找到三元组 (h, r, t)
            if len(data) == 3:
                triplets.append(f"({data[0]}, {data[1]}, {data[2]})")
        return triplets

    # --- 2. 展平数据 ---
    all_triplets = flatten_triplets(cluster_chain_of_entities)

    # 去重处理，避免推理路径中有重复的三元组干扰 LLM
    seen = set()
    unique_triplets = [x for x in all_triplets if not (x in seen or seen.add(x))]

    triplets_str = "\n".join(unique_triplets) if unique_triplets else "No relevant triplets found."
    topic_entities_str = ", ".join(enames)

    # --- 3. 构建 Prompt (使用之前定义的 answer_prompt) ---
    prompt = answer_prompt.format(
        topic_entities=topic_entities_str,
        triplets=triplets_str,
        question=question
    )

    # --- 4. 调用 LLM ---
    try:
        response = chat_model.chat_(prompt)

        # --- 5. 解析输出 ---
        # 匹配最外层的 [ ... ]
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            list_str = match.group()
            # 安全转换为 Python 列表
            try:
                ans_list = ast.literal_eval(list_str)
                if isinstance(ans_list, list):
                    # 确保只取前三个，且元素转为字符串
                    return [str(name) for name in ans_list[:3]]
            except:
                # 如果 ast 失败，尝试通过逗号分隔
                clean_str = list_str.replace('[', '').replace(']', '').replace('"', '').replace("'", "")
                return [name.strip() for name in clean_str.split(',')][:3]

        # 兜底处理：如果没找到列表格式，将整个回复视为一个字符串答案
        return [response.strip()]

    except Exception as e:
        print(f"Error in utils.answer: {e}")
        return []

if __name__ == '__main__':
    # 1 test get_topic_entities
    ner_model = NER_Model(ner_model_id='iic/nlp_raner_named-entity-recognition_chinese-base-cmeee', device='cuda:0')
    retriever = Retriever(retrievel_type="dense", retriever_version="/home/jiebei/ywkgqa/.retriv/dr_corom_emb")
    kg_entities=["飞机", "大炮"]
    print("---------------------------------")
    print(get_topic_entities("请问大炮是什么", kg_entities, retriever, ner_model))
    print("---------------------------------")
    print(get_topic_entities("请问四合一手术是什么", kg_entities, retriever, ner_model))
    print("---------------------------------")
    
    # 2 test relation_search_prune
    # from kgdao import KGDao
    # import argparse
    # from llm import ChatModel


    # parser = argparse.ArgumentParser()
    # parser.add_argument("--remove_unnecessary_rel", type=bool,
    #                     default=True, help="whether removing unnecessary relations.")
    # args = parser.parse_args()
    # neo4j_url = "neo4j://localhost:7687"
    # neo4j_username = "neo4j"
    # neo4j_password = "KOBEforever668!"
    # neo4j_database = "ywkgqa"
    # dao = KGDao(neo4j_url, neo4j_username, neo4j_password, neo4j_database)
    # provider = "ollama"
    # base_url = "http://localhost:11434"
    # model_name = "qwen3:8b"

    # chat_model = ChatModel(provider=provider, model_name=model_name, base_url=base_url)
    # res = relation_search_prune(dao, "id1", "name1", pre_relations=[], pre_head=True, question="aaa", chat_model=chat_model, args=args)

    # aaa = 1

    # 3 test entity_search
    # from kgdao import KGDao
    # import argparse
    # from llm import ChatModel


    # neo4j_url = "neo4j://localhost:7687"
    # neo4j_username = "neo4j"
    # neo4j_password = "KOBEforever668!"
    # neo4j_database = "ywkgqa"
    # dao = KGDao(neo4j_url, neo4j_username, neo4j_password, neo4j_database)
    # res = entity_search(dao, "id1", "name2", False)

    # aaa = 1


    # 4 test entity_condition_prune
    # from llm import ChatModel
    # question = "aaa"
    # ere = {}
    # ere['e1'] = {}
    # ere['e1']['head'] = {}
    # ere['e1']['head']['r1'] = []
    # ere['e1']['head']['r1'].extend(['e3', 'e4'])


    # ere['e1']['tail'] = {}
    # ere['e1']['tail']['r2'] = []
    # ere['e1']['tail']['r2'].extend(['e5', 'e6'])

    # entid_name = {'e1': 'e1',  'e3': 'e3', 'e4': 'e4', 'e5': 'e5', 'e6': 'e6'}
    # name_entid = {'e1': 'e1', 'e3': 'e3', 'e4': 'e4', 'e5': 'e5', 'e6': 'e6'}
    # # 就算断网也没关系，只要缓存里有，它会自动加载
    # model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')

    # provider = "ollama"
    # base_url = "http://localhost:11434"
    # model_name = "qwen3:8b"

    # chat_model = ChatModel(provider=provider, model_name=model_name, base_url=base_url)
    # res = entity_condition_prune(question, ere, entid_name, name_entid, chat_model, model)
    # aaa = 4

 




