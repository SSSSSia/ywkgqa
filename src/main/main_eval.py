import json
import argparse
from sentence_transformers import SentenceTransformer, util
# 假设这些模块在你的环境中可用
from kgdao import KGDao
from llm import ChatModel
import utils


def calculate_metrics(predicted_ans, golden_ans):
    """
    计算基于包含关系的评价指标。
    由于预测和真实答案都是字符串，这里我们将预测视为一个整体。
    如果预测包含真实答案，则认为 Recall 为 1，反之则为 0。
    """
    if not predicted_ans:
        return 0, 0, 0

    # 简单的包含逻辑：如果真实答案在预测字符串中
    # 也可以根据需求进一步拆分字符串（如 split(',')）进行更细粒度的计算
    is_correct = golden_ans.lower() in predicted_ans.lower()

    precision = 1.0 if is_correct else 0.0
    recall = 1.0 if is_correct else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


if __name__ == '__main__':
    # --- 1. 环境与设置 ---
    class Args:
        def __init__(self):
            self.remove_unnecessary_rel = True
            self.temperature_exploration = 0.3


    args = Args()

    # 数据库与模型设置
    dao = KGDao("neo4j://localhost:7687", "neo4j", "KOBEforever668!", "ywkgqa")
    chat_model = ChatModel(provider="ollama", model_name="qwen3:8b", base_url="http://localhost:11434")
    model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')

    # --- 2. 批量处理初始化 ---
    data_path = "../data/test_processed.json"
    total_precision, total_recall, total_f1 = 0, 0, 0
    count = 0

    print(f"开始处理数据集: {data_path}\n" + "-" * 30)

    # --- 3. 遍历 JSON Lines 数据 ---
    with open(data_path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                question = data["question"]
                eids = data["eids"]
                enames = data["enames"]
                golden_answer = data["answer entity"]

                # --- 重置每个问题的状态 ---
                eid2name = dict(zip(eids, enames))
                name2eid = dict(zip(enames, eids))
                pre_relations = []
                pre_heads = [-1] * len(eids)
                cluster_chain_of_entities = []
                current_eids = list(eids)

                print(f"开始 QA 循环，问题: {question}")
                print(f"初始实体: {eid2name}")

                # --- 4. 推理循环 (Max 5 hops) ---
                for depth in range(1, 6):
                    print(f"\n=== 跳数深度 {depth} ===")
                    current_entity_relations_list = []
                    for i, eid in enumerate(current_eids):
                        results = utils.relation_search_prune(
                            dao, eid, eid2name.get(eid, "Unknown"),
                            pre_relations, pre_heads[i],
                            question, chat_model, args
                        )
                        current_entity_relations_list.extend(results)

                    if not current_entity_relations_list:
                        break

                    ent_rel_ent_dict = {}
                    for ent_rel in current_entity_relations_list:
                        curr_entity_id = ent_rel['entity']
                        relation = ent_rel['relation']
                        is_head = ent_rel['head']
                        direction = 'head' if is_head else 'tail'

                        found_entities = utils.entity_search(dao, curr_entity_id, relation, is_head)
                        found_ids = []
                        for item in found_entities:
                            f_id, f_name = item['id'], item['name']
                            eid2name[f_id] = f_name
                            name2eid[f_name] = f_id
                            found_ids.append(f_id)

                        # 填充结构并进行嵌套初始化
                        if curr_entity_id not in ent_rel_ent_dict:
                            ent_rel_ent_dict[curr_entity_id] = {}

                        if direction not in ent_rel_ent_dict[curr_entity_id]:
                            ent_rel_ent_dict[curr_entity_id][direction] = {}

                        ent_rel_ent_dict[curr_entity_id][direction][relation] = found_ids



                    # 剪枝
                    flag, chain_of_entities, entities_id, filter_relations, filter_head, _ = utils.entity_condition_prune(
                        question, ent_rel_ent_dict, eid2name, name2eid, chat_model, model
                    )

                    cluster_chain_of_entities.append(chain_of_entities)
                    current_eids = entities_id
                    pre_relations = filter_relations
                    pre_heads = filter_head
                    print(f"深度 {depth} 结果 - 继续: {flag}, 实体数量: {len(eids)}")
                    if not flag:
                        break

                # --- 5. 获取预测答案与评估 ---
                predicted_answer = utils.answer(question, enames, cluster_chain_of_entities, chat_model)

                p, r, f1 = calculate_metrics(predicted_answer[0], golden_answer)
                total_precision += p
                total_recall += r
                total_f1 += f1
                count += 1

                print(f"序号: {index} | 预测: {predicted_answer} | 真实: {golden_answer} | F1: {f1:.2f}")

            except Exception as e:
                print(f"序号: {index} | 处理失败，原因: {str(e)}")
                continue

    # --- 6. 最终统计 ---
    if count > 0:
        avg_p = total_precision / count
        avg_r = total_recall / count
        avg_f1 = total_f1 / count
        print("-" * 30)
        print(f"最终结果 ({count} 个样本):")
        print(f"Average Precision: {avg_p:.4f}")
        print(f"Average Recall:    {avg_r:.4f}")
        print(f"Average F1 Score:  {avg_f1:.4f}")
    else:
        print("未成功处理任何数据。")