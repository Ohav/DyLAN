import math
import random
import re
from utils import parse_code_completion, generate_answer, parse_judge_attitude, try_int, get_score
import numpy as np
from prompt_lib import *
import pickle
import torch


class LLMNeuron:
    
    def __init__(self, role, mtype="gpt-3.5-turbo", ans_parser=parse_code_completion, qtype="single_choice"):
        self.role = role
        self.mtype = mtype
        self.qtype = qtype
        self.ans_parser = ans_parser
        self.reply = None
        self.answer = ""
        self.active = False
        self.importance = 0
        self.to_edges = []
        self.from_edges = []
        self.question = None
        self.last_logprobs = None

        if mtype == "gpt-3.5-turbo":
            self.model = "gpt-3.5-turbo-0301"
        elif mtype == "gpt-4":
            self.model = "gpt4"
        elif 'llama' in mtype:
            self.model = mtype
        else:
            raise NotImplementedError(f"Error init model type: {mtype}")

        def find_array(text):
            # Find all matches of array pattern
            matches = re.findall(r'\[\[(.*?)\]\]', text)
            if matches:
                # Take the last match and remove spaces
                last_match = matches[-1].replace(' ', '')
                # Convert the string to a list of integers
                try:
                    ret = list(map(int, last_match.split(',')))
                except:
                    ret = []
                return ret
            else:
                return []
        self.weights_parser = find_array
        
        self.prompt_tokens = 0
        self.completion_tokens = 0

        self.monitor_info = {'Programmer': {'path': '/home/morg/students/ohavbarbi/DyLAN/monitors/first_train/Programmer/poly_ent_kur_deg_5.pkl', 'threshold': 0.15}, 'AlgorithmDeveloper': {'path': '/home/morg/students/ohavbarbi/DyLAN/monitors/first_train/AlgorithmDeveloper/poly_ent_deg_1.pkl', 'threshold': 0.26}, 'ComputerScientist': {'path': '/home/morg/students/ohavbarbi/DyLAN/monitors/first_train/ComputerScientist/poly_ent_kur_deg_4.pkl', 'threshold': 0.1}, 'PythonAssistant': {'path': '/home/morg/students/ohavbarbi/DyLAN/monitors/first_train/PythonAssistant/poly_kur_deg_2.pkl', 'threshold': 0.3}}
        with open(self.monitor_info[self.role]['path'], 'rb') as f:
            self.monitor_info[self.role]['monitor'] = pickle.load(f)
        
        self.reset_count = 1

        self.judge_roles = None


    def get_reply(self):
        return self.reply

    def get_answer(self):
        return self.answer

    def deactivate(self):
        self.active = False
        self.reply = None
        self.answer = ""
        self.question = None
        self.importance = 0

        self.prompt_tokens = 0
        self.completion_tokens = 0

    def monitor(self, logprobs, rid):
        if self.reset_count <= 0:
            return False
        if rid < 0:
            return False
        print("-------------------------------------------------------------------")
        print("Start monitor")
        limit = 9
        collected = {k: [] for k in self.judge_roles}
        if self.role not in ["PythonAssistant", "AlgorithmDeveloper"]:
            collected['[['] = []

        for i in range(len(logprobs) - 1):
            for text in collected.keys():
                first_token = logprobs[i][0][0].lower()
                if (text in first_token) or ((len(text) > 3 and (first_token in text))):
                    index = i
                    # found = try_int(logprobs[index][0][0].strip()) >= 0
                    while index < i + limit:
                        index = index + 1
                        if index >= len(logprobs):
                            break
                        found = try_int(logprobs[index][0][0].strip()) >= 0
                        if found:
                            tokens = [p[0] for p in logprobs[index]]
                            index_logprobs = np.array([p[1] for p in logprobs[index]])
                            collected[text].append({'tokens': tokens, 'logprobs': index_logprobs})

        scores = {'entropy': [], 'varentropy': [], 'kurtosis': []}
        for k in collected.keys():
            for i in range(len(collected[k])):
                for s in scores.keys():
                    scores[s].append(get_score(collected[k][i]['logprobs'], s))
        
        if len(scores['entropy']) == 0:
            print("No scores for monitor")
            return False
        row = {'max_entropy': max(scores['entropy']), 
               'max_varentropy': max(scores['varentropy']),
               'max_kurtosis': max(scores['kurtosis']),
               'turn_count': rid}

        features = torch.Tensor([row['turn_count'], row['max_entropy'], row['max_varentropy'], row['max_kurtosis']]).reshape(1, -1)
        print(features)
        monitor_score = self.monitor_info[self.role]['monitor'](features)
        result = monitor_score > self.monitor_info[self.role]['threshold']
        
        if result:
            print(f"Decided to reset! Score {monitor_score} > {self.monitor_info[self.role]['threshold']}")
            self.reset_count -= 1
        
        print("-------------------------------------------------------------------")
        print()
        print()
        print()
        return result

    def activate(self, question, rid=-1):
        self.question = question
        self.active = True
        # get context and genrate reply
        contexts, formers = self.get_context()
        # print("formers: ", formers)
        # shuffle
        original_idxs = [mess[1] for mess in formers]
        random.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers = [mess[0] for mess in formers]
        # print("shuffled: ", shuffled_idxs)

        contexts.append(construct_message(formers, question, self.qtype))
        print(contexts)
        self.reply, self.prompt_tokens, self.completion_tokens, self.last_logprobs = generate_answer(contexts, self.model)
        print(self.get_reply())
        monitor_result = self.monitor(self.last_logprobs, rid)
        if monitor_result:
            # Reset!
            return False
        # parse answer
        self.answer = self.ans_parser(self.reply, question)
        weights = self.weights_parser(self.reply)
        if len(weights) != len(formers) - 1:
            weights = [0 for _ in range(len(formers))]
        else:
            res_weights = []
            if formers[0].role == "Ranker":
                res_weights.append(3)
            for wid, weight in enumerate(weights):
                res_weights.append(weight)
                if formers[wid + 1].role == "Ranker":
                    res_weights.append(3)
            weights = res_weights
        
        shuffled_pairs = list(zip(shuffled_idxs, weights, formers))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        weights, formers = [weight for _, weight, _ in sorted_pairs], [(former, eid) for eid, _, former in sorted_pairs]

        lp = 0
        for _, eid in formers:
            self.from_edges[eid].weight = weights[lp] / 5 if 0 < weights[lp] <= 5 else (1 if weights[lp] > 5 else 0)
            lp += 1
        # normalize weights
        total = sum([self.from_edges[eid].weight for _, eid in formers])
        if total > 0:
            for _, eid in formers:
                self.from_edges[eid].weight /= total
        else:
            for _, eid in formers:
                self.from_edges[eid].weight = 1 / len(formers)

        print(self.answer)
        print([edge.weight for edge in self.from_edges])
        return True
        
    def get_context(self):
        if self.qtype == "single_choice":
            sys_prompt = ROLE_MAP[self.role] + "\n" + SYSTEM_PROMPT_MMLU
        elif self.qtype == "code_completion":
            if len(self.from_edges) == 0:
                sys_prompt = ROLE_MAP_INIT[self.role]
            else:
                sys_prompt = ROLE_MAP[self.role]
        else:
            raise NotImplementedError("Error init question type")
        contexts = [{"role": "system", "content": sys_prompt}]
        
        formers = [(edge.a1, eid) for eid, edge in enumerate(self.from_edges) if edge.a1.reply is not None and edge.a1.active]
        return contexts, formers
        
    def get_conversation(self):
        if not self.active:
            return []

        contexts, formers = self.get_context()
        contexts.append(construct_message([mess[0] for mess in formers], self.question, self.qtype))
        contexts.append({"role": "assistant", "content": self.reply})
        return contexts


class JudgeNeuron:
    
    def __init__(self, role, mtype="gpt-3.5-turbo", ans_parser=parse_judge_attitude, qtype="single_choice"):
        self.role = role
        self.mtype = mtype
        self.qtype = qtype
        self.ans_parser = ans_parser
        self.reply = None
        self.answer = ""
        self.active = False
        self.importance = 0
        self.to_edges = []
        self.from_edges = []
        self.question = None
        self.resp_cost = 1 if role not in TOOL_LIST else 0
        self.last_logprobs = None

        if mtype == "gpt-3.5-turbo":
            self.model = "gpt-3.5-turbo-0301"
        elif mtype == "gpt-4":
            self.model = "gpt4"
        elif 'llama' in mtype:
            self.model = mtype
        else:
            raise NotImplementedError("Error init model type")

        def find_array(text, formers):
            if self.role == "Ranker":
                results = []
                for former in formers:
                    if self.answer[former]:
                        results.append(5)
                    else:
                        results.append(0)
                return results
            if self.role == "Passer":
                results = []
                for former in formers:
                    if self.answer[former] == "The code doesn't have syntax error.":
                        results.append(5)
                    else:
                        results.append(0)
                return results
            if self.role == "Tester":
                results = []
                for former in formers:
                    if "Passed tests:\n" not in self.answer[former]:
                        results.append(0)
                    else:
                        total_tests = len(self.answer[former].splitlines()) - 1
                        if "Failed tests:\n" in self.answer[former]:
                            total_tests -= 2
                        flag_pass = False
                        pass_tests = 0
                        for line in self.answer[former].splitlines():
                            if flag_pass:
                                if "Failed tests:" in line:
                                    break
                                pass_tests += 1
                            if "Passed tests:" in line:
                                flag_pass = True
                        pass_tests -= 1
                        results.append(math.ceil(pass_tests / total_tests * 5))
                return results

            if self.role != "Reflector" and self.role != "Debugger" and self.role != "QualityManager":
                raise NotImplementedError("Error init role type")

            # Find all matches of array pattern
            matches = re.findall(r'\[\[(.*?)\]\]', text)
            if matches:
                # Take the last match and remove spaces
                last_match = matches[-1].replace(' ', '')
                # Convert the string to a list of integers
                try:
                    ret = list(map(int, last_match.split(',')))
                except:
                    ret = []
                return ret
            else:
                return []
        self.weights_parser = find_array

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.unit_tests = []

    def get_reply(self):
        return self.reply

    def get_answer(self):
        return self.answer
    
    def get_unit_tests(self):
        return self.unit_tests

    def deactivate(self):
        self.active = False
        self.reply = None
        self.answer = ""
        self.question = None
        self.importance = 0

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.unit_tests = []

    def activate(self, question, rid=-1):
        self.question = question
        self.active = True
        # get context and genrate reply
        contexts, formers = self.get_context()
        # print("formers: ", formers)
        # shuffle
        original_idxs = [mess[1] for mess in formers]
        random.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        formers = [mess[0] for mess in formers]
        # print("shuffled: ", shuffled_idxs)

        if self.role == "Ranker" and len(formers) <= 2:
            self.reply = "[1, 2]"
            print(self.get_reply())

        elif self.role not in TOOL_LIST:
            contexts.append(construct_judge_message(formers, question, self.qtype, self.role))
            print(contexts)
            self.reply, self.prompt_tokens, self.completion_tokens, self.last_logprobs = generate_answer(contexts, self.model)
            print(self.get_reply())
        else:
            self.reply = formers
            print(self.get_reply())

        # parse answer
        self.answer = self.ans_parser(self.reply, question, self.role, formers)
        if self.role == "Tester":
            self.answer, self.unit_tests = self.answer

        weights = self.weights_parser(self.reply, formers)
        if len(weights) != len(formers):
            weights = [0 for _ in range(len(formers))]
        
        shuffled_pairs = list(zip(shuffled_idxs, weights, formers))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        weights, formers = [weight for _, weight, _ in sorted_pairs], [(former, eid) for eid, _, former in sorted_pairs]

        lp = 0
        for _, eid in formers:
            self.from_edges[eid].weight = weights[lp] / 5 if 0 < weights[lp] <= 5 else (1 if weights[lp] > 5 else 0)
            lp += 1
        # normalize weights
        total = sum([self.from_edges[eid].weight for _, eid in formers])
        if total > 0:
            for _, eid in formers:
                self.from_edges[eid].weight /= total
        else:
            for _, eid in formers:
                self.from_edges[eid].weight = 1 / len(formers)

        print(self.answer)
        print([edge.weight for edge in self.from_edges])
        
    def get_context(self):
        if self.qtype == "code_completion":
            sys_prompt = JUDGE_MAP[self.role]
        else:
            raise NotImplementedError("Error init question type")
        contexts = [{"role": "system", "content": sys_prompt}]
        
        formers = [(edge.a1.answer, eid) for eid, edge in enumerate(self.from_edges) if edge.a1.reply is not None and edge.a1.active]
        return contexts, formers
        
    def get_conversation(self):
        if not self.active:
            return []

        contexts, formers = self.get_context()
        contexts.append(construct_message([mess[0] for mess in formers], self.question, self.qtype))
        contexts.append({"role": "assistant", "content": self.reply})
        return contexts


class LLMEdge:

    def __init__(self, a1, a2):
        self.weight = 0
        self.a1 = a1
        self.a2 = a2
        self.a1.to_edges.append(self)
        self.a2.from_edges.append(self)

    def zero_weight(self):
        self.weight = 0
