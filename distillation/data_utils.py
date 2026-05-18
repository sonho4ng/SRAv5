from torch.nn.functional import pad
from torch.utils.data import Dataset
import pandas as pd
import torch
import json
from transformers import PreTrainedTokenizer

from dataclasses import dataclass


N_SPAN = 1024


def longest_common_subsequence(a, b, s_i=0, s_j=0) -> list:
    a = a.numpy()
    b = b.numpy()
    m, n = len(a), len(b)
    
    i = s_i
    j = s_j
    result = []

    while i < m and j < n:
        if a[i][1] == 0:
            i += 1
            continue
        if b[j][1] == 0:
            j += 1
            continue
            
        if a[i][1] == b[j][1]:
            result.append((i+1, j+1))
            i += 1
            j += 1
        elif a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1

        if i+1 < m and j+1 < n:
            if (a[i][1] == 0 and a[i+1][1] > 0) or (b[j][1] == 0 and b[j+1][1] > 0):
                while j < n and b[j][1] > 0:
                    j += 1

                while i < m and a[i][1] > 0:
                    i += 1

                i += 1
                j += 1
                result.append((i, j))


    size = len(result)
    # if size > 256:
    if size > N_SPAN:
        step = size / N_SPAN
        return [result[int((i + 1) * step) - 1] for i in range(N_SPAN)]
    # else:
    #     new_result = [result[i] for i in range(size - 1, -1, -16)]
    #     return new_result[::-1]
            
    return result

def get_pooler_tensor(segments_idxs):
    # Tạo chỉ số segment đã pad cho toàn bộ batch
    padded_idx_batch = []
    max_seg, max_len_all = 0, 0
    pad_multiple = 4

    for seg_idx, max_len in segments_idxs:
        max_len_all = max(max_len_all, max_len)
        max_seg = max(max_seg, len(seg_idx))

        padded = torch.stack([
            pad(x, (0, max_len - len(x)), value=-1)
            for x in seg_idx
        ])  # (num_segments, max_len)

        padded_idx_batch.append(padded)

    # Pad toàn bộ batch về cùng shape (B, max_seg, max_len_all)
    def pad2d(t, h, w):
        return pad(t, (0, w - t.size(1), 0, h - t.size(0)), value=-1)

    # max_seg = int(math.ceil(max_seg / pad_multiple) * pad_multiple)
    padded_idx_batch = torch.stack([
        pad2d(p, max_seg, max_len_all) for p in padded_idx_batch
    ])  # (B, max_seg, max_len_all)

    # Tạo mask và gather từ X
    mask = padded_idx_batch != -1
    safe_idx = padded_idx_batch.masked_fill(~mask, 0)

    return {'safe_idx': safe_idx, 'mask': mask}


TEACHER_OFFSET = 0
STUDENT_OFFSET = 0


# custom setup for orther tokenizers
def prepare_pooler(student_padding_side, student_offset_mapping, student_masks,
                   teacher_padding_side, teacher_offset_mapping, teacher_masks, 
                   student_start=-1, teacher_start=-1):
    student_seg_idxs, teacher_seg_idxs = [], []
    for student_offset, student_mask, teacher_offset, teacher_mask in zip(student_offset_mapping, student_masks,
                                                                          teacher_offset_mapping, teacher_masks):

        student_seg_idx = []
        teacher_seg_idx = []

        # Modify for orther tokenizers
        if student_padding_side == 'left':
            student_start = (len(student_mask) - sum(student_mask) + STUDENT_OFFSET).item()
        else:
            student_start = STUDENT_OFFSET
        # student_seg_idx = [torch.arange(0, 1)]

        if teacher_padding_side == 'left':
            teacher_start = (len(teacher_mask) - sum(teacher_mask) + TEACHER_OFFSET).item()
        else:
            teacher_start = TEACHER_OFFSET
        # teacher_seg_idx = [torch.arange(teacher_start - 1, teacher_start)]
       

        token_offset_start = [(student_start, teacher_start)]

        # longest_common_offset = self.longest_common_subsequence(student_offset, teacher_offset)

        longest_common_offset = token_offset_start + longest_common_subsequence(student_offset, teacher_offset, 
                                                                                student_start, teacher_start) 
        student_max_len, teacher_max_len = 1, 1

        for i in range(1, len(longest_common_offset)):
            student_seg_idx.append(torch.arange(longest_common_offset[i - 1][0], longest_common_offset[i][0]))
            teacher_seg_idx.append(torch.arange(longest_common_offset[i - 1][1], longest_common_offset[i][1]))
            student_max_len = max(student_max_len, student_seg_idx[-1].size(0))
            teacher_max_len = max(teacher_max_len, teacher_seg_idx[-1].size(0))

        student_seg_idxs.append((student_seg_idx, student_max_len))
        teacher_seg_idxs.append((teacher_seg_idx, teacher_max_len))

    return get_pooler_tensor(student_seg_idxs), get_pooler_tensor(teacher_seg_idxs)


def prepare_pooler_v2(student_starts, student_offset_mapping,
                      teacher_starts, teacher_offset_mapping):
    student_seg_idxs, teacher_seg_idxs = [], []
    for student_start, student_offset, teacher_start, teacher_offset in zip(student_starts, student_offset_mapping,
                                                                            teacher_starts, teacher_offset_mapping):

        student_seg_idx = []
        teacher_seg_idx = []

        student_start, teacher_start = student_start.item(), teacher_start.item()
       
        token_offset_start = [(student_start, teacher_start)]

        longest_common_offset = token_offset_start + longest_common_subsequence(student_offset, teacher_offset, 
                                                                                student_start, teacher_start) 
        student_max_len, teacher_max_len = 1, 1

        for i in range(1, len(longest_common_offset)):
            student_seg_idx.append(torch.arange(longest_common_offset[i - 1][0], longest_common_offset[i][0]))
            teacher_seg_idx.append(torch.arange(longest_common_offset[i - 1][1], longest_common_offset[i][1]))
            student_max_len = max(student_max_len, student_seg_idx[-1].size(0))
            teacher_max_len = max(teacher_max_len, teacher_seg_idx[-1].size(0))

        student_seg_idxs.append((student_seg_idx, student_max_len))
        teacher_seg_idxs.append((teacher_seg_idx, teacher_max_len))

    return get_pooler_tensor(student_seg_idxs), get_pooler_tensor(teacher_seg_idxs)


class ClassificationDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index]['text'], self.dataset.iloc[index]['label']
    
class SentencePairClassificationDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return ((self.dataset.iloc[index]['premise'], 
                self.dataset.iloc[index]['hypothesis']), 
                self.dataset.iloc[index]['label'])
    
class STSDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return ((self.dataset.iloc[index]['sentence1'], 
                self.dataset.iloc[index]['sentence2']),
                self.dataset.iloc[index]['score'])
    
class BiSTSDataset(Dataset):
    def __init__(self, file_path):
        self.dataset = pd.read_csv(file_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (self.dataset.iloc[index]['sentence1'], 
                self.dataset.iloc[index]['sentence2'],
                self.dataset.iloc[index]['score'])


@dataclass
class BiDataCollator:
    student_tokenizer: PreTrainedTokenizer = None
    teacher_tokenizer: PreTrainedTokenizer = None
    do_train: bool = True
    max_len: int = 512
    pad_to_multiple_of: int = 4
    return_tensors: str = 'pt'
    padding: bool = True
    return_offsets_mapping: bool = True
    n_span: int = 4


    def __call__(self, batch):
        text1s, text2s, scores = [], [], []
        for text1, text2, label in batch:
            text1s.append(text1)
            text2s.append(text2)
            scores.append(label)
        
        sentence_inp1s = self.prepare(text1s)
        sentence_inp2s = self.prepare(text2s)

        return sentence_inp1s, sentence_inp2s, torch.tensor(scores)
    
    def prepare(self, texts):
        student_inputs = self.student_tokenizer(
            texts,
            truncation=True,
            padding=self.padding,
            max_length=self.max_len,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_offsets_mapping=self.return_offsets_mapping and self.do_train
        )

        if not self.do_train:
            return student_inputs, None

        student_token_offset_mapping = student_inputs.pop('offset_mapping')

        teacher_inputs = self.teacher_tokenizer(
            texts,
            truncation=True,
            padding=self.padding,
            max_length=self.max_len,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_offsets_mapping=self.return_offsets_mapping
        )

        teacher_token_offset_mapping = teacher_inputs.pop('offset_mapping')

        student_pooler_tensor, teacher_pooler_tensor = prepare_pooler(self.student_tokenizer.padding_side,
                                                                      student_token_offset_mapping,
                                                                      student_inputs['attention_mask'],
                                                                      self.teacher_tokenizer.padding_side,
                                                                      teacher_token_offset_mapping,
                                                                      teacher_inputs['attention_mask'])

        student_inputs['pooler_safe_idx'] = student_pooler_tensor['safe_idx']
        student_inputs['pooler_mask'] = student_pooler_tensor['mask']
        teacher_inputs['pooler_safe_idx'] = teacher_pooler_tensor['safe_idx']
        teacher_inputs['pooler_mask'] = teacher_pooler_tensor['mask']

        return student_inputs, teacher_inputs


@dataclass
class DataCollator:
    student_tokenizer: PreTrainedTokenizer = None
    teacher_tokenizer: PreTrainedTokenizer = None
    do_train: bool = True
    max_len: int = 512
    pad_to_multiple_of: int = 4
    return_tensors: str = 'pt'
    padding: bool = True
    return_offsets_mapping: bool = True
    n_span: int = 4


    def __call__(self, batch):
        texts, labels = [], []
        for text, label in batch:
            texts.append(text)
            labels.append(label)
        
        student_inputs = self.student_tokenizer(
            texts,
            truncation=True,
            padding=self.padding,
            max_length=self.max_len,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_offsets_mapping=self.return_offsets_mapping and self.do_train
        )

        if not self.do_train:
            return student_inputs, None, torch.tensor(labels)

        student_token_offset_mapping = student_inputs.pop('offset_mapping')

        teacher_inputs = self.teacher_tokenizer(
            texts,
            truncation=True,
            padding=self.padding,
            max_length=self.max_len,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_offsets_mapping=self.return_offsets_mapping
        )
        teacher_token_offset_mapping = teacher_inputs.pop('offset_mapping')

        student_pooler_tensor, teacher_pooler_tensor = prepare_pooler(self.student_tokenizer.padding_side,
                                                                      student_token_offset_mapping,
                                                                      student_inputs['attention_mask'],
                                                                      self.teacher_tokenizer.padding_side,
                                                                      teacher_token_offset_mapping,
                                                                      teacher_inputs['attention_mask'])

        student_inputs['pooler_safe_idx'] = student_pooler_tensor['safe_idx']
        student_inputs['pooler_mask'] = student_pooler_tensor['mask']
        teacher_inputs['pooler_safe_idx'] = teacher_pooler_tensor['safe_idx']
        teacher_inputs['pooler_mask'] = teacher_pooler_tensor['mask']

        return student_inputs, teacher_inputs, torch.tensor(labels)


class LLMDataset(Dataset):
    def __init__(self, file_path, student_tokenizer, teacher_tokenizer, prompt_max_len=512):

        self.dataset = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.dataset.append(data)

                s_prompt = student_tokenizer(
                    data['prompt'], 
                    max_length=prompt_max_len,
                    truncation=True,
                    add_special_tokens=False
                )
                data['prompt'] = student_tokenizer.decode(s_prompt['input_ids'])
                data['s_prompt_len'] = len(s_prompt['input_ids'])

                t_prompt = teacher_tokenizer(data['prompt'], add_special_tokens=False)
                data['t_prompt_len'] = len(t_prompt['input_ids'])


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return (self.dataset[index]['prompt'], self.dataset[index]['output'], 
                self.dataset[index]['s_prompt_len'], self.dataset[index]['t_prompt_len'])
    

@dataclass
class LLMDataCollator:
    student_tokenizer: PreTrainedTokenizer = None
    teacher_tokenizer: PreTrainedTokenizer = None
    do_train: bool = True
    max_len: int = 512
    pad_to_multiple_of: int = 4
    return_tensors: str = 'pt'
    padding: bool = True
    return_offsets_mapping: bool = True
    n_span: int = 4


    def __call__(self, batch):
        prompts, fulls, prompt_lengths, teacher_prompt_lengths = [], [], [], []
        for prompt, output, prompt_length, teacher_prompt_length in batch:
            prompts.append(prompt)
            fulls.append(prompt + output)
            prompt_lengths.append(prompt_length)
            teacher_prompt_lengths.append(teacher_prompt_length)

        
        student_inputs = self.student_tokenizer(
            fulls,
            truncation=True,
            padding=self.padding,
            max_length=self.max_len,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_offsets_mapping=self.return_offsets_mapping and self.do_train,
            add_special_tokens=False
        )

        # tokenized_prompts = self.student_tokenizer(
        #     prompts, 
        #     max_length=self.max_len, 
        #     padding=self.padding, 
        #     truncation=True,
        #     return_tensors=self.return_tensors,
        # )
        
        labels = student_inputs["input_ids"].clone().detach()
        input_lengths = student_inputs["attention_mask"].sum(dim=1)
        # labels[labels == self.student_tokenizer.pad_token_id] = -100
        # prompt_lengths = tokenized_prompts["attention_mask"].sum(dim=1)
        prompt_lengths = torch.tensor(prompt_lengths)

        for i in range(len(labels)):
            labels[i, :prompt_lengths[i]] = -100
            labels[i, (input_lengths[i] + 1):] = -100

        if not self.do_train:
            return student_inputs, None, labels

        student_token_offset_mapping = student_inputs.pop('offset_mapping')

        teacher_inputs = self.teacher_tokenizer(
            fulls,
            truncation=True,
            padding=self.padding,
            max_length=self.max_len,
            return_tensors=self.return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_offsets_mapping=self.return_offsets_mapping,
            add_special_tokens=False
        )
        # teacher_prompts = self.teacher_tokenizer(
        #     prompts, 
        #     max_length=self.max_len, 
        #     padding=self.padding, 
        #     truncation=True,
        #     return_tensors=self.return_tensors,
        # )
        # teacher_prompt_lengths = teacher_prompts["attention_mask"].sum(dim=1)
        teacher_prompt_lengths = torch.tensor(teacher_prompt_lengths)

        teacher_token_offset_mapping = teacher_inputs.pop('offset_mapping')

        student_pooler_tensor, teacher_pooler_tensor = prepare_pooler_v2(prompt_lengths,
                                                                         student_token_offset_mapping,
                                                                         teacher_prompt_lengths,
                                                                         teacher_token_offset_mapping,)

        student_inputs['pooler_safe_idx'] = student_pooler_tensor['safe_idx']
        student_inputs['pooler_mask'] = student_pooler_tensor['mask']
        teacher_inputs['pooler_safe_idx'] = teacher_pooler_tensor['safe_idx']
        teacher_inputs['pooler_mask'] = teacher_pooler_tensor['mask']

        return student_inputs, teacher_inputs, labels

