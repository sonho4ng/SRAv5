from dataclasses import dataclass, field
from typing import List, Optional, Any
import os



@dataclass
class Arguments:
    train_data: str = field(
        default=None, metadata={"help": "Path to training data"}
    )
    val_data: str = field(
        default=None, metadata={"help": "Path to validation data"}
    )
    test_data: str = field(
        default=None, metadata={"help": "Path to test data"}
    )

    num_labels: int = field(default=2, metadata={"help": "Number of labels"})

    batch_size: int = field(default=8)
    val_batch_size: int = field(default=32)


    max_len: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    query_max_len: int = field(
        default=64,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    passage_max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    pad_to_multiple_of: int = field(default=2, metadata={"help": ""})

    temperature: Optional[float] = field(default=0.02)
    distill_temperature: Optional[float] = field(default=2.0)
    sentence_pooling_method: str = field(default='cls', metadata={"help": "the pooling method, should be cls or mean"})
    normlized: bool = field(default=True)
    
    knowledge_distillation: bool = field(default=True, metadata={"help": "Use knowledge distillation"})
    finetune_hidden_states: bool = field(default=True)
    output_attentions: bool = field(default=True)
    
    teach_device: str = field(default='cuda:1')
    student_device: str = field(default='cuda:0')

    num_train_epochs: int = field(default=1)

    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.03)

    teacher_loss_weight: float = field(default=0.5)
    embedding_loss_weight: float = field(default=1)
    orthogonal_loss_weight: float = field(default=1000)
    hard_label_loss_weight: float = field(default=1.0)
    loss_type: str = field(default='ce')
    vector_embedding_warmup_ratio: float = field(default=0.0)


    teacher_layers_mapping: List[int] = field(default=list)
    student_encoder_layers_finetuned: List[int] = field(default=list)
    n_encoder_finetuned: int = field(default=6)
    finetune_embedding: bool = field(default=False)

    orthogonal: bool = field(default=True)
    span_loss: bool = field(default=True)
    der_loss: bool = field(default=True)

    span_weight_pooling: bool = field(default=True)
    span_loss_weight: bool = field(default=True)

    p: float = field(default=1.0)

    hidden_loss_weights: List[float] = field(default=None)
    teacher_embedding_dimension: int = field(default=1024)


    output_dir: Optional[str] = field(default=None, metadata={"help": "Where to store the final model"})

    teacher_model: str = field(default='')
    teacher_tokenizer: str = field(default='')
    student_model: str = field(default='google-bert/bert-base-uncased')
    student_tokenizer: str = field(default='google-bert/bert-base-uncased')
    hf_token: str = field(default='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')

    load_student_tokenizer_kwargs: dict = field(default_factory=dict)
    load_teacher_tokenizer_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")
            
        if len(self.teacher_layers_mapping) != len(self.student_encoder_layers_finetuned):
            raise ValueError("teacher_layers_mapping and student_encoder_layers_finetuned should have the same length")

