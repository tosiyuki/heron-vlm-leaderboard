import torch
import transformers
from PIL import Image
from typing import Dict, Any
from plugins.base_adapter import BaseAdapter
from transformers import AutoProcessor, LlamaTokenizer
import logging

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.llava_gpt2 import LlavaGpt2ForCausalLM
from llava.train.arguments_dataclass import ModelArguments, DataArguments, TrainingArguments
from llava.train.dataset import tokenizer_image_token

class LlavaJpAdapter(BaseAdapter):
    dependencies = [
        'torch>=1.9.0',
        'transformers>=4.37.2',
        'pillow>=8.0.0',
        'open_clip_torch>=2.26.1',
    ]

    def __init__(self, model_name: str, device: str, config: Dict[str, Any]):
        super().__init__(model_name, device)
        self.config = config
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = device
        self.conv_mode = "v1"
        self._initialize_model()

    def _initialize_model(self):
        try:
            self.model = LlavaGpt2ForCausalLM.from_pretrained(
                self.model_name, 
                low_cpu_mem_usage=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager"
            )
            self.model.eval()
            self.model.to(self.device)

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name,
                model_max_length=1532,
                padding_side="right",
                use_fast=False,
            )
            self.image_size = self.model.get_model().vision_tower.image_processor.size["height"]
            if self.model.get_model().vision_tower.scales is not None:
                self.image_size = self.model.get_model().vision_tower.image_processor.size["height"] * len(self.model.get_model().vision_tower.scales)
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise

    @classmethod
    def supports_model(cls, model_name: str) -> bool:
        return model_name.startswith("toshi456/llava-jp-1.3b")

    async def generate_response(self, question: str, image_path: str) -> str:
        try:
            # image preprocess
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.model.get_model().vision_tower.image_processor(
                image, 
                return_tensors='pt', 
                size={"height": self.image_size, "width": self.image_size}
            )['pixel_values'].half().cuda().to(torch.bfloat16)

            # token preprocess
            inp = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            inputs = tokenizer_image_token(
                prompt, 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0)
            inputs = inputs.to(self.device)
            inputs = inputs[:, :-1]

            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs=inputs,
                    images=image_tensor,
                    do_sample=False,
                    temperature=0.,
                    max_length=256,
                    no_repeat_ngram_size=2,
                )
            
            input_token_len = inputs.shape[1]
            n_diff_input_output = (inputs != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            response = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            response = response.strip()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            if response.endswith(stop_str):
                response = response[:-len(stop_str)]
            return response.strip()
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "Error generating response"

    async def verify(self) -> bool:
        try:
            test_question = "What is in this image?"
            test_image_path = "test.jpg"
            response = await self.generate_response(test_question, test_image_path)
            return len(response) > 0
        except Exception as e:
            logging.error(f"Verification failed: {str(e)}")
            return False

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "max_length": int,
            "do_sample": bool,
            "temperature": float,
            "no_repeat_ngram_size": int,
        }

def register_plugin(manager):
    manager.register_adapter("llava_jp", LlavaJpAdapter)