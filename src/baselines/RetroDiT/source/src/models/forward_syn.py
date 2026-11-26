
import torch

from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.utils.load_hugging_face_model import load_model

__all__ = ['ForwardSyn']

class ForwardSyn:
    def __new__(
            cls,
            model_name: str = None,
            batch_size: int = 32,
            pretrain_path: str = None
        ):
        if model_name is None:
            return None
        instance = super().__new__(cls)
        return instance

    def __init__(
            self,
            model_name: str,
            batch_size: int = 32,
            pretrain_path: str = None,
        ):
        self.tokenizer = load_model(
            model_name, AutoTokenizer, pretrain_path
        )
        self.model = load_model(
            model_name, AutoModelForSeq2SeqLM, pretrain_path
        )
        self.batch_size = batch_size
        self.device = None
        self.model.eval()
    
    def to(self, device):
        if self.device != device:
            self.device = device
            self.model.to(device)
        return self

    def predict(
            self,
            reactants_list: list[str],
            max_length: int = 512,
            num_beams: int = 1
        ) -> list[str]:
        assert self.device is not None, "Must call .to(device) before predict"
        predictions = []

        reactants_list_ = []
        for sample in reactants_list:
            if sample is None:
                reactants_list_.append('')
            else:
                reactants_list_.append(sample)
        
        for i in tqdm(
            range(0, len(reactants_list_), self.batch_size),
            desc='forward prediction'
        ):
            batch = reactants_list_[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_length=max_length, 
                    num_beams=num_beams
                )
            
            batch_predictions = [
                self.tokenizer.decode(output, skip_special_tokens=True).replace(" ", "")
                for output in outputs
            ]
            predictions.extend(batch_predictions)
        
        return predictions


if __name__ == "__main__":
    predictor = ForwardSyn(
        "sagawa/ReactionT5v2-forward",
        batch_size=16
    )
    predictor.to('cuda:0')
    
    reactants_list = [
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O.CCN(CC)CC",
        "c1ccc(cc1)C=O.c1ccccc1",
        "CC(=O)O.CCO"
    ]
    
    
    products = predictor.predict(reactants_list)
    
    for reactants, product in zip(reactants_list, products):
        print(f"Reactants: {reactants}")
        print(f"Product:   {product}\n")