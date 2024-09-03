from enum import Enum
from typing import Dict, Union
import pandas as pd
from ollama import Client
from pathlib import Path

MODELS = ["llama3.1:8b-instruct-fp16", "mistral-nemo:12b-instruct-2407-fp16", "gemma2:9b-instruct-fp16"]

class ModelsEnum(str, Enum): 
    llama = "llama3"
    mistral = "mistral"
    gemma = "gemma2"

model_dict: Dict[ModelsEnum, str] = {
    ModelsEnum.llama: "llama3.1:8b-instruct-fp16",
    ModelsEnum.mistral: "mistral-nemo:12b-instruct-2407-fp16",
    ModelsEnum.gemma: "gemma2:9b-instruct-fp16"
}


class LLMEvaluator():

    def __init__(self, model: ModelsEnum, port: int, lang: str, dataset_name: str, task_name: str, output_dir_path: Path, report_path: Path):

        assert len(model) > 0, "Model name must not be empty."
        assert len(lang) > 0, "Language must not be empty."
        assert lang in ["de", "en"], "Language must be 'de' or 'en'."
        assert len(dataset_name) > 0, "Dataset name must not be empty."
        assert output_dir_path is not None, "Output dir path must not be None."
        assert report_path is not None, "Report path must not be None."

        self.model = model_dict[model]
        self.model_name = model.value

        # ensure that the requested model is available
        self.client = Client(host=f'http://localhost:{port}')
        available_models = [x["name"] for x in self.client.list()["models"]]
        if self.model not in available_models:
            print(f"Model {self.model} is not available. Pulling it now.")
            self.client.pull(self.model)
            print(f"Model {self.model} has been pulled successfully.")
        available_models = [x["name"] for x in self.client.list()["models"]]

        assert self.model in available_models, f"Model {self.model} is not available. Available models are: {available_models}"

        # ensure that output dir exists
        if not output_dir_path.exists():
            print(f"Output directory {output_dir_path} does not exist. Creating it now.")
            output_dir_path.mkdir(parents=True, exist_ok=True)

        self.language = lang
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.output_file_path = output_dir_path / f"{self.model_name}_{dataset_name}_{task_name}.csv"
        self.report_path = report_path

        print ("----------------------------------------")
        print(f"Experiment Ready: Model={self.model_name}, Language={self.language}, Dataset={self.dataset_name}, Task={self.task_name}")

    def _add_results_to_report(self, results: Dict[str, Union[str, float]]):
        # read existing report, or create new one
        if self.report_path.exists():
            df_current = pd.read_csv(self.report_path)
        else:
            df_current = pd.DataFrame(columns=["Model", "Dataset", "Language", "Task"])
        
        df_new = pd.DataFrame({
            "Model": [self.model],
            "Dataset": [self.dataset_name],
            "Language": [self.language],
            "Task": [self.task_name],
            **{k: [v] for k, v in results.items()}
        })
        df = pd.concat([df_current, df_new])
        df.to_csv(self.report_path, index=False)
        
    def _evaluate(self):
        raise NotImplementedError("This method must be implemented in a subclass.")

    def _report(self):
        raise NotImplementedError("This method must be implemented in a subclass.")
    
    def start(self, report_only: bool = False):
        if not report_only:
            self._evaluate()
        self._report()