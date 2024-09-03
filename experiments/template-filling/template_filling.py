from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import typer
import evaluate
import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from experiments.evaluator import ModelsEnum, LLMEvaluator  # noqa: E402

app = typer.Typer()

class TemplateFillingEvaluator(LLMEvaluator):

    german_system_prompt = """
Du bist ein System zur Unterstützung bei der Analyse großer Textmengen. Du wirst dem Nutzer helfen, alle Fragen korrekt zu beantworten.
"""

    german_prompt = """
Bitte extrahiere die Antwort auf die folgende Frage aus dem untenstehenden Kontext:

Kontext: {}
Frage: {}

Bitte antworte in diesem Format. Wenn die Frage nicht mit dem Kontext beantwortet werden kann MUSST du mit 'Nicht beantwortbar' reagieren. Du musst keine Begründung angeben.
Antwort: <Antwort> ODER <Nicht beantwortbar>
Begründung: <Begründung>

Denke daran, die Antwort MUSS wörtlich aus dem Text extrahiert werden. Du musst die Antwort aus dem Text extrahieren (falls möglich), nicht generieren!
"""

    english_system_prompt = """
You are a system to support the analysis of large amounts of text. You will assist the user by extracting the required information from the provided documents. You will always answer in the required format and use no other formatting than expected by the user!    
"""

    def __init__(self, dataset: pd.DataFrame, slots: List[str], prompt, model: ModelsEnum, port: int, lang: str, dataset_name: str, task_name: str, output_dir_path: Path, report_path: Path):
        # call parent
        super(TemplateFillingEvaluator, self).__init__(model=model, port=port, lang=lang, dataset_name=dataset_name, task_name=task_name, output_dir_path=output_dir_path, report_path=report_path)

        # ensure that slots are dataset keys 
        assert all(slot in dataset.columns for slot in slots)
        self.slots = slots
        self.dataset = dataset

        self.prompt = prompt
        if lang == "de":
            self.system_prompt = self.german_system_prompt.strip()
        elif lang == "en":
            self.system_prompt = self.english_system_prompt.strip()
        else:
            raise ValueError("Language not supported. Please choose 'de' or 'en'.")

    def _parse_response(self, response: str) -> Dict[str, str]:
        result: Dict[str, str] = {
            slot: "" for slot in self.slots
        }

        for line in response.strip().split("\n"):
            if not line.strip():
                continue
            if ":" not in line:
                continue

            splitted_line = line.split(":")
            if len(splitted_line) != 2:
                continue
            
            slot = splitted_line[0].strip()
            answer = splitted_line[1].strip()

            if slot.startswith("<"):
                slot = slot[1:]
            if slot.endswith(">"):
                slot = slot[:-1]

            if slot.startswith("**"):
                slot = slot[2:]
            if slot.endswith("**"):
                slot = slot[:-2]

            if slot.lower() not in result:
                continue

            if answer.lower() == "none":
                continue

            answer = answer.strip().lower()

            result[slot.lower()] = answer

        return result

    def _prompt_ollama(self, text: str) -> Tuple[str, Dict[str, str]]:
        response = self.client.chat(model=self.model, messages=[
            {
                'role': 'system',
                'content': self.system_prompt,
            },
            {
                'role': 'user',
                'content': self.prompt.format(text).strip(),
            },
        ])
        
        message = response["message"]["content"]
        prediction = self._parse_response(message)
        return message, prediction

    def _evaluate(self):
        golds = []
        preds = []
        messages = []   
        for idx, sample in tqdm(self.dataset.iterrows(), desc="Evaluating"):
            gold = {
                slot: sample[slot].tolist() for slot in self.slots
            }

            message, prediction = self._prompt_ollama(text=sample["doctext"])

            golds.append(gold)
            preds.append(prediction)
            messages.append(message)

        # store the evaluation results
        pd.DataFrame({
            "Gold": golds,
            "Prediction": preds,
            "Message": messages
        }).to_parquet(self.output_file_path.with_suffix(".parquet"))

    def _transform_to_squad(self, data: List[Dict[str, List[str]]], is_gold: bool) -> Dict[str, List[Dict[str, str]]]:
        transformed = {
            slot: [] for slot in self.slots
        }

        for idx, datapoint in enumerate(data):
            for slot in self.slots:
                assert slot in datapoint

                if is_gold:
                    transformed[slot].append({'answers': {"answer_start": [0], "text": datapoint[slot]}, "id": str(idx)})
                else:
                    has_answer = datapoint[slot] is not None and len(datapoint[slot]) > 0
                    transformed[slot].append({'prediction_text': datapoint[slot][0] if has_answer else '', 'id': str(idx), 'no_answer_probability': 0.0 if has_answer else 1.0})

        return transformed

    def _report(self):
        # load the predictions
        df = pd.read_parquet(self.output_file_path.with_suffix(".parquet"))
        golds = df["Gold"].to_list()
        preds = df["Prediction"].to_list()

        squad_v2_metric = evaluate.load("squad_v2")

        gold_transformed = self._transform_to_squad(golds, is_gold=True)
        pred_transformed = self._transform_to_squad(preds, is_gold=False)

        ems = []
        f1s = []
        for slot in self.slots:
            assert len(gold_transformed[slot]) == len(pred_transformed[slot])
            print(f"Slot: {slot}")
            results = squad_v2_metric.compute(references=gold_transformed[slot], predictions=pred_transformed[slot])
            print(results)
            if results is None:
                raise ValueError("The evaluation results are empty.")
            ems.append(results["exact"])
            f1s.append(results["f1"])
        em = round(sum(ems) / len(ems), 2)
        f1 = round(sum(f1s) / len(f1s), 2)

        # write reports
        self._add_results_to_report({
            "Exact Match": em,
            "F1": f1,
        })

@app.command()
def muc4(model: ModelsEnum, port: int, report_only: bool = False):
    # load dataset
    df = pd.read_parquet("datasets/muc/muc.parquet")

    slots = ["incident", "perpetrator", "group perpetrator", "victim", "target", "weapon"]

    # prompts
    user_prompt = """
I want you to extract the following information about incidents from the text below. The slots are:

Incident: One of 'Arson', 'Attack', 'Bombing', 'Kidnapping', 'Robbery'
Perpetrator: An individual perpetrator
Group Perpetrator: A group or organizational perpetrator
Victim: Sentient victims of the incident
Target: Physical objects targeted by the incident
Weapon: Weapons employed by the perpetrators

Please extract the information about the incident (if any) from the following text:
{}

Respond in the following format:
Incident: <incident type>
Perpetrator: <perpetrator>
Group Perpetrator: <group perpetrator>
Victim: <victim>
Target: <target>
Weapon: <weapon>

e.g.
Incident: Arson
Perpetrator: John Doe
Group Perpetrator: None
Victim: None
Target: Building
Weapon: Matches

If there is no information about a certain slot in the provided text, leave it empty with "None".
Also, if there is no incident in the text, you have to leave the rest of the slots empty.

Remember, you MUST extract the information verbatim from the text, do not generate it!
"""

    # start evaluator
    evaluator = TemplateFillingEvaluator(
        model=model,
        port=port,
        lang="en",
        task_name="template-filling",
        dataset_name="MUC4",
        output_dir_path=Path("experiments/template-filling/results/"),
        report_path=Path("experiments/template-filling/report.csv"),
        dataset=df,
        slots=slots,
        prompt=user_prompt,
    )
    evaluator.start(report_only=report_only)
    
@app.command()
def test():
    print("Hello, World!")

if __name__ == "__main__":
    app()
